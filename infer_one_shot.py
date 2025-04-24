import torch
from dataclasses import dataclass, field
from einops import rearrange
import os
from torch.utils.data import DataLoader
import tgs
from tgs.models.image_feature import ImageFeature
from tgs.utils.saving import SaverMixin
from tgs.utils.config import parse_structured
from tgs.utils.ops import points_projection_my, points_projection
from tgs.utils.misc import load_module_weights
from tgs.utils.typing import *
from tgs.models.code_attn import code_attn
from tgs.utils.ops import scale_tensor
import torch.nn.functional as F
from utils import compute_error, VGGLoss
import random

import cv2
import numpy as np
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning import Trainer, loggers
import copy
import evaluator
# from .DINAR.discriminators.style_gan_v2 import Discriminator
import yaml
import config
from spatial import SpatialEncoder
from livehand.input_encoder import read_mano_uv_obj, save_obj_for_debugging, get_uvd
from pytorch3d.ops import knn_points, knn_gather
import trimesh
import time

from tgs.models.verts_refinement import additional_features_fc


class TGS(torch.nn.Module, SaverMixin):
    @dataclass
    class Config:
        radius_texture: float = 1.0
        weights: Optional[str] = None
        weights_ignore_modules: Optional[List[str]] = None

        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        pose_embedder_cls: str = ""
        pose_embedder: dict = field(default_factory=dict)

        image_feature: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_shade_cls: str = ""
        tokenizer_shade: dict = field(default_factory=dict)

        tokenizer_texture_cls: str = ""
        tokenizer_texture: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        backbone_shade_cls: str = ""
        backbone_shade: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        post_processor_texture_cls: str = ""
        post_processor_texture: dict = field(default_factory=dict)

        renderer_cls: str = ""
        renderer: dict = field(default_factory=dict)

        pointcloud_generator_cls: str = ""
        pointcloud_generator: dict = field(default_factory=dict)

        pointcloud_encoder_shade_cls: str = ""
        pointcloud_encoder_shade: dict = field(default_factory=dict)

        pointcloud_encoder_texture_cls: str = ""
        pointcloud_encoder_texture: dict = field(default_factory=dict)
    
    cfg: Config

    def load_weights(self, weights: str, ignore_modules: Optional[List[str]] = None):
        state_dict = load_module_weights(
            weights, ignore_modules=ignore_modules, map_location="cpu"
        )
        self.load_state_dict(state_dict, strict=False)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self._save_dir: Optional[str] = None

        assert self.cfg.camera_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.camera_embedder.pop("weights") if "weights" in self.cfg.camera_embedder else None
        self.camera_embedder = tgs.find(self.cfg.camera_embedder_cls)(**self.cfg.camera_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.camera_embedder.load_state_dict(state_dict)

        assert self.cfg.pose_embedder_cls == 'tgs.models.networks.MLP'
        weights = self.cfg.pose_embedder.pop("weights") if "weights" in self.cfg.pose_embedder else None
        self.pose_embedder = tgs.find(self.cfg.pose_embedder_cls)(**self.cfg.pose_embedder)
        if weights:
            from tgs.utils.misc import load_module_weights
            weights_path, module_name = weights.split(":")
            state_dict = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            self.pose_embedder.load_state_dict(state_dict)

        self.image_feature = ImageFeature(self.cfg.image_feature)

        self.tokenizer_shade = tgs.find(self.cfg.tokenizer_shade_cls)(self.cfg.tokenizer_shade)
        self.tokenizer_texture = tgs.find(self.cfg.tokenizer_texture_cls)(self.cfg.tokenizer_texture)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.backbone_shade = tgs.find(self.cfg.backbone_shade_cls)(self.cfg.backbone_shade)


        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.post_processor_texture = tgs.find(self.cfg.post_processor_texture_cls)(
            self.cfg.post_processor_texture
        )

        self.renderer = tgs.find(self.cfg.renderer_cls)(self.cfg.renderer)

        # pointcloud generator
        self.pointcloud_generator = tgs.find(self.cfg.pointcloud_generator_cls)(self.cfg.pointcloud_generator)

        self.point_encoder_shade = tgs.find(self.cfg.pointcloud_encoder_shade_cls)(self.cfg.pointcloud_encoder_shade)
        self.point_encoder_texture = tgs.find(self.cfg.pointcloud_encoder_texture_cls)(self.cfg.pointcloud_encoder_texture)


        self.identity_code_book = torch.nn.Parameter(torch.clamp(torch.normal(mean=0.0, std=0.02, size=(27, 1, 33, 64, 128)), -1, 1))
        self.identity_code_one_shot = torch.nn.Parameter(torch.zeros(size=(1, 1, 33, 64, 128)))

        self.sp_encoder = SpatialEncoder(sp_level = 4)
        self.additional_features_fc = additional_features_fc(852,51)

        self.map_bias = torch.nn.Parameter(torch.zeros(size=(80, 64, 128)))

        self.color_w = torch.nn.Parameter(torch.ones(size=(3*16,)))
        self.color_b = torch.nn.Parameter(torch.zeros(size=(3*16,1024,2048)))
        self.xyz_b = torch.nn.Parameter(torch.zeros(size=(3,)))
        self.xyz_b_map = torch.nn.Parameter(torch.zeros(size=(3,1024,2048)))
        self.opacity_b = torch.nn.Parameter(torch.zeros(size=(1,1024,2048)))

        # load checkpoint
        if self.cfg.weights is not None:
            self.load_weights(self.cfg.weights, self.cfg.weights_ignore_modules)
    
    def query_triplane_texture(
        self,
        positions: Float[Tensor, "*B N 2"],
        triplanes: Float[Tensor, "*B 1 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
       
        positions = scale_tensor(positions, (-self.cfg.radius_texture, self.cfg.radius_texture), (-1, 1))
        indices2D: Float[Tensor, "B N 2"] = positions[:, :, None]

        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            triplanes.squeeze(1),
            indices2D,
            align_corners=True,
            mode="bilinear",
        )

        out = out.view(*out.shape[:2], -1).permute(0, 2, 1)
        if not batched:
            out = out.squeeze(0)

        return out

    def _forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:

        batch_size, n_input_views = batch["rgb_cond"].shape[:2]
        pointclouds = batch["points"]
        t_point = batch["ret"]["targets"]['mesh_t']

        n_points = pointclouds.shape[1]
        capture_id = batch['ret']['human_idx']
        identity_code = self.identity_code_one_shot.squeeze(0)
        out={"points":pointclouds}

        # Camera modulation
        camera_extri = batch["c2w_cond"].view(*batch["c2w_cond"].shape[:-2], -1)
        camera_intri = batch["intrinsic_normed_cond"].view(*batch["intrinsic_normed_cond"].shape[:-2], -1)

        camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
        camera_feats = self.camera_embedder(camera_feats)

        pose = batch['ret']['targets']['input_mano_pose'].view(batch_size,1,-1)
        pose_feats = self.pose_embedder(pose)

        c2w_cond = batch["c2w_cond"].squeeze(1)
        w2c_cond = batch["w2c_cond"].squeeze(1)
        intrinsic_cond = batch["intrinsic_cond"].squeeze(1)
        face = batch['ret']['targets']['face_world']

        targets = batch['ret']['targets']
        vert3d_uv=targets['vert_uv']
        face_uv=targets['face_uv'].long()[0]
        face_uv_xy=targets['face_uv_xy'][0]

        verts_uv,verts_d=[],[]
        for b in range(batch_size):
            vert_uv, vert_d, intermediates_vert = get_uvd(pointclouds[b], vert3d_uv[0], face_uv, face_uv_xy)
            vert_uv = vert_uv.unsqueeze(0)
            vert_d = vert_d.unsqueeze(0)
            verts_uv.append(vert_uv)
            verts_d.append(vert_d)
        vert_uv=torch.cat(verts_uv,0)
        vert_d=torch.cat(verts_d,0)

        # normalize it to [-1, 1]
        vert_uv[..., 0] = 2.0 * (vert_uv[..., 0] /1) - 1.0
        vert_uv[..., 1] = 2.0 * (vert_uv[..., 1] /0.5) - 1.0
        
        vert_uv_sp = self.sp_encoder(vert_uv)
        pointclouds_sp = self.sp_encoder(pointclouds)
        identity_code_vert = self.query_triplane_texture(vert_uv, self.identity_code_one_shot.repeat(batch_size,1,1,1,1))
        point_cond_embeddings_texture = self.point_encoder_texture(torch.cat([vert_uv, vert_uv_sp, identity_code_vert], dim=-1))
        

        _,mink_idxs_world,_=knn_points(pointclouds,pointclouds,K=100)
        _,mink_idxs_texture,_=knn_points(t_point,t_point,K=100)
        mink_idxs_inter=(mink_idxs_world == mink_idxs_texture).sum(-1)<10
        mink_idxs_inter = mink_idxs_inter.unsqueeze(-1)
        point_cond_embeddings_shade = self.point_encoder_shade(torch.cat([vert_uv, vert_uv_sp, pointclouds, pointclouds_sp, mink_idxs_inter, pose_feats.repeat(1,n_points,1), camera_feats.repeat(1,n_points,1)], dim=-1))

        tokens_texture: Float[Tensor, "B Ct Nt"] = self.tokenizer_texture(batch_size, cond_embeddings=point_cond_embeddings_texture)
        tokens_shade: Float[Tensor, "B Ct Nt"] = self.tokenizer_shade(batch_size, cond_embeddings=point_cond_embeddings_shade)

        tokens_texture = self.backbone(
            tokens_texture,
            modulation_cond=identity_code,
        )

        tokens_shade = self.backbone_shade(
            tokens_shade,
            modulation_cond=torch.cat([pose_feats, camera_feats],dim=-1),
        )

        tokens_texture = tokens_texture + tokens_shade

        scene_codes_texture = self.tokenizer_texture.detokenize(tokens_texture)
        scene_codes_texture = self.post_processor_texture(scene_codes_texture)
        scene_codes_texture = torch.cat([scene_codes_texture[:,0], scene_codes_texture[:,1]], dim=-1).unsqueeze(1)
        scene_codes_texture = scene_codes_texture + torch.cat([self.map_bias[...,:64],self.map_bias[...,:64]],dim=-1)

        additional_features = torch.cat([vert_uv, vert_uv_sp, pointclouds, pointclouds_sp, mink_idxs_inter, identity_code_vert, pose_feats.repeat(1,n_points,1)], dim=-1)
        additional_features = self.additional_features_fc(additional_features)

        rend_out = self.renderer(scene_codes_texture=scene_codes_texture,
                                vert_uv=vert_uv,
                                mink_idxs_inter = mink_idxs_inter,
                                color_w = self.color_w,
                                query_points=pointclouds,
                                query_points_tar=batch["points_tar"],
                                additional_features=additional_features,
                                height=256,
                                width=256,
                                intrinsic_input = batch["intrinsic_cond"],
                                w2c_input = batch["w2c_cond"],
                                vert3d_uv=vert3d_uv, 
                                face_uv=face_uv, 
                                face_uv_xy=face_uv_xy,
                                **batch)
        return {**out, **rend_out}
    
    def forward(self, batch):
        out = self._forward(batch)
        batch_size = batch["index"].shape[0]
        for b in range(batch_size):
            if batch["view_index"][b, 0] == 0:
                out["3dgs"][b].save_ply(self.get_save_path(f"3dgs/{batch['instance_id'][b]}.ply"))

            for index, render_image in enumerate(out["comp_rgb"][b]):
                view_index = batch["view_index"][b, index]
                self.save_image_grid(
                    f"video/{batch['instance_id'][b]}/{view_index}.png",
                    [
                        {
                            "type": "rgb",
                            "img": render_image,
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                )
        

class HandLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, cfg: dict, cfg_model: dict, TGS_cfg: dict):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.kwargs = self.cfg['models']['VANeRF']
        self.tgs_cfg = copy.deepcopy(TGS_cfg)
        self.cfg_model = cfg_model
        self.idx = 0
        self.expname = cfg['expname']
        self.save_dir = f'{cfg["out_dir"]}/{cfg["expname"]}'
        self.save_hyperparameters()
        self.dataset = Dataset
        self.video_dirname = 'video'
        self.images_dirname = 'images'
        self.test_dst_name = cfg['test_dst_name']
        # self.nkpt_r,self.nkpt_l=778,778
        self.nkpt_r,self.nkpt_l=21,21
        self.model = TGS(self.tgs_cfg.system)
        self.model.set_save_dir("outputs")
        # self.discriminator=Discriminator(image_size=cfg['models']['Discriminator']['params']['image_size'],activation_layer=cfg['models']['Discriminator']['params']['activation_layer'], channel_multiplier=cfg['models']['Discriminator']['params']['channel_multiplier'])
        self.vgg_loss = VGGLoss()
        self.evaluator = evaluator.Evaluator()
        # load checkpoint
        self.pretrained_path = './EXPERIMENTS/pretrain_model.ckpt'
        pretrained_model = torch.load(self.pretrained_path)
        self.load_state_dict(pretrained_model['state_dict'], strict=False)
        for name, param in self.model.named_parameters():
            if 'map_bias' in name or 'color_w' in name or 'color_b' in name or 'opacity_b' in name or 'identity_code' in name:
                continue
            param.requires_grad = False

    def configure_optimizers(self):
        opt_g=torch.optim.Adam(self.model.parameters(), lr=self.cfg['training'].get('lr', 1e-5))
        StepLR_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[2,5,10,20,35,50,75], gamma=0.5)
        optim_dict_g = {'optimizer': opt_g, 'lr_scheduler': StepLR_g}
        return [optim_dict_g]

    @classmethod
    def from_config(cls, cfg_hand, cfg_model, cfg_tgs):
        return cls(cfg_hand, cfg_model, cfg_tgs)

    def train_dataloader(self, batch_size=None):
        train_dataset = self.dataset.from_config(self.cfg['dataset'], 'train', self.cfg)
        return torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            num_workers=self.cfg['training'].get('train_num_workers', 0),
            batch_size=self.cfg['training'].get('train_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self, batch_size=None):
        val_dataset = self.dataset.from_config(self.cfg['dataset'], 'val', self.cfg)
        return torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            num_workers=self.cfg['training'].get('val_num_workers', 0),
            batch_size=self.cfg['training'].get('val_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self, batch_size=None):
        test_dataset = self.dataset.from_config(self.cfg['dataset'], 'test', self.cfg)
        return torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            num_workers=self.cfg['training'].get('val_num_workers', 0),
            batch_size=self.cfg['training'].get('val_batch_size', 1) if batch_size is None else batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def save_ckpt(self, **kwargs):
        pass

    def test_epoch_end(self, outputs):
        results = {key: torch.stack([x[key] for x in outputs]).mean() for key in outputs[0].keys()}
        results = {key: float(val.item()) if torch.is_tensor(val) else float(val) for key, val in results.items()}

        path = os.path.join(
            self.save_dir,  f'test_{self.test_dst_name}_{self.current_epoch}_{self.global_step}.yml')

        with open(path, 'w') as f:
            yaml.dump(results, f)

        print('Results saved in', path)
        print(results)

    @staticmethod
    def collate_fn(items):
        """ Modified form of :func:`torch.utils.data.dataloader.default_collate` that will strip samples from
        the batch if they are ``None``.
        """
        try:
            items = [item for item in items if item is not None]
            return torch.utils.data.dataloader.default_collate(items) if len(items) > 0 else None
        except Exception as e:
            return None
    
    def load_ckpt(self, ckpt_path):
        assert os.path.exists(ckpt_path), f'Checkpoint ({ckpt_path}) does not exists!'
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["state_dict"])
        return ckpt['epoch'], ckpt['global_step']

    @staticmethod
    def compute_test_metric(rendered_img, gt_img, mask=None, max_val=1.0):
        """
        Args:
            rendered_img (torch.tensor): (3, H, W) or (B, 3, H, W) [0, 1.0]
            gt_img (torch.tensor): (3, H, W) or (B, 3, H, W) [0, 1.0]
            mask (torch.tensor: torch.bool): (3, H, W) or (B, 3, H, W) [0, 1.0]
        """
        assert rendered_img.shape == gt_img.shape
        if len(rendered_img.shape) == 3:
            rendered_img = rendered_img.unsqueeze(0)
            gt_img = gt_img.unsqueeze(0)
        mask = mask.view(1, *mask.shape[-2:]) if mask is not None else mask

        # B,3,H,W
        ssim = K.metrics.ssim(rendered_img, gt_img, window_size=7, max_val=max_val)
        ssim = ssim.permute(0, 2, 3, 1)[mask] if mask is not None else ssim
        ssim = ssim.mean()

        if mask is not None:
            rendered_img = rendered_img.permute(0, 2, 3, 1)[mask]
            gt_img = gt_img.permute(0, 2, 3, 1)[mask]

        return {
            f'psnr': K.metrics.psnr(rendered_img, gt_img, max_val=max_val),
            f'ssim': ssim,
        }

    def save_test_image(self, batch, rendered_img, gt_img, mask=None, face_mask=None):
        """
        Args:
            rendered_img (torch.tensor): (3, H, W) [0, 1.0]
        """
        index = batch['index']
        sub_id = index['frame'][0]
        tar_cam_id = index['tar_cam_id'][0]
        # prepare directory
        dst_dir = os.path.join(
            self.save_dir,
            f'{self.images_dirname}_{self.test_dst_name}',  # _{self.current_epoch}_{self.global_step}
            sub_id)
        cond_mkdir(dst_dir)

        # save images
        if rendered_img is not None:
            rendered_img = tensor_to_image(rendered_img)  # H,W,3
            rendered_img = (rendered_img*255.).astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.pred.png')
            cv2.imwrite(path, rendered_img[:, :, ::-1])

        if gt_img is not None:
            gt_img = tensor_to_image(gt_img)
            gt_img = (gt_img*255.).astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.gt.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])
        
        if mask is not None:
            mask = (mask*255.).squeeze().unsqueeze(-1).repeat(1, 1,  3)
            gt_img = mask.detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.mask.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])

        if face_mask is not None:
            face_mask = (face_mask*255.).squeeze().unsqueeze(-1).repeat(1, 1,  3)
            gt_img = face_mask.detach().cpu().numpy().astype(np.uint8)
            path = os.path.join(dst_dir, f'{tar_cam_id}.face_mask.png')
            cv2.imwrite(path, gt_img[:, :, ::-1])

    def training_step(self, batch, batch_idx):
        out = self.model._forward(batch)
        lambdas = self.kwargs.get("lambdas", {})
        expname = self.cfg.get('expname', 'ex_wild')
        os.makedirs(os.path.dirname("./vis/"+expname+"/"), exist_ok=True)
        if "comp_rgb_input" in out:
            out['input_img'] = batch['rgb_cond'].squeeze(1).permute(0, 3, 1, 2)
            out["tex_cal_fine_input"] = out["comp_rgb_input"].squeeze(1).permute(0, 3, 1, 2)
            out["alpha_fine_input"] = out['comp_mask_input'].float().mean(-1).unsqueeze(-1).squeeze(1).permute(0, 3, 1, 2)
            out["tar_alpha_input"] = batch['mask_cond'].float().squeeze(1).permute(0, 3, 1, 2)

            rgb_pred = out["tex_cal_fine_input"][0].permute(1, 2, 0).detach().cpu().numpy()*255
            cv2.imwrite("./vis/"+expname+"/rgb"+str(batch_idx)+"_pred.jpg", rgb_pred[...,[2,1,0]])
            # if batch['bbox_mask'] is not None:
            bbox_mask = batch['bbox_mask'].unsqueeze(1)
            if bbox_mask.shape[-1] ==3:
                bbox_mask = bbox_mask[...,0]
            out["tex_cal_fine_input"][bbox_mask.repeat(1,3,1,1) == 0] =0
            rgb_pred = out["tex_cal_fine_input"][0].permute(1, 2, 0).detach().cpu().numpy()*255

        rgb_input = batch['rgb_cond'].squeeze(1)[0].detach().cpu().numpy()*255
        cv2.imwrite("./vis/"+expname+"/rgb"+str(batch_idx)+"_ref.jpg", rgb_input[...,[2,1,0]])
        
        loss, err_dict = compute_error(out_nerf=out, vggloss=self.vgg_loss, lambdas=lambdas)

        color_w_loss = (self.model.color_w-1).pow(2.0)
        color_b_loss = (self.model.color_b).abs()
        opacity_b_loss = (self.model.opacity_b).pow(2.0)
        map_bias_loss = (self.model.map_bias).pow(2.0)
        color_loss = 100*color_b_loss.mean() + opacity_b_loss.mean()
        loss = loss + color_loss + 0.01*map_bias_loss.mean()

        frame_index = str(int(batch['ret']['frame_index'][0]))
        view_index = str(int(batch['ret']['cam_ind'][0]))

        return {'loss':loss}    


    def test_step(self, batch, batch_nb):
        self.evaluator.result_dir = os.path.join(
            self.save_dir,
            f'{self.images_dirname}_{self.test_dst_name}')

        out = self.model._forward(batch)
        out['tar_img'] = batch['rgb_cond'].squeeze(1).permute(0, 3, 1, 2)
        out["tex_cal_fine"] = out["comp_rgb_input"].squeeze(1).permute(0, 3, 1, 2)
        rendered_image = out["tex_cal_fine"]
        rgb_pred = rendered_image[0].permute(1, 2, 0).detach().cpu().numpy()*255
        human_idx = str(int(batch['ret']['human_idx'][0]))
        frame_index = str(int(batch['ret']['frame_index'][0]))
        view_index = str(int(batch['ret']['cam_ind'][0]))
        bbox_mask = batch['bbox_mask'].unsqueeze(1)
        if bbox_mask.shape[-1] ==3:
            bbox_mask = bbox_mask[...,0]
        out["tex_cal_fine"][bbox_mask.repeat(1,3,1,1) == 0] =0
        rendered_image = out["tex_cal_fine"]
        scores = self.evaluator.compute_score(
            rendered_image,
            out['tar_img'],
            input_imgs=batch['rgb_cond'].squeeze(1),
            mask_at_box=batch['ret']['mask_at_box'],
            human_idx=human_idx,
            frame_index=frame_index,
            view_index=view_index,
        )
        scores = {key: torch.tensor(val) for key, val in scores.items()}
        return scores


if __name__ == "__main__":
    import argparse
    import subprocess
    from tgs.utils.config import ExperimentConfig, load_config
    from dataset_one_shot import Dataset, load_cfg

    from tgs.utils.misc import todevice, get_device

    parser = argparse.ArgumentParser("Triplane Gaussian Splatting")
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument("--config_hand", default="vanerf_triplane.json", help="path to config file")
    parser.add_argument("--model_ckpt")
    parser.add_argument("--num_gpus", default=1)
    parser.add_argument("--out", default="outputs", help="path to output folder")
    parser.add_argument("--cam_dist", default=1.9, type=float, help="distance between camera center and scene center")
    parser.add_argument(
        "--run_val", action='store_true',
    )
    parser.add_argument(
        "--repose", action='store_true',
    )
    parser.add_argument(
        "--in_the_wild", action='store_true',
    )
    parser.add_argument("--image_preprocess", action="store_true", help="whether to segment the input image by rembg and SAM")
    args, extras = parser.parse_known_args()

    device = get_device()

    cfg: ExperimentConfig = load_config(args.config, cli_args=extras)
    from huggingface_hub import hf_hub_download

    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    
    cfg_hand = config.load_cfg(args.config_hand)
    cfg_hand['expname'] = cfg_hand.get('expname', 'default')
    config.save_config(os.path.join(cfg_hand['out_dir'], cfg_hand['expname']), cfg_hand)

    # create model
    model = HandLightningModule.from_config(cfg_hand, cfg_hand.get('method', None), cfg)
    
    val_key = cfg_hand["training"].get("model_selection_metric", 'val_PSNR')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{cfg_hand["out_dir"]}/{cfg_hand["expname"]}/ckpts/',
        filename='model-{epoch:04d}-{%s:.4f}' % val_key,
        verbose=True,
        monitor=val_key,
        mode=cfg_hand["training"].get("model_selection_mode", 'max'),
        save_top_k=-1,
        save_last=True,
        save_on_train_epoch_end=True,
    )
    last_ckpt = os.path.join(checkpoint_callback.dirpath, f"{checkpoint_callback.CHECKPOINT_NAME_LAST}.ckpt")
    if not os.path.exists(last_ckpt):
        last_ckpt = None
    if args.model_ckpt is not None:  # overwrite last ckpt if specified model path
        last_ckpt = args.model_ckpt

    resume_from_checkpoint = cfg_hand.get('resume_from_checkpoint', last_ckpt)

    # create trainer
    logger = loggers.TestTubeLogger(
        save_dir=cfg_hand["out_dir"],
        name=cfg_hand['expname'],
        debug=False,
        create_git_tag=False
    )
    trainer = Trainer(
        max_epochs=cfg_hand["training"]["max_epochs"],
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        resume_from_checkpoint=resume_from_checkpoint,
        logger=logger,
        gpus=args.num_gpus,
        num_sanity_val_steps=0,
        benchmark=True,
        detect_anomaly=True,
        # terminate_on_nan=False,
        accumulate_grad_batches=cfg_hand["training"].get("accumulate_grad_batches", 1),
        # fast_dev_run=args.fast_dev_run,
        strategy="ddp" if args.num_gpus != 1 else None,
        **cfg_hand["training"].get('pl_cfg', {})
    )

    # run training
    if args.run_val and not args.in_the_wild:
        trainer.test(model, ckpt_path=resume_from_checkpoint, verbose=True)
    elif args.run_val and args.in_the_wild:
        trainer.test(model_in_the_wild, ckpt_path=resume_from_checkpoint, verbose=True)
    else:
        trainer.fit(model)
        model.save_ckpt()
    
