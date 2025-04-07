from dataclasses import dataclass, field
import torch
from einops import rearrange

import tgs
from tgs.utils.base import BaseModule
from tgs.utils.typing import *

class SimplePointGenerator(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        camera_embedder_cls: str = ""
        camera_embedder: dict = field(default_factory=dict)

        image_tokenizer_cls: str = ""
        image_tokenizer: dict = field(default_factory=dict)

        tokenizer_cls: str = ""
        tokenizer: dict = field(default_factory=dict)

        backbone_cls: str = ""
        backbone: dict = field(default_factory=dict)

        post_processor_cls: str = ""
        post_processor: dict = field(default_factory=dict)

        pointcloud_upsampling_cls: str = ""
        pointcloud_upsampling: dict = field(default_factory=dict)

        flip_c2w_cond: bool = True

    cfg: Config

    def configure(self) -> None:
        super().configure()

        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(
            self.cfg.image_tokenizer
        )

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

        self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)

        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)

        self.post_processor = tgs.find(self.cfg.post_processor_cls)(
            self.cfg.post_processor
        )

        self.pointcloud_upsampling = tgs.find(self.cfg.pointcloud_upsampling_cls)(self.cfg.pointcloud_upsampling)

    def forward(self, batch, encoder_hidden_states=None, **kwargs):
        batch_size, n_input_views = batch["rgb_cond"].shape[:2]

        if encoder_hidden_states is None:
            # Camera modulation
            c2w_cond = batch["c2w_cond"].clone()
            w2c_cond = batch["w2c_cond"].clone()
            if self.cfg.flip_c2w_cond:
                c2w_cond[..., :3, 1:3] *= -1
            camera_extri = c2w_cond.view(*c2w_cond.shape[:-2], -1)
            camera_intri = batch["intrinsic_normed_cond"].view(
                *batch["intrinsic_normed_cond"].shape[:-2], -1)
            camera_feats = torch.cat([camera_intri, camera_extri], dim=-1)
            # camera_feats = rearrange(camera_feats, 'B Nv C -> (B Nv) C')

            camera_feats = self.camera_embedder(camera_feats)

            encoder_hidden_states: Float[Tensor, "B Cit Nit"] = self.image_tokenizer(
                rearrange(batch["rgb_cond"], 'B Nv H W C -> B Nv C H W'),
                modulation_cond=camera_feats,
            )
            encoder_hidden_states = rearrange(
                encoder_hidden_states, 'B Nv C Nt -> B (Nv Nt) C', Nv=n_input_views)

        tokens: Float[Tensor, "B Ct Nt"] = self.tokenizer(batch_size)

        tokens = self.backbone(
            tokens,
            encoder_hidden_states=encoder_hidden_states,
            modulation_cond=None,
        )
        # pointclouds = self.post_processor(self.tokenizer.detokenize(tokens))
        # print(pointclouds.shape) #[2, 2048, 3] [8, 1556, 3]
        pointclouds = batch["points"]
        print(pointclouds.shape)
        upsampling_input = {
            "input_image_tokens": encoder_hidden_states.permute(0, 2, 1),
            "input_image_tokens_global": encoder_hidden_states[:, :1],
            "c2w_cond": c2w_cond,
            "w2c_cond": w2c_cond,
            "rgb_cond": batch["rgb_cond"],
            "intrinsic_cond": batch["intrinsic_cond"],
            "intrinsic_normed_cond": batch["intrinsic_normed_cond"],
            "points": pointclouds.float(),
            "tar_cam":batch['ret']['targets']['tar_cam']
        }
        up_results = self.pointcloud_upsampling(upsampling_input)
        # print(up_results.shape) list
        up_results.insert(0, pointclouds)
        pointclouds = up_results[-1]
        print(pointclouds.shape)
        out = {
            "points": pointclouds,
            "up_results": up_results
        }
        return out