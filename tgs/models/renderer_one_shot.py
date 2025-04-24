from dataclasses import dataclass, field
from collections import defaultdict
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from tgs.utils.typing import *
from tgs.utils.base import BaseModule
from tgs.utils.ops import trunc_exp
from tgs.models.networks import MLP
from tgs.utils.ops import scale_tensor
from tgs.models.verts_refinement import vert_valid, vert_pos_refinement
from tgs.models.inter_attn import inter_attn
from tgs.models.self_attn import SelfAttn
from livehand.input_encoder import read_mano_uv_obj, save_obj_for_debugging, get_uvd

from einops import rearrange, reduce
import trimesh

inverse_sigmoid = lambda x: np.log(x / (1 - x))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height, width, znear, zfar, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)


        self.zfar = 1000.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.projection_matrix = getProjectionMatrix_refine(intrinsic, self.height, self.width, self.znear, self.zfar).transpose(0, 1).to(w2c.device)
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @staticmethod
    def from_w2c(w2c, intrinsic, height, width, znear, zfar):
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width, znear=znear, zfar=zfar)

class GaussianModel(NamedTuple):
    xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().numpy())
        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

class GSLayer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        in_channels: int = 128
        feature_channels: dict = field(default_factory=dict)
        xyz_offset: bool = True
        restrict_offset: bool = False
        use_rgb: bool = False
        clip_scaling: Optional[float] = None
        init_scaling: float = -5.0
        init_density: float = 0.1

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        self.out_layers = nn.ModuleList()
        for key, out_ch in self.cfg.feature_channels.items():
            if key == "shs" and self.cfg.use_rgb:
                out_ch = 3
            layer = nn.Linear(self.cfg.in_channels, out_ch)

            # initialize
            if not (key == "shs" and self.cfg.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, self.cfg.init_scaling)
            elif key == "rotation":
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                nn.init.constant_(layer.bias, inverse_sigmoid(self.cfg.init_density))

            self.out_layers.append(layer)

    def forward(self, x, pts):
        ret = {}
        for k, layer in zip(self.cfg.feature_channels.keys(), self.out_layers):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.cfg.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.cfg.clip_scaling)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.cfg.use_rgb:
                    v = torch.sigmoid(v)
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                if self.cfg.restrict_offset:
                    max_step = 1.2 / 32
                    v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pts if self.cfg.xyz_offset else pts
            ret[k] = v

        return GaussianModel(**ret)

class GS3DRenderer(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        mlp_network_config: Optional[dict] = None
        gs_out: dict = field(default_factory=dict)
        sh_degree: int = 3
        scaling_modifier: float = 1.0
        random_background: bool = False
        radius: float = 1.0
        radius_texture: float = 1.0
        feature_reduction: str = "concat"
        projection_feature_dim: int = 773
        background_color: Tuple[float, float, float] = field(
            default_factory=lambda: (1.0, 1.0, 1.0)
        )

    cfg: Config

    def configure(self, *args, **kwargs) -> None:
        if self.cfg.feature_reduction == "mean":
            mlp_in = 80
        elif self.cfg.feature_reduction == "concat":
            mlp_in = 80 * 3
        else:
            raise NotImplementedError
        mlp_in = 80+50+1
        if self.cfg.mlp_network_config is not None:
            self.mlp_net = MLP(mlp_in, self.cfg.gs_out.in_channels, **self.cfg.mlp_network_config)
        else:
            self.cfg.gs_out.in_channels = mlp_in
        self.gs_net = GSLayer(self.cfg.gs_out)
        self.gs_valid = vert_valid(verts_f_dim = mlp_in)
        # self.inter_attn = inter_attn(f_dim = mlp_in)
        self.self_attn_layer = SelfAttn(f_dim = mlp_in)
        self.vert_pos_refinement = vert_pos_refinement(verts_f_dim = mlp_in)
        self.threshold_low = 0.1
        self.threshold_high = 0.9

    def forward_gs(self, x, p):
        if self.cfg.mlp_network_config is not None:
            x = self.mlp_net(x)
        return self.gs_net(x, p)

    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ret_mask: bool = True,
        color_w = None,
        xyz_b = None,
        color_b = None,
        opacity_b = None,
        ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.cfg.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.cfg.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = gs.xyz

        if xyz_b is not None:
            means3D = means3D + xyz_b

        means2D = screenspace_points
        opacity = gs.opacity

        if opacity_b != None:
            opacity = opacity + opacity_b.view(-1,1)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.cfg.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
            if color_w is not None:
                colors_precomp = colors_precomp*color_w.view(-1,16,3)[:,0,:] + color_w.view(-1,16,3)[:,1,:] -1
                # colors_precomp = colors_precomp*color_w.view(-1,16,3)[:,2,:] + color_w.view(-1,16,3)[:,3,:] -1

            if color_b is not None:
                colors_precomp = colors_precomp+ color_b.view(-1,16,3)[:,0,:]
        else:
            shs = gs.shs
            if color_w is not None:
                shs = shs*color_w.view(-1,16,3)
            if color_b is not None:
                shs = shs*color_w.view(-1,16,3) + color_b.view(-1,16,3)

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            rendered_image, radii = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
        
        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),
            "comp_rgb_bg": bg_color
        }
        
        if ret_mask:
            mask_bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_camera.height),
                image_width=int(viewpoint_camera.width),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=mask_bg_color,
                scale_modifier=self.cfg.scaling_modifier,
                viewmatrix=viewpoint_camera.world_view_transform,
                projmatrix=viewpoint_camera.full_proj_transform.float(),
                sh_degree=0,
                campos=viewpoint_camera.camera_center,
                prefiltered=False,
                debug=False
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            with torch.autocast(device_type=self.device.type, dtype=torch.float32):
                rendered_mask, radii = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    colors_precomp = torch.ones_like(means3D),
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = cov3D_precomp)
                ret["comp_mask"] = rendered_mask.permute(1, 2, 0)

        return ret
    
    def query_triplane(
        self,
        positions: Float[Tensor, "*B N 3"],
        triplanes: Float[Tensor, "*B 3 Cp Hp Wp"],
    ) -> Dict[str, Tensor]:
        batched = positions.ndim == 3
        if not batched:
            # no batch dimension
            triplanes = triplanes[None, ...]
            positions = positions[None, ...]
        positions=positions-positions.mean(-2).unsqueeze(-2)
        positions = scale_tensor(positions, (-self.cfg.radius, self.cfg.radius), (-1, 1))
        
        indices2D: Float[Tensor, "B 3 N 2"] = torch.stack(
                (positions[..., [0, 1]], positions[..., [0, 2]], positions[..., [1, 2]]),
                dim=-3,
            )
        out: Float[Tensor, "B3 Cp 1 N"] = F.grid_sample(
            rearrange(triplanes, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=3),
            rearrange(indices2D, "B Np N Nd -> (B Np) () N Nd", Np=3),
            align_corners=True,
            mode="bilinear",
        )

        if self.cfg.feature_reduction == "concat":
            out = rearrange(out, "(B Np) Cp () N -> B N (Np Cp)", Np=3)
        elif self.cfg.feature_reduction == "mean":
            out = reduce(out, "(B Np) Cp () N -> B N Cp", Np=3, reduction="mean")
        else:
            raise NotImplementedError
        
        if not batched:
            out = out.squeeze(0)

        return out

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

    def forward_single_batch(
        self,
        gs_hidden_features: Float[Tensor, "Np Cp"],
        query_points: Float[Tensor, "Np 3"],
        w2cs: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        znear,
        zfar,
        background_color: Optional[Float[Tensor, "3"]],
        color_w = None,
        xyz_b = None,
        color_b = None,
        opacity_b = None,
        vert3d_uv = None, 
        face_uv=None, 
        face_uv_xy=None,
    ):

        if_gs_valid = self.gs_valid(gs_hidden_features, query_points)
        query_points_valid = query_points[(if_gs_valid.squeeze(1))>self.threshold_low]
        gs_hidden_features_valid = gs_hidden_features[if_gs_valid.squeeze(1)>self.threshold_low]

        query_points_copied = query_points[(if_gs_valid.squeeze(1))>self.threshold_high]
        gs_hidden_features_copied = gs_hidden_features[if_gs_valid.squeeze(1)>self.threshold_high] 
        query_points_copied = self.vert_pos_refinement(gs_hidden_features_copied, query_points_copied)
        
        query_points_valid = torch.cat([query_points_valid, query_points_copied], dim=-2)
        gs_hidden_features_copied = torch.cat([gs_hidden_features_valid, gs_hidden_features_copied], dim=-2)
        gs: GaussianModel = self.forward_gs(gs_hidden_features_copied, query_points_valid)
        out_list = []
       
        vert_uv, vert_d, intermediates_vert = get_uvd(query_points_valid, vert3d_uv[0], face_uv, face_uv_xy)
        vert_uv = vert_uv.unsqueeze(0)
        vert_d = vert_d.unsqueeze(0)
        # normalize it to [-1, 1]
        vert_uv[..., 0] = 2.0 * (vert_uv[..., 0] /1) - 1.0
        vert_uv[..., 1] = 2.0 * (vert_uv[..., 1] /0.5) - 1.0


        if color_b != None:
            color_b = self.query_triplane_texture(vert_uv, color_b.unsqueeze(0).unsqueeze(0)).squeeze(0)
        if opacity_b != None:
            opacity_b = self.query_triplane_texture(vert_uv, opacity_b.unsqueeze(0).unsqueeze(0)).squeeze(0)

        for w2c, intrinsic in zip(w2cs, intrinsics):
            out_list.append(self.forward_single_view(
                                gs, 
                                Camera.from_w2c(w2c = w2c, intrinsic = intrinsic, height = height, width = width, znear = znear, zfar = zfar),
                                background_color,
                                color_w = color_w,
                                xyz_b = xyz_b,
                                color_b = color_b,
                                opacity_b = opacity_b,
                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs

        return out

    def forward(self, 
        scene_codes_texture,
        vert_uv,
        query_points: Float[Tensor, "B Np 3"],
        w2c: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        znear=0.71, 
        zfar=1.42,
        query_points_tar=None,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B 3"]] = None,
        intrinsic_input: Float[Tensor, "B Nv 4 4"] = None,
        w2c_input: Float[Tensor, "B Nv 4 4"] = None,
        texture_rgb = None,
        face = None,
        gs_hidden_features: Float[Tensor, "B Np Cp"] = None,
        mink_idxs_inter = None,
        color_w = None,
        xyz_b = None,
        color_b = None,
        opacity_b = None,
        vert3d_uv=None, 
        face_uv=None, 
        face_uv_xy=None,
        aux_cam_w2c=None,
        aux_cam_intrinsic=None,
        **kwargs):
        batch_size = scene_codes_texture.shape[0]

        out_list = []
        out_list_input = []

        gs_hidden_features_texture = self.query_triplane_texture(vert_uv, scene_codes_texture)
        gs_hidden_features = gs_hidden_features_texture

        if additional_features is not None:
            gs_hidden_features = torch.cat([gs_hidden_features, additional_features], dim=-1)

        mink_idxs_inter = mink_idxs_inter.squeeze(-1)
        if mink_idxs_inter.max()>0:
            gs_hidden_features_list = []
            for b in range(batch_size):
                gs_hidden_features_b = gs_hidden_features[b].unsqueeze(0)
                mink_idxs_inter_b = mink_idxs_inter[b].unsqueeze(0)
                if mink_idxs_inter_b.max()>0:
                    if mink_idxs_inter.shape[1] >30000:
                        n_part = 8
                        part_len = int(mink_idxs_inter.shape[1]/n_part)
                        for p in range(n_part):
                            mink_idxs_inter_b_part = mink_idxs_inter_b*0
                            mink_idxs_inter_b_part[:,part_len*p:part_len*(p+1)] = mink_idxs_inter_b[:,part_len*p:part_len*(p+1)]
                            mink_idxs_inter_b_part = mink_idxs_inter_b_part.bool()
                            if mink_idxs_inter_b_part.max()<0.5:
                                continue
                            gs_hidden_features_b[mink_idxs_inter_b_part] = self.self_attn_layer(gs_hidden_features_b[mink_idxs_inter_b_part].unsqueeze(0)).squeeze(0)
                    else:
                        gs_hidden_features_b[mink_idxs_inter_b] = self.self_attn_layer(gs_hidden_features_b[mink_idxs_inter_b].unsqueeze(0)).squeeze(0)
                gs_hidden_features_list.append(gs_hidden_features_b)
            gs_hidden_features = torch.cat(gs_hidden_features_list, dim=0)

        if query_points_tar is not None:
            query_points = query_points_tar

        out = defaultdict(list)
        out_input = defaultdict(list)

        if aux_cam_w2c is not None:
            out_list_aux = []
            
            for b in range(batch_size):
                out_list_aux.append(self.forward_single_batch(
                    gs_hidden_features=gs_hidden_features[b],
                    query_points=query_points[b],
                    w2cs=aux_cam_w2c[0],
                    intrinsics=aux_cam_intrinsic[0],
                    height=height, 
                    width=width,
                    znear=znear,
                    zfar=zfar,
                    background_color=background_color[b] if background_color is not None else None,
                    color_w = color_w,
                    xyz_b = xyz_b,
                    color_b = color_b,
                    opacity_b = opacity_b,
                    vert3d_uv=vert3d_uv, 
                    face_uv=face_uv, 
                    face_uv_xy=face_uv_xy,),
                )
            
            for out_ in out_list_aux:
                for k, v in out_.items():
                    out[k].append(v)
                for k, v in out.items():
                    if isinstance(v[0], torch.Tensor):
                        out[k] = torch.stack(v, dim=0)
                    else:
                        out[k] = v

        else:
            for b in range(batch_size):
                out_list_input.append(self.forward_single_batch(
                    gs_hidden_features=gs_hidden_features[b],
                    query_points=query_points[b],
                    w2cs=w2c_input[b],
                    intrinsics=intrinsic_input[b],
                    height=height, 
                    width=width,
                    znear=znear,
                    zfar=zfar,
                    background_color=background_color[b] if background_color is not None else None,
                    color_w = color_w,
                    xyz_b = xyz_b,
                    color_b = color_b,
                    opacity_b = opacity_b,
                    vert3d_uv=vert3d_uv, 
                    face_uv=face_uv, 
                    face_uv_xy=face_uv_xy,),
                )
            

            for out_ in out_list_input:
                for k, v in out_.items():
                    out_input[k+'_input'].append(v)
            for k, v in out_input.items():
                if isinstance(v[0], torch.Tensor):
                    out_input[k] = torch.stack(v, dim=0)
                else:
                    out_input[k] = v

            for k, v in out_input.items():
                out[k] = v

        return out
        
