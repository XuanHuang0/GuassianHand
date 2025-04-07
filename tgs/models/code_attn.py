import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from tgs.models.self_attn import SelfAttn


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class MLP_res_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, in_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = x + self._ff_block(self.layer_norm(x))
        return x


class img_feat_to_grid(nn.Module):
    def __init__(self, map_size, code_map_f_dim, n_heads=4, dropout=0.01):
        super().__init__()
        self.code_map_f_dim = code_map_f_dim
        self.map_size = map_size
        self.position_embeddings = nn.Embedding(map_size * map_size, code_map_f_dim)
        self.self_attn = SelfAttn(code_map_f_dim, n_heads=n_heads, hid_dim=code_map_f_dim, dropout=dropout)

    def forward(self, img):
        bs = img.shape[0]
        # print(img.shape)
        assert img.shape[1] == self.code_map_f_dim
        assert img.shape[2] == self.map_size
        assert img.shape[3] == self.map_size

        position_ids = torch.arange(self.map_size * self.map_size, dtype=torch.long, device=img.device)
        position_ids = position_ids.unsqueeze(0).repeat(bs, 1)

        # print(position_ids.shape) #[8, 4096]

        position_embeddings = self.position_embeddings(position_ids)

        # print(position_embeddings.shape) #[8, 4096, 32]

        # print(img.shape) #[8, 10, 64, 32, 32]

        # img=img.reshape(-1,*img.shape[2:])

        # img = rearrange(img, "B Np Cp Hp Wp -> (B Np) Cp Hp Wp", Np=10),

        # print(img.shape) #[8, 10, 64, 32, 32]

        # grid_feat = F.relu(self.proj(img))

        grid_feat = img

        # print(grid_feat.shape)

        grid_feat = grid_feat.view(bs, self.code_map_f_dim, -1).transpose(-1, -2)

        # print(grid_feat.shape)

        grid_feat = grid_feat + position_embeddings

        # print(grid_feat.shape)

        grid_feat = self.self_attn(grid_feat)

        return grid_feat


# class img_attn(nn.Module):
#     def __init__(self, verts_f_dim, img_f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
#         super().__init__()
#         self.img_f_dim = img_f_dim
#         self.verts_f_dim = verts_f_dim

#         self.fc = nn.Linear(img_f_dim, verts_f_dim)
#         self.Attn = SelfAttn(verts_f_dim, n_heads=n_heads, hid_dim=verts_f_dim, dropout=dropout)

#     def forward(self, verts_f, img_f):
#         assert verts_f.shape[2] == self.verts_f_dim
#         assert img_f.shape[2] == self.img_f_dim
#         assert verts_f.shape[0] == img_f.shape[0]
#         V = verts_f.shape[1]

#         img_f = self.fc(img_f)

#         x = torch.cat([verts_f, img_f], dim=1)
#         x = self.Attn(x)

#         verts_f = x[:, :V]

        return verts_f

class attn(nn.Module):
    def __init__(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        super().__init__()

        self.build_inter_attn(f_dim, n_heads, d_q, d_v, dropout)

        for m in self.modules():
            weights_init(m)

    def build_inter_attn(self, f_dim, n_heads=4, d_q=None, d_v=None, dropout=0.1):
        if d_q is None:
            d_q = f_dim // n_heads
        if d_v is None:
            d_v = f_dim // n_heads

        self.n_heads = n_heads
        self.d_q = d_q
        self.d_v = d_v
        self.norm = d_q ** 0.5
        self.f_dim = f_dim

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_qs = nn.Linear(f_dim, n_heads * d_q)
        self.w_ks = nn.Linear(f_dim, n_heads * d_q)
        self.w_vs = nn.Linear(f_dim, n_heads * d_v)
        self.fc = nn.Linear(n_heads * d_v, f_dim)

        self.layer_norm1 = nn.LayerNorm(f_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(f_dim, eps=1e-6)
        self.ffL = MLP_res_block(f_dim, f_dim, dropout)
        self.ffR = MLP_res_block(f_dim, f_dim, dropout)

    def inter_attn(self, code_map, texture_map):
        BS, n_map, V, fdim = code_map.shape
        assert fdim == self.f_dim
        BS, n_map, V, fdim = texture_map.shape
        assert fdim == self.f_dim

        code_map2 = self.layer_norm1(code_map)
        texture_map2 = self.layer_norm2(texture_map)

        code_mapq = self.w_qs(code_map2).view(BS, n_map, V, self.n_heads, self.d_q).transpose(2, 3)  # BS x h x V x q
        code_mapk = self.w_ks(code_map2).view(BS, n_map, V, self.n_heads, self.d_q).transpose(2, 3)  # BS x h x V x q
        code_mapv = self.w_vs(code_map2).view(BS, n_map, V, self.n_heads, self.d_v).transpose(2, 3)  # BS x h x V x v

        texture_mapq = self.w_qs(texture_map2).view(BS, n_map, V, self.n_heads, self.d_q).transpose(2, 3)  # BS x h x V x q
        texture_mapk = self.w_ks(texture_map2).view(BS, n_map, V, self.n_heads, self.d_q).transpose(2, 3)  # BS x h x V x q
        texture_mapv = self.w_vs(texture_map2).view(BS, n_map, V, self.n_heads, self.d_v).transpose(2, 3)  # BS x h x V x v

        attn = torch.matmul(texture_mapq, code_mapk.transpose(-1, -2)) / self.norm  # bs, h, V, V

        attn = F.softmax(attn, dim=-1)  # bs, h, V, V

        attn = self.dropout1(attn)

        # print(attn.shape)
        feat_code = torch.matmul(attn, code_mapv).transpose(2, 3).contiguous().view(BS, n_map, V, -1)
        feat_code = self.dropout2(self.fc(feat_code))
        # print(feat_code.shape)
        texture_map_attned = self.ffR(texture_map + feat_code)

        return texture_map_attned

    def forward(self, code_map, texture_map):
        BS, V, fdim = code_map.shape
        assert fdim == self.f_dim
        BS, V, fdim = texture_map.shape
        assert fdim == self.f_dim

        bs = texture_map.shape[0]
        n_code = code_map.shape[0]

        # print(code_map.shape)
        code_map = code_map.unsqueeze(0).repeat(bs, 1, 1, 1)
        # print(code_map.shape)
        # code_map = code_map.reshape(-1, *code_map.shape[2:])
        # print(code_map.shape)

        # print(texture_map.shape)
        texture_map = texture_map.unsqueeze(1).repeat(1, n_code, 1, 1)
        # print(texture_map.shape)
        # texture_map = texture_map.reshape(-1,*texture_map.shape[2:])
        # print(texture_map.shape)

        texture_map_attned = self.inter_attn(code_map, texture_map)
        # print(code_map.shape)
        # print(texture_map.shape)
        # code_map = code_map.reshape(bs, n_code, *code_map.shape[1:])
        # texture_map = texture_map.reshape(bs, n_code, *texture_map.shape[1:])
        # print(code_map.shape)
        # print(texture_map.shape)

        return texture_map_attned


class code_attn(nn.Module):
    def __init__(self, map_size, map_f_dim,
                 n_code=5,
                 n_heads=4,
                 dropout=0.01):
        super().__init__()
        
        self.map_f_dim = map_f_dim
        self.map_size = map_size

        self.codebook=nn.Parameter(torch.rand([n_code, map_f_dim, map_size, map_size]))
        self.encoder_code = img_feat_to_grid(map_size, map_f_dim, n_heads, dropout)
        self.encoder_texture = img_feat_to_grid(map_size, map_f_dim, n_heads, dropout)
        self.attn = attn(map_f_dim, n_heads=n_heads, dropout=dropout)

        # for m in self.modules():
        #     weights_init(m)

    def forward(self, texture_map):
        assert texture_map.shape[-3] == self.map_f_dim
        assert texture_map.shape[-2] == self.map_size
        assert texture_map.shape[-1] == self.map_size
        bs = texture_map.shape[0]

        code_map = self.encoder_code(self.codebook)
        texture_map = self.encoder_texture(texture_map)
        
        # print(code_map.shape)
        # print(texture_map.shape)

        texture_map_attned = self.attn(code_map, texture_map)
        # print(texture_map_attned.shape)
        texture_map_attned = texture_map_attned.mean(1)
        # print(texture_map_attned.shape)
        texture_map_attned = texture_map_attned.reshape(bs, self.map_size, self.map_size, self.map_f_dim)
        # print(texture_map_attned.shape)
        texture_map_attned = texture_map_attned.permute(0,3,1,2)
        # print(texture_map_attned.shape)

        return texture_map_attned

if __name__ == "__main__":
    texture_map = torch.rand([8, 64, 32, 32])
    # code_book = torch.rand([10, 64, 32, 32])
    img_attn=img_ex(map_size = 32, map_f_dim = 64, n_code = 10)

    img_attn(texture_map)
