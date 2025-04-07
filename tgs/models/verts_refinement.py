import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)


class MLP_block(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ff_block(self, x):
        x = self.fc2(self.dropout1(F.relu(self.fc1(x))))
        return self.dropout2(x)

    def forward(self, x):
        x = self._ff_block(self.layer_norm(x))
        return x


class vert_pos_refinement(nn.Module):
    def __init__(self, verts_f_dim, radius=0.001, if_detach = False):
        super().__init__()
        self.radius = radius
        self.detach = if_detach
        self.verts_f_dim = verts_f_dim
        self.ff = MLP_block(verts_f_dim+3, (verts_f_dim+3)//4)
        self.fc = nn.Linear((verts_f_dim+3)//4, 3)
        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f, verts_position):
        assert verts_f.shape[-1] == self.verts_f_dim
        if self.detach:
            verts_f_pos = torch.cat([verts_f.detach(), verts_position.detach()], dim=-1)
            print("detach")
        else:    
            verts_f_pos = torch.cat([verts_f, verts_position], dim=-1)
        verts_f_pos = self.ff(verts_f_pos)
        verts_bias = self.fc(verts_f_pos)
        verts_bias = F.tanh(verts_bias)*self.radius
        # print(verts_bias.max())
        # print(verts_bias.min())
        verts_position_refined = verts_position.detach() + verts_bias
        return verts_position_refined

class vert_valid(nn.Module):
    def __init__(self, verts_f_dim, if_detach = False):
        super().__init__()
        # self.threshold = threshold
        self.verts_f_dim = verts_f_dim
        self.detach = if_detach
        self.ff = MLP_block(verts_f_dim+3, (verts_f_dim+3)//4)
        self.fc = nn.Linear((verts_f_dim+3)//4, 1)
        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f, verts_position):
        assert verts_f.shape[-1] == self.verts_f_dim
        if self.detach:
            verts_f_pos = torch.cat([verts_f.detach(), verts_position.detach()], dim=-1)
            print("detach")
        else:
            verts_f_pos = torch.cat([verts_f, verts_position], dim=-1)
        verts_f_pos = self.ff(verts_f_pos)
        valid = self.fc(verts_f_pos)
        valid = F.sigmoid(valid)
        # valid = valid>self.threshold
        return valid

class vert_inter_info(nn.Module):
    def __init__(self, verts_f_dim):
        super().__init__()
        self.verts_f_dim = verts_f_dim
        self.ff1 = MLP_block(verts_f_dim+6, (verts_f_dim+6)//2)
        self.ff2 = MLP_block((verts_f_dim+6)//2, verts_f_dim)
        # self.fc = nn.Linear((verts_f_dim+3)//2, verts_f_dim)
        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f, verts_position, verts_uv, verts_inter):
        assert verts_f.shape[-1] == self.verts_f_dim
        verts_f_inter = torch.cat([verts_f, verts_position, verts_uv, verts_inter], dim=-1)
        verts_f_inter = self.ff1(verts_f_inter)
        verts_f_inter = self.ff2(verts_f_inter)
        return verts_f_inter

class vert_inter_info_fc(nn.Module):
    def __init__(self, verts_f_dim):
        super().__init__()
        self.verts_f_dim = verts_f_dim
        # self.ff1 = MLP_block(verts_f_dim+6, (verts_f_dim+6)//2)
        # self.ff2 = MLP_block((verts_f_dim+6)//2, verts_f_dim)
        self.fc1 = nn.Linear(verts_f_dim, verts_f_dim//4)
        self.fc2 = nn.Linear((verts_f_dim+3)//4, 50)
        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f_inter):
        assert verts_f_inter.shape[-1] == self.verts_f_dim
        verts_f_inter = self.fc1(verts_f_inter)
        verts_f_inter = self.fc2(verts_f_inter)
        return verts_f_inter

class additional_features_fc(nn.Module):
    def __init__(self, f_dim_in, f_dim_out):
        super().__init__()
        self.ff1 = MLP_block(f_dim_in, f_dim_out)
        # self.ff2 = MLP_block(f_dim_in//4, f_dim_out)

        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f):
        verts_f = self.ff1(verts_f)
        # verts_f = self.ff2(verts_f)
        return verts_f

class identity_code_infer(nn.Module):
    def __init__(self, n_token, token_dim):
        super().__init__()
        self.token_dim = token_dim
        self.n_token = n_token
        # self.ff1 = MLP_block(verts_f_dim+6, (verts_f_dim+6)//2)
        # self.ff2 = MLP_block((verts_f_dim+6)//2, verts_f_dim)
        self.fc1 = nn.Linear(token_dim, 33)
        self.norm = nn.LayerNorm(33)
        self.fc2 = nn.Linear(n_token*33, 33)
        for m in self.modules():
            weights_init(m)

    def forward(self, verts_f_inter):
        # assert verts_f_inter.shape[-1] == self.verts_f_dim
        bs = verts_f_inter.shape[0]
        verts_f_inter = self.fc1(verts_f_inter)
        verts_f_inter = self.norm(verts_f_inter)
        verts_f_inter = self.fc2(verts_f_inter.view(bs,-1))
        verts_f_inter = F.sigmoid(verts_f_inter)
        verts_f_inter = verts_f_inter*2-1
        return verts_f_inter

if __name__ == "__main__":
    # verts_f = torch.rand([8, 24674, 773])
    # verts_position = torch.rand([8, 24674, 3])
    # verts_refinement = vert_pos_refinement(verts_f_dim = 773)
    # verts_position = verts_refinement(verts_f, verts_position)
    # print(verts_position.shape)

    verts_pose_f = torch.rand([8, 24674, 768])
    verts_position = torch.rand([8, 24674, 3])
    verts_inter = torch.rand([8, 24674, 1])