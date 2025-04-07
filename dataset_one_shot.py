import json
import cv2
import numpy as np
import os, sys
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader

import imageio as imageio
import copy
from glob import glob
import os.path as osp
from mis_utils import edge_subdivide, read_mano_uv_obj
from PIL import Image, ImageDraw
import random
import json
# from pycocotools.coco import COCO
import scipy.io as sio
import smplx
from torchvision import transforms
import trimesh

# mano layer
smplx_path = './smplx/models/'
mano_layer = {'right': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=True), 'left': smplx.create(smplx_path, 'mano', use_pca=False, is_rhand=False)}
# fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
if torch.sum(torch.abs(mano_layer['left'].shapedirs[:,0,:] - mano_layer['right'].shapedirs[:,0,:])) < 1:
    print('Fix shapedirs bug of MANO')
    mano_layer['left'].shapedirs[:,0,:] *= -1

import torchvision.transforms as transforms
import pickle

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

def concat_meshes(mesh_list):
    '''manually concat meshes'''
    cur_vert_number = 0
    cur_face_number = 0
    verts_list = []
    faces_list = []
    for idx, m in enumerate(mesh_list):
        verts_list.append(m.vertices)
        faces_list.append(m.faces + cur_vert_number)
        cur_vert_number += len(m.vertices)

    combined_mesh = trimesh.Trimesh(np.concatenate(verts_list),
        np.concatenate(faces_list), process=False
    )
    return combined_mesh

def FillHole(im_in,SavePath,savePathIn):
    im_floodfill = im_in.copy()
    h, w = im_in.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    isbreak = False
    seedPoint=(0,0)
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if(im_floodfill[i][j]==0):
                seedPoint=(i,j)
                isbreak = True
                break
        if(isbreak):
            break
    cv2.floodFill(im_floodfill, mask,seedPoint, 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_in | im_floodfill_inv
    return im_out

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, **kwargs):
        self.split = split # train, test, val
        self.mode = split # train, test
        self.sc_factor = 1
        if split == 'val':
            self.mode = 'test'
        self.edit=kwargs.get('edit', False)
        if self.edit:
            self.mode = 'train'
        else:
            if split == 'train':
                self.mode = 'test'
        self.nearmin = 2
        self.farmax = 0
        self.input_per_frame = kwargs.get('input_per_frame_test', 1)
        self.num_input_view = kwargs.get('num_input_view', 1)
        self.if_color_jitter=kwargs.get('color_jitter', False)
        self.if_mask_sa=kwargs.get('mask_sa', False)

        self.pose_sequence = kwargs.get('pose_sequence', None)

        self.stage = kwargs.get('stage', 1)

        self.if_render_mask=kwargs.get('render_mask', False)
        self.if_edge_subdivide=kwargs.get('edge_subdivide', False)
        self.if_edge_subdivide_hd=kwargs.get('edge_subdivide_hd', False)
        self.djd = kwargs.get('djd', False)
        if self.djd:
            print('big angle test!!!!!!')
        if self.mode == 'train' and self.if_color_jitter:
            self.jitter = self.color_jitter()
        self.annot_path = './InterHand2.6M/annotations'
        joint_regressor = np.load('./smplx/models/mano/J_regressor_mano_ih26m.npy')
        self.joint_regressor=torch.tensor(joint_regressor)
        self.image2tensor = transforms.Compose([transforms.ToTensor(), ])
        self.vt, self.ft_l, self.ft_r, self.change_r, self.change_l= self.get_uvf()
        self.sequence_names = []
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_joint_3d.json')) as f:
            self.joints = json.load(f)
        with open(osp.join(self.annot_path, self.mode, 'InterHand2.6M_' + self.mode + '_MANO_NeuralAnnot.json')) as f:
            self.manos = json.load(f)

        self.use_intag_preds = kwargs.get('use_intag_preds', False)
        self.repose = kwargs.get('repose', True)
        self.ratio=kwargs.get('ratio', 1)

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)
    
    def get_uvf(self):
        vt_r, ft_r, f_r = read_mano_uv_obj('./mano_uv/original mano template/hand.obj')  
        vt_l = vt_r
        vt_r=vt_r/2
        vt_l[...,0]=0.5+vt_l[...,0]/2
        vt_l[...,1]=vt_l[...,1]/2
        vt=np.concatenate((vt_r,vt_l))
        change_r = np.load("./mano_uv/change/change_r.npy")
        change_l=np.load('./mano_uv/change/change_l.npy', allow_pickle=True)
        face_left=np.load('./mano_uv/change/face_left.npy', allow_pickle=True)
        ft_l = face_left
        return vt, ft_l, ft_r, change_r, change_l

    def color_jitter(self):
        ops = []
        ops.extend(
            [transforms.ColorJitter(brightness=(0.2, 2),
                                    contrast=(0.3, 2), saturation=(0.2, 2),
                                    hue=(-0.5, 0.5)), ]
        )
        return transforms.Compose(ops)

    @staticmethod
    def get_mask_at_box(bounds, K, R, T, H, W):
        ray_o, ray_d = Dataset.get_rays(H, W, K, R, T)

        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = Dataset.get_near_far(bounds, ray_o, ray_d)
        return mask_at_box.reshape((H, W)),near.min(),far.max()
    
    def load_human_bounds_pred(self, vert_world_pred):
        
        xyz=vert_world_pred

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    def load_human_bounds(self, capture_id,frame_idx, hand_type):
        mano_valid=np.zeros((2,))
        if hand_type == 'right' or hand_type == 'left':
            
            mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)
            root_pose = mano_pose[0].view(1,3)
            hand_pose = mano_pose[1:,:].view(1,-1)
            shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
            trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
            output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].detach().numpy()
            if hand_type == 'left':
                mano_valid[1]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh0,mesh,axis=0) #(778*2,3)
            else:
                mano_valid[0]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh,mesh0,axis=0) #(778*2,3)
            
        else:
            for hand in ('right', 'left'):
                try:
                    mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
                    root_pose = mano_pose[0].view(1,3)
                    hand_pose = mano_pose[1:,:].view(1,-1)
                    shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
                    trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
                    output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                    mesh = output.vertices[0].detach().numpy()
                    if hand == 'left':
                        mano_valid[1]=1
                    else:
                        mano_valid[0]=1
                except:
                    mesh=np.zeros((778,3))
                    mano_pose=np.zeros((16,3))
                    shape=np.zeros((1,10))
                    trans=np.zeros((1,3))
                if hand == 'left':
                    mesh_left=mesh
                else:
                    mesh_right=mesh
            mesh=np.append(mesh_right,mesh_left,axis=0) #(778*2,3)

        xyz=mesh
        if hand_type == 'right':
            xyz=mesh[:778,:]
        elif hand_type == 'left':
            xyz=mesh[778:,:]

        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[2] -= 0.05
        max_xyz[2] += 0.05
        bounds = np.stack([min_xyz, max_xyz], axis=0)
        return bounds

    def handtype_str2array(self, hand_type):
        if hand_type == 'right':
            return np.array([1,0], dtype=np.float32)
        elif hand_type == 'left':
            return np.array([0,1], dtype=np.float32)
        elif hand_type == 'interacting':
            return np.array([1,1], dtype=np.float32)
        else:
            assert 0, print('Not supported hand type: ' + hand_type)

    def load_mano2(self, capture_id,frame_idx, hand_type, t_pose_params=False):
        mano_valid=np.zeros((2,))
        if hand_type == 'right' or hand_type == 'left':
            
            mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['pose']).view(-1,3)
            root_pose = mano_pose[0].view(1,3)
            
            Rh = root_pose
            R = cv2.Rodrigues(Rh)[0].astype(np.float32)

            hand_pose = mano_pose[1:,:].view(1,-1)
            shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['shape']).view(1,-1)
            trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand_type]['trans']).view(1,3)
            output = mano_layer[hand_type](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
            mesh = output.vertices[0].detach().numpy()
            face = output.faces[0].detach().numpy()

            
            Th = trans.astype(np.float32)

            mano_pose=mano_pose.view(1,-1)
            mano_pose=np.squeeze(mano_pose)
            shape=np.squeeze(shape)
            trans=np.squeeze(trans)
            
            mano_pose0=np.zeros(mano_pose.shape)
            shape0=np.zeros(shape.shape)
            trans0=np.zeros(trans.shape)

            if hand_type == 'left':
                mano_valid[1]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh0,mesh,axis=0) #(778*2,3)

                mano_pose=np.append(mano_pose0,mano_pose,axis=0)
                shape=np.append(shape0,shape,axis=0)
                trans=np.append(trans0,trans,axis=0)
            else:
                mano_valid[0]=1
                mesh0=np.zeros((778,3))
                mesh=np.append(mesh,mesh0,axis=0) #(778*2,3)

                mano_pose=np.append(mano_pose,mano_pose0,axis=0)
                shape=np.append(shape,shape0,axis=0)
                trans=np.append(trans,trans0,axis=0)
            
        else:
            for hand in ('right', 'left'):
                mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['pose']).view(-1,3)
                shape = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['shape']).view(1,-1)
                trans = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)][hand]['trans']).view(1,3)
                if t_pose_params == True:
                    mano_pose = torch.FloatTensor(np.zeros((16,3)).astype(np.float32))
                    if hand == 'left':
                        trans = torch.FloatTensor(np.ones((1,3)).astype(np.float32))*0.5
                    else:
                        trans = torch.FloatTensor(np.zeros((1,3)).astype(np.float32))
                    shapes = torch.FloatTensor(np.zeros((1,10)).astype(np.float32))

                root_pose = mano_pose[0].view(1,3)
                hand_pose = mano_pose[1:,:].view(1,-1)
                Th = trans.numpy().astype(np.float32)
                Rh = root_pose.numpy()
                R = cv2.Rodrigues(Rh)[0].astype(np.float32)

                output = mano_layer[hand](global_orient=root_pose, hand_pose=hand_pose, betas=shape, transl=trans)
                mesh = output.vertices[0].detach().numpy()
                face = mano_layer[hand].faces

                joint= torch.matmul(self.joint_regressor, torch.tensor(mesh).float())
                if self.if_edge_subdivide:
                    mesh,face,_=edge_subdivide(mesh, face)
                    mesh,face,_=edge_subdivide(mesh, face)
                    if self.if_edge_subdivide_hd:
                        mesh,face,_=edge_subdivide(mesh, face)

                nxyz = np.zeros_like(mesh).astype(np.float32)

                if hand == 'left':
                    mano_valid[1]=1
                else:
                    mano_valid[0]=1
           
                if hand == 'left':
                    mesh_left=mesh
                    face_left=face
                    joint_left=joint
                   
                    mano_pose=mano_pose.reshape(1,-1)
                    mano_pose_left=np.squeeze(mano_pose)
                    shape_left=np.squeeze(shape)
                    trans_left=np.squeeze(trans)

                    xyz = np.dot(mesh - Th_r, R_r)
                    xyz_l=xyz
                    Rh_l=Rh
                    Th_l=Th
                    R_l=R
                    cxyz_l = xyz.astype(np.float32)
                    nxyz_l = nxyz.astype(np.float32)

                else:
                    mesh_right=mesh
                    face_right=face
                    joint_right=joint

                    mano_pose=mano_pose.reshape(1,-1)
                    mano_pose_right=np.squeeze(mano_pose)
                    shape_right=np.squeeze(shape)
                    trans_right=np.squeeze(trans)
                    
                    Rh_r=Rh
                    Th_r=Th
                    R_r=R
                    xyz = np.dot(mesh - Th, R)
                    xyz_r=xyz
                    
                    cxyz_r = xyz.astype(np.float32)
                    nxyz_r = nxyz.astype(np.float32)

            xyz = np.append(xyz_r,xyz_l,axis=-2)
            cxyz = np.append(cxyz_r,cxyz_l,axis=-2)
            nxyz = np.append(nxyz_r,nxyz_l,axis=-2)
            Rh = np.append(Rh_r,Rh_l,axis=-2)
            Th = np.append(Th_r,Th_l,axis=-2)
            feature = np.concatenate([cxyz, nxyz], axis=1).astype(np.float32)

            # obtain the bounds for coord construction
            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)
            min_xyz -= 0.05
            max_xyz += 0.05

            bounds = np.stack([min_xyz, max_xyz], axis=0)

            # construct the coordinate
            xyz=torch.from_numpy(xyz)
            xyz=xyz.numpy()

            dhw = xyz[:, [2, 1, 0]]
            min_dhw = min_xyz[[2, 1, 0]]
            max_dhw = max_xyz[[2, 1, 0]]
            voxel_size = np.array([0.005, 0.005, 0.005])
            coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32)

            # construct the output shape
            out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)

            x = 32
            out_sh = (out_sh | (x - 1)) + 1

            joint_world=np.append(joint_right,joint_left,axis=0)

            mesh_right=trimesh.Trimesh(mesh_right,face_right)
            mesh_left=trimesh.Trimesh(mesh_left,face_left)
            mesh=concat_meshes([mesh_right,mesh_left])
            face=mesh.faces
            mesh=mesh.vertices

            vert_right=mesh_right.vertices
            vert_right_uv=vert_right[self.change_r.astype(int),:]
            vert_left=mesh_left.vertices
            vert_left_uv=vert_left[self.change_l.astype(int),:]

            mesh_right_uv=trimesh.Trimesh(vert_right_uv,self.ft_r, process=False)
            mesh_left_uv=trimesh.Trimesh(vert_left_uv,self.ft_l, process=False)
            mesh_uv=concat_meshes([mesh_right_uv,mesh_left_uv])
            face_uv=mesh_uv.faces
            vert_uv=mesh_uv.vertices

            mano_pose=np.append(mano_pose_right.unsqueeze(0),mano_pose_left.unsqueeze(0),axis=0)
            shape=np.append(shape_right.unsqueeze(0),shape_left.unsqueeze(0),axis=0)
            trans=np.append(trans_right.unsqueeze(0),trans_left.unsqueeze(0),axis=0)
            
            # obtain the original bounds for point sampling
            min_mesh = np.min(mesh, axis=0)
            max_mesh = np.max(mesh, axis=0)
            min_mesh -= 0.05
            max_mesh += 0.05
            can_bounds = np.stack([min_mesh, max_mesh], axis=0)

        return joint_world, mesh, face, face_uv, vert_uv, mano_pose, shape, trans, Rh_r, Th_r, R_r, coord, out_sh, can_bounds, bounds, feature


    def __len__(self):
        if self.split=='train':
            if self.edit:
                return 100
            return 50
        elif self.split=='val':
            return 1
        else:
            if self.pose_sequence == 'oneshot_reg_i':
                return 14
            if self.pose_sequence == 'oneshot_nv':
                return 50
            if self.pose_sequence == 'oneshot_train':
                return 1
            if self.edit:
                return 10000
            return 349

    
    def __getitem__(self, index):

        prob = np.random.randint(9000000)
        if self.split == 'test':
            if self.pose_sequence == 'oneshot_nv':
                # index = index*2
                with open(osp.join("./processed_dataset/",self.mode,'index_identity_os_i_test_nv', '{}.pkl'.format(index)), 'rb') as file:
                    data = pickle.load(file)
            elif self.pose_sequence == 'oneshot_train':
                index = 2
                with open(osp.join("./processed_dataset/",self.mode,'index_identity_test_i_one_shot', '{}.pkl'.format(index)), 'rb') as file:
                    data = pickle.load(file)
            else:
                index = index*10
                with open(osp.join("./processed_dataset/",self.mode,'index_identity_test_i_one_shot', '{}.pkl'.format(index)), 'rb') as file:
                    data = pickle.load(file)
                if self.edit:
                    with open(osp.join("./processed_dataset/",self.mode,'index_identity_all_train_i', '{}.pkl'.format(index)), 'rb') as file:
                        data = pickle.load(file)
        else:
            index = 2
            with open(osp.join("./processed_dataset/",self.mode,'index_identity_test_i_one_shot', '{}.pkl'.format(index)), 'rb') as file:
                data = pickle.load(file)
            if self.edit:
                index = 11388
                with open(osp.join("./processed_dataset/",self.mode,'index_identity0_train_i', '{}.pkl'.format(index)), 'rb') as file:
                    data = pickle.load(file)
            

        idx=data['idx']
        frame_idx = data['frame']
        capture_id = data['capture']
        cam = data['cam']
        hand_type ='interacting'

        tar_cam={}  
        targets = {}
        input_imgs, input_msks, input_K, input_Rt = [], [], [], []
        input_R, input_t = [],[]
        idx = 1

        with open(osp.join("./processed_dataset/",self.mode,'annotation', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.pkl'), 'rb') as file:
            anno = pickle.load(file)

            in_T, in_R=anno['camera']['t'].reshape(3), anno['camera']['R']
            in_Rt = np.concatenate((in_R.reshape(3,3), in_T.reshape(3, 1)), axis=1)

            in_K = anno['camera']['in_K']
            princpt = in_K[0:2, 2].astype(np.float32)
            focal = np.array( [in_K[0, 0], in_K[1, 1]], dtype=np.float32)
            campos = anno['camera']['campos']
            camrot = anno['camera']['camrot']
            img_info=anno['image_info']

            if self.edit:
                img_path=osp.join('/home/huangx/GaussianHand/example_img/nips.png')
                mask_path=osp.join("./processed_dataset/",self.mode,'mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
            else:
                img_path=osp.join("./processed_dataset/",self.mode,'image', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')
                if self.if_mask_sa:
                    mask_path=osp.join("./processed_dataset/",self.mode,'mask_sa', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
                else:
                    mask_path=osp.join("./processed_dataset/",self.mode,'mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
            
            mask_path_mano=osp.join("./processed_dataset/",self.mode,'mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
            input_msk_mano = (imageio.imread(mask_path_mano) >=100).astype(np.uint8) #0为黑色

            bbox_mask_path=osp.join("./processed_dataset/",self.mode,'bbox_mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
            try:
                bbox_mask = (imageio.imread(bbox_mask_path) >=100).astype(np.uint8) #0为黑色
            except:
                bbox_mask = input_msk_mano*0+1
            if bbox_mask.shape[-1] ==3:
                bbox_mask = bbox_mask[...,0]
            
            if self.edit:
                input_msk = (imageio.imread(mask_path) >=100).astype(np.uint8) #0为黑色
                if input_msk.shape[-1] == 3:
                    input_msk = (input_msk.mean(-1) > 0.5).astype(np.uint8)
            else:
                try:
                    input_msk = (imageio.imread(mask_path) >=100).astype(np.uint8) #0为黑色
                    # if input_msk.shape[-1] == 3:
                    #     input_msk = (input_msk.mean(-1) > 0.5).astype(np.uint8)
                except:
                    mask_path=osp.join("./processed_dataset/",self.mode,'mask', 'capture'+str(capture_id)+'/cam'+str(cam)+'/frame'+str(frame_idx)+'.jpg')        
                    input_msk = (imageio.imread(mask_path) >=100).astype(np.uint8) #0为黑色
            
                if self.if_render_mask:
                    if input_msk.shape[-1] == 3:
                        input_msk = (input_msk.mean(-1) > 0.5).astype(np.uint8)
                    input_msk=FillHole(input_msk, None, None)
                    input_msk = np.concatenate([input_msk[...,None], input_msk[...,None], input_msk[...,None]], -1)
                    input_msk[input_msk_mano == 0] = 0

            input_img = imageio.imread(img_path)
            #减小色偏
            if self.mode == 'train' and self.if_color_jitter:
                input_img = Image.fromarray(input_img)
                torch.manual_seed(prob)
                input_img = self.jitter(input_img)
                input_img = np.array(input_img)
                torch.manual_seed(prob)
            input_img = input_img.astype(np.float32) / 255.

            if self.edit:
                H, W = int(256), int(256)
                input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)
                H, W = int(input_img.shape[0] ), int(input_img.shape[1])
                input_img_all, input_msk_all= input_img, input_msk
            else:
                input_img_all, input_msk_all= input_img, input_msk
                H, W = int(input_img.shape[0] * self.ratio), int(input_img.shape[1] * self.ratio)
                input_img, input_msk = cv2.resize(input_img, (W, H), interpolation=cv2.INTER_AREA), cv2.resize(input_msk, (W, H), interpolation=cv2.INTER_NEAREST)
                H, W = int(input_img.shape[0] ), int(input_img.shape[1])

            input_img[input_msk == 0] = 0
            input_msk = (input_msk != 0)  # bool mask : foreground (True) background (False)

            if not 1. in input_msk:
                print('input_msk black!!!!!'+img_path)
            input_msk = input_msk.astype(np.uint8) * 255
            input_img = self.image2tensor(input_img)
            input_msk = self.image2tensor(input_msk).bool()
            input_msk_mano = self.image2tensor(input_msk_mano).bool()
            in_K0 = in_K.copy()
            in_K[:2] = in_K[:2] * self.ratio
            princpt_s = in_K[0:2, 2].astype(np.float32)
            focal_s = np.array( [in_K[0, 0], in_K[1, 1]], dtype=np.float32)

            if not self.edit:
                input_img_all[input_msk_all == 0] = 0
            input_msk_all = (input_msk_all != 0)  # bool mask : foreground (True) background (False)

            if not 1. in input_msk_all:
                print('input_msk black!!!!!'+img_path)
            input_img_all = self.image2tensor(input_img_all)
            input_msk_all = self.image2tensor(input_msk_all).bool()

            joint_world, mesh, face, face_uv, vert_uv, mano_pose, shape, trans, Rh_r, Th_r, R_r, coord, out_sh, can_bounds, bounds, feature=self.load_mano2(capture_id,frame_idx, hand_type)
            joint_world_t, mesh_t, face_t, _, _, _, _, _, _, _, _, _, _, _, _, _=self.load_mano2(capture_id,frame_idx, hand_type, t_pose_params=True)
            if_edge_subdivide = self.if_edge_subdivide
            mesh_cam = np.dot(in_R, mesh.transpose(1,0)).transpose(1,0) + in_T.reshape(1,3)
            
            self.if_edge_subdivide = False
            joint_world_or, mesh_or, face_or, face_uv_or, vert_uv_or, _, _, _, _, _, _, _, _, _, _, _=self.load_mano2(capture_id,frame_idx, hand_type)
            self.if_edge_subdivide = if_edge_subdivide
            face_uv_xy_or=self.vt[face_uv_or,:]
            mesh_cam_or = np.dot(in_R, mesh_or.transpose(1,0)).transpose(1,0) + in_T.reshape(1,3)
            
        
            face_uv_xy=self.vt[face_uv,:]

            mesh_cam = np.dot(in_R, mesh.transpose(1,0)).transpose(1,0) + in_T.reshape(1,3)

            if not self.use_intag_preds:
                kpt3d = np.array(self.joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float32)/1000
            else:
                kpt3d = joint_world_pred
                vert_world_pred_tar=vert_world_pred
            image = input_img
            torch3d_T_colmap = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            tar_R = (torch3d_T_colmap @ in_R).T
            tar_T = torch3d_T_colmap @ in_T
            tar_cam['input_R']=torch.from_numpy(tar_R).float()
            tar_cam['input_T']=torch.from_numpy(tar_T).float()
            tar_cam['input_focal']=torch.from_numpy(focal).float()
            tar_cam['input_princpt']=torch.from_numpy(princpt).float()
            tar_cam['input_K']=torch.from_numpy(in_K0).float()
            targets['input_mano_pose']=torch.from_numpy(mano_pose).float()

            targets.update({
                    "joint_world":torch.from_numpy(joint_world).float(),
                    'vert_world':torch.from_numpy(mesh).float(),
                    'vert_cam':torch.from_numpy(mesh_cam).float(),
                    'vert_cam_or':torch.from_numpy(mesh_cam_or).float(),
                    'face_world':torch.from_numpy(face).float(),
                    'face_world_or':torch.from_numpy(face_or).float(),
                    'vert_uv':torch.from_numpy(vert_uv).float(),
                    'face_uv':torch.from_numpy(face_uv).float(),
                    'mesh_t':torch.from_numpy(mesh_t).float(),
                    'face_t':torch.from_numpy(face_t).float(),
                    'face_uv_xy':torch.from_numpy(face_uv_xy).float(),
            })

            if not self.use_intag_preds:
                targets['coord'] = coord
                targets['out_sh'] = out_sh
                targets['bounds'] = bounds
                
            targets["joint_world_input"]=torch.from_numpy(joint_world).float()
            targets["vert_world_input"]=torch.from_numpy(mesh).float()
            targets['input_img_all']=input_img_all.float()
            targets['input_msk_all']=input_msk_all
                       
            # append data
            input_imgs.append(input_img)
            input_msks.append(input_msk)
            input_K.append(torch.from_numpy(in_K))
            input_Rt.append(torch.from_numpy(in_Rt))
            input_R.append(torch.from_numpy(in_R))
            input_t.append(torch.from_numpy(in_T))

        
        hand_type_array=self.handtype_str2array(hand_type)
        ret = {
            'images': torch.stack(input_imgs),
            'images_masks': torch.stack(input_msks),
            'K': torch.stack(input_K),
            'Rt': torch.stack(input_Rt),
            'kpt3d': torch.from_numpy(kpt3d),
            'hand_type':torch.from_numpy(hand_type_array),
            'i': frame_idx,
            'human_idx': int(capture_id),
            'sessision': capture_id,
            'frame_index': frame_idx,
            'human': capture_id,
            'cam_ind': cam,
            "index": {"camera": "cam", "segment": 'VANeRF', "tar_cam_id": idx,
                "frame": f"{capture_id}_{frame_idx}", "ds_idx": cam},
        }

        ret['targets']=targets
        ret["txt"]= "interacting hands with black background"
        
        if self.use_intag_preds:
            bounds = self.load_human_bounds_pred(vert_world_pred_tar)
        else:
            if self.repose:
                bounds = self.load_human_bounds(capture_id,tar_frame, hand_type)
            else:
                bounds = self.load_human_bounds(capture_id,frame_idx, hand_type)
        ret['mask_at_box'],near,far = self.get_mask_at_box(
            bounds,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy(),
            H, W)
        ret['znear'], ret['zfar']=near,far
        if near<self.nearmin:
            self.nearmin=near
        if far>self.farmax:
            self.farmax=far  
        ret['bounds'] = bounds
        ret['mask_at_box'] = ret['mask_at_box'].reshape((H, W))
        x, y, w, h = cv2.boundingRect(ret['mask_at_box'].astype(np.uint8))

        mano_pose = torch.FloatTensor(self.manos[str(capture_id)][str(frame_idx)]['right']['pose']).view(-1,3)
        root_pose = mano_pose[0].reshape(-1)
        
        Rh = root_pose.numpy()
        R,_ = cv2.Rodrigues(Rh)
        R = torch.from_numpy(R)

        headpose = torch.eye(4)
        headpose[:3, :3] = input_Rt[0][:3, :3].t()
        headpose[:3, 3] = torch.from_numpy(kpt3d[0])
        ret['headpose'] = headpose
        input_img_all = targets['input_img_all'].permute(1,2,0)
        input_msk_all = targets['input_msk_all'].permute(1,2,0)

        ray_o, ray_d = Dataset.get_rays(H, W,
            input_K[0].numpy(),
            input_Rt[0][:3, :3].numpy(),
            input_Rt[0][:3, -1].numpy())
        ray_o = torch.from_numpy(ray_o.astype(np.float32))
        ray_d = torch.from_numpy(ray_d.astype(np.float32))
        view_index=torch.as_tensor([0])

        cond_c2w: Float[Tensor, "4 4"] = torch.cat(
            [input_Rt[0], torch.zeros_like(input_Rt[0][:1])], dim=0
        )
        cond_c2w[3, 3] = 1.0

        world_view_transform = torch.tensor(getWorld2View2(input_R[0].numpy(), input_t[0].numpy())).transpose(0, 1)

        cond_w2c = cond_c2w
        cond_c2w=torch.inverse(cond_c2w)
        campos=campos*0.001
       
        cond_c2w_tar: Float[Tensor, "4 4"] = torch.cat(
            [input_Rt[0], torch.zeros_like(input_Rt[0][:1])], dim=0
        )
        cond_c2w_tar[3, 3] = 1.0

        cond_w2c_tar = cond_c2w_tar
        cond_c2w_tar=torch.inverse(cond_c2w_tar)


        intrinsic_normed_cond = input_K[0].clone()


        intrinsic_normed_cond[..., 0, 2] /= W
        intrinsic_normed_cond[..., 1, 2] /= H
        intrinsic_normed_cond[..., 0, 0] /= W
        intrinsic_normed_cond[..., 1, 1] /= H

        intrinsic_normed_cond_tar = input_K[0].clone()
        intrinsic_normed_cond_tar[..., 0, 2] /= W
        intrinsic_normed_cond_tar[..., 1, 2] /= H
        intrinsic_normed_cond_tar[..., 0, 0] /= W
        intrinsic_normed_cond_tar[..., 1, 1] /= H

        if self.edit:
            input_img_all = input_img_all[:, :, :3]
        else:
            input_img_all = input_img_all[
                :, :, :3
            ] * input_msk_all + torch.as_tensor([0., 0., 0.])[None, None, :] * (1 - input_msk_all.float())  
        out = {
            "rgb_cond": input_img_all.float().unsqueeze(0), #
            "tar_img": input_img_all.float().unsqueeze(0),
            "input_R": input_R[0].unsqueeze(0),
            "input_t": input_t[0].unsqueeze(0),
            "tar_R": input_R[0].unsqueeze(0),
            "tar_t": input_t[0].unsqueeze(0),
            "world_view_transform": world_view_transform,
            "c2w_cond": cond_c2w.unsqueeze(0), #
            "w2c_cond": cond_w2c.unsqueeze(0), #
            "mask_cond": input_msk_all.unsqueeze(0), #
            "tar_msk_all":input_msk_all.unsqueeze(0),
            "intrinsic_cond": input_K[0].unsqueeze(0).float(), #
            "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0).float(), #
            "view_index": torch.as_tensor([0]), #
            'bbox_mask' : bbox_mask,
            "intrinsic": input_K[0].unsqueeze(0).float(), #
            "intrinsic_normed": intrinsic_normed_cond.unsqueeze(0).float(), #
            "c2w": cond_c2w.unsqueeze(0).float(), #
            "w2c": cond_w2c.unsqueeze(0).float(), #
            "points":torch.from_numpy(mesh).float(),
            "points_tar":torch.from_numpy(mesh).float(),
            "ret":ret,
        }

        instance_id = os.path.split(img_path)[-1].split('.')[0]
        out["index"] = torch.as_tensor(index)
        out["background_color"] = torch.as_tensor([0., 0., 0.]) #
        out["instance_id"] = instance_id

        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": 256, "width": 256})
        return batch

    @classmethod
    def from_config(cls, dataset_cfg, data_split, cfg):
        ''' Creates an instance of the dataset.

        Args:
            dataset_cfg (dict): input configuration.
            data_split (str): data split (`train` or `val`).
        '''
        assert data_split in ['train', 'val', 'test', 'test_visualize']

        dataset_cfg = copy.deepcopy(dataset_cfg)
        dataset_cfg['is_train'] = data_split == 'train'
        if f'{data_split}_cfg' in dataset_cfg:
            dataset_cfg.update(dataset_cfg[f'{data_split}_cfg'])
        if dataset_cfg['is_train']:
            dataset = cls(split=data_split, **dataset_cfg)
        elif data_split == 'test_visualize':
            # skip every 6th data sample (there are 6 cameras per person)
            dataset = TestDataset(split='test', sample_frame=1, sample_camera=6, **dataset_cfg)
        else:
            dataset = TestDataset(split=data_split, **dataset_cfg)
        return dataset

    @staticmethod
    def get_rays(H, W, K, R, T):
        rays_o = -np.dot(R.T, T).ravel()

        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32), indexing='xy')

        xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
        pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
        pixel_world = np.dot(pixel_camera - T.ravel(), R)
        rays_d = pixel_world - rays_o[None, None]
        rays_o = np.broadcast_to(rays_o, rays_d.shape)

        return rays_o, rays_d

    @staticmethod
    def get_near_far(bounds, ray_o, ray_d, boffset=(-0.01, 0.01)):
        """calculate intersections with 3d bounding box"""
        bounds = bounds + np.array([boffset[0], boffset[1]])[:, None]
        nominator = bounds[None] - ray_o[:, None]
        # calculate the step of intersections at six planes of the 3d bounding box
        ray_d[np.abs(ray_d) < 1e-5] = 1e-5
        d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
        # calculate the six interections
        p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
        # calculate the intersections located at the 3d bounding box
        min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
        eps = 1e-6
        p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                        (p_intersect[..., 0] <= (max_x + eps)) * \
                        (p_intersect[..., 1] >= (min_y - eps)) * \
                        (p_intersect[..., 1] <= (max_y + eps)) * \
                        (p_intersect[..., 2] >= (min_z - eps)) * \
                        (p_intersect[..., 2] <= (max_z + eps))
        # obtain the intersections of rays which intersect exactly twice
        mask_at_box = p_mask_at_box.sum(-1) == 2
        p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
            -1, 2, 3)

        # calculate the step of intersections
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        norm_ray = np.linalg.norm(ray_d, axis=1)
        d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
        d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)

        return near, far, mask_at_box

def draw_keypoints(img, kpts, color=(255, 0, 0), size=3):
    for i in range(kpts.shape[0]):
        kp2 = kpts[i].tolist()
        kp2 = [int(kp2[0]), int(kp2[1])]
        img = cv2.circle(img, kp2, 0, color, size)
    return img

class TestDataset(Dataset):
    def __init__(self, split, sample_frame=30, sample_camera=1, **kwargs):
        super().__init__( split, **kwargs)

def load_cfg(path):
    """ Load configuration file.
    Args:
        path (str): model configuration file.
    """
    if path.endswith('.json'):
        with open(path, 'r') as file:
            cfg = json.load(file)
    elif path.endswith('.yml') or path.endswith('.yaml'):
        with open(path, 'r') as file:
            cfg = yaml.safe_load(file)
    else:
        raise ValueError('Invalid config file.')

    return cfg

if __name__ == "__main__":
    cfg_path = "/home/huangx/GaussianHand/config/one_shot.json"
    cfg = load_cfg(cfg_path)
    dataset = Dataset.from_config(cfg['dataset'], 'train', cfg)
    # dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
    for i in tqdm(range(1000)):
        dataset.__getitem__(i)