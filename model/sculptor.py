import torch
import torch.nn as nn
from utils import to_tensor
import numpy as np
from model.lbs import *
import einops

class SCULPTOR_layer(nn.Module):
    def __init__(self):
        super().__init__()
        
        sculptor_para = np.load('model/paradict.npy',allow_pickle=True).item()
    
        self.dtype = torch.float32
        
        self.skullmesh_face = sculptor_para['skullmesh_face']
        self.facialmesh_face = sculptor_para['facialmesh_face']
        
        self.template_skull, self.template_face = sculptor_para['template_skull'], sculptor_para['template_face']
    
        self.skull_shape, self.face_shape = sculptor_para['skull_shape'], sculptor_para['face_shape']
        
        self.skull_div, self.face_div = sculptor_para['skull_div'], sculptor_para['face_div']
        self.register_buffer('template', to_tensor(np.vstack([sculptor_para['template_skull'], sculptor_para['template_face']])))
        
        face_shape_dir = np.vstack([sculptor_para['face_div'], sculptor_para['face_shape']])
        skull_shape_dir = np.vstack([sculptor_para['skull_div'], sculptor_para['skull_shape']])
        shape_dir = np.hstack([skull_shape_dir,face_shape_dir])
        shape_dir = einops.rearrange(shape_dir, 'N V d -> V d N')

        self.register_buffer('shape_dirs', to_tensor(shape_dir))
        self.register_buffer('parents', to_tensor(sculptor_para['parents']).long())
        self.register_buffer('J_reg', to_tensor(sculptor_para['J_reg']))
        self.register_buffer('pose_dir', to_tensor(sculptor_para['pose_dir']))
        self.register_buffer('lbs_weights', to_tensor(sculptor_para['lbs_weights']))

    
    def split_mesh(self, batch_mesh):
        skulls = batch_mesh[:, :self.template_skull.shape[0]]
        faces = batch_mesh[:, self.template_skull.shape[0]:]
        return skulls, faces
    
    def J_regressor(self, neutral):
        return torch.einsum('jV,NVd-> Njd', [self.J_reg, neutral])
    
    def forward(self, beta_s, pose_theta, jaw_offset):
        batch_size = beta_s.shape[0]
        device, dtype = beta_s.device, beta_s.dtype
        Full_pose = pose_theta
        shape_offset = ShapeBlendShape(beta_s, self.shape_dirs)
        Tpi = self.template + shape_offset
        joints = self.J_regressor(Tpi)
        pose_offset, rot_mats = PoseBlendShape(Full_pose, self.pose_dir, batch_size, device, dtype)
        T,_ = Trans_WG(rot_mats, joints, self.parents, jaw_offset, self.lbs_weights, batch_size, dtype)
        V_Morphed = LinearBlendSkinning(Tpi+pose_offset, T, batch_size, dtype, device)
        
        return V_Morphed
    