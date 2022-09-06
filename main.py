from model.sculptor import SCULPTOR_layer
import torch
import numpy as np


def beta_s_Preparation(batch_size, n_shape, dtype=torch.float32):
    return torch.tensor(np.zeros((batch_size, n_shape))+1e-5,dtype=dtype)

def PosePreparation(batch_size, num_joints=1, dtype=torch.float32):
    zero_pose = torch.tensor(np.tile(np.zeros(3*(num_joints+1)).reshape(1,3*(num_joints+1)),(batch_size,1)),dtype=dtype)
    return zero_pose

def JawOffsetPreparation(batch_size, dtype=torch.float32):
    zero_offset = torch.tensor(np.zeros((batch_size, 3, 1)),dtype=dtype)
    return zero_offset

if __name__ =='__main__':
    sculptor = SCULPTOR_layer()

    batch_size=60
    beta, pose, jaw = beta_s_Preparation(batch_size, 60), PosePreparation(batch_size), JawOffsetPreparation(batch_size)
    
    new_result = sculptor(beta, pose, jaw)
    
