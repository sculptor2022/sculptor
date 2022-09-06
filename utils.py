import torch

def to_tensor(x, dtype=torch.float32):

    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, dtype=dtype)