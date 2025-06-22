import argparse
from tqdm import tqdm
import numpy as np
import os

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

def make_unless_exits(url:str) -> None:
    if not os.path.exists(url):
        os.makedirs(url)

def print_argparse(args: argparse.Namespace) -> None:
    for arg in vars(args):
        print(f'--{arg} = {getattr(args, arg)}')

class BatchTransform(object):
    def __init__(self, transforms: nn.Module) -> None:
        super().__init__()
        self.transforms = transforms
    
    def transf(self, ls: torch.Tensor) -> torch.Tensor:
        ret = []
        for i in range(ls.size(0)):
            tmp = torch.unsqueeze(self.transforms(ls[i]), dim=0)
            ret.append(tmp)
        return torch.cat(ret, dim=0)
    
    def tran_one(self, x: torch.Tensor) -> torch.Tensor:
        return torch.unsqueeze(self.transforms(x), dim=0)
    
    def inner_transforms(self) -> nn.Module:
        return self.transforms
    
def count_ttl_params(model: nn.Module, filter_by_grad=False, requires_grad=True):
    if not filter_by_grad:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)
    
def cal_norm(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    for idx, (features, _) in tqdm(enumerate(loader), total=len(loader)):
        channel_size = features.shape[1]
        if idx == 0:
            mean = torch.zeros((channel_size), dtype=torch.float32)
            std = torch.zeros((channel_size), dtype=torch.float32)
        features = torch.transpose(features, 1, 0)
        features = features.reshape(channel_size, -1)
        mean += features.mean(1)
        std += features.std(1)
    mean /= len(loader)
    std /= len(loader)
    return mean.detach().numpy(), std.detach().numpy()

def store_model_structure_to_txt(model: nn.Module, output_path: str) -> None:
    model_info = str(model)
    with open(output_path, 'w') as f:
        f.write(model_info)

def parse_mean_std(arg: str):
    ret = []
    for it in arg.split(','):
        ret.append(float(it.strip()))
    return ret

def relative_path(args:argparse.Namespace, *p):
    return os.path.join(args.full_output_path, *p)