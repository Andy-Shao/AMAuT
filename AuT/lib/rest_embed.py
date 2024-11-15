from typing import Any
from collections import OrderedDict

import torch 
import torch.nn as nn

from .embedding import MlpBlock

class Embedding(nn.Module):
    def __init__(self, num_channels:int, token_len:int, embed_size:int) -> None:
        super(Embedding, self).__init__()

        self.ext_lin1 = nn.ModuleList([MlpBlock(fin=token_len, fout=token_len) for _ in range(2)])
        
        self.ext_lin2 = nn.ModuleList()
        for i in range(5):
            if i == 0:
                self.ext_lin2.append(MlpBlock(fin=token_len, fout=embed_size))
            else:
                self.ext_lin2.append(MlpBlock(fin=embed_size, fout=embed_size))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for lin in self.ext_lin1:
            x = lin(x)
        for lin in self.ext_lin2:
            x = lin(x)

        return x
    
class RestNet(nn.Module):
    def __init__(self, cin:int) -> None:
        super(RestNet, self).__init__()
        width = 64
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('max_pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict(
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', RestNetBlock(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', RestNetBlock(cin=width*4, cout=width*4, cmid=width)) for i in range(7)]
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', RestNetBlock(cin=width*4, cout=width*16, cmid=width*4, stride=2))]+
                [(f'unit{i:d}', RestNetBlock(cin=width*16, cout=width*16, cmid=width*4)) for i in range(9)]
            )))
        ))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #TODO
        return x

class RestNetBlock(nn.Module):
    def __init__(self, cin:int, cout=None, cmid=None, stride=1) -> None:
        super(RestNetBlock, self).__init__()
        cout = cout or cin
        cmid = cmid or cin//4

        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=32, num_channels=cmid, eps=1e-6)
        
        self.conv2 = conv3x3(cmid, cmid, stride=stride, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=cmid, eps=1e-6)

        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=cout, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride=stride, bias=False)
            self.ds_gn = nn.GroupNorm(num_groups=cout, num_channels=cout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.ds_gn(residual)

        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

def conv1x1(cin:int, cout:int, stride=1, groups=1, bias=False) -> nn.Conv2d:
    return StdConv2d(
        in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0, bias=bias,
        groups=groups
    )

def conv3x3(cin:int, cout:int, stride=1, groups=1, bias=False) -> nn.Conv2d:
    return StdConv2d(
        in_channels=cin, out_channels=cout, kernel_size=3, stride=1, padding=1, bias=bias,
        groups=groups
    )

class StdConv2d(nn.Conv2d):
    def forward(self, x) -> Any:
        import torch.nn.functional as F

        w = self.weight
        var, mean = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-5)
        return F.conv2d(
            input=x, weight=w, bias=self.bias, stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        )