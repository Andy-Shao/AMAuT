from typing import Any

import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, marsked_rate:float) -> None:
        super(Embedding, self).__init__()
        width = 64
        self.restnet = RestNet50(cin=num_channels, embed_size=embed_size, width=width)
        self.drop_out = nn.Dropout(p=marsked_rate)
        self.patch_embedding = nn.Conv1d(in_channels=width*16, out_channels=embed_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, channel_num, token_num, token_len = x.size()
        x = x.reshape(batch_size, -1, token_len)
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)
        x = self.drop_out(x)

        return x
    
class RestNet50(nn.Module):
    def __init__(self, cin:int, embed_size:int, width:int) -> None:
        super(RestNet50, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(32, width, eps=1e-6),
            nn.ReLU()
        )

        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.layer1 = nn.ModuleList()
        self.layer1.append(RestNetBlock(cin=width, cout=width*4, cmid=width))
        for _ in range(6): self.layer1.append(RestNetBlock(cin=width*4, cout=width*4, cmid=width))

        self.layer2 = nn.ModuleList()
        self.layer2.append(RestNetBlock(cin=width*4, cout=width*16, cmid=width*4, stride=2))
        for _ in range(8): self.layer2.append(RestNetBlock(cin=width*16, cout=width*16, cmid=width*4))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layer1: x = l(x)
        for l in self.layer2: x = l(x)
        return x

class RestNetBlock(nn.Module):
    def __init__(self, cin:int, cout:int, cmid:int, stride=1) -> None:
        super(RestNetBlock, self).__init__()

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

def conv1x1(cin:int, cout:int, stride=1, bias=False) -> nn.Conv2d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=1, stride=stride, padding=0, bias=bias
    )

def conv3x3(cin:int, cout:int, stride=1, groups=1, bias=False) -> nn.Conv2d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=3, stride=stride, padding=1, bias=bias,
        groups=groups
    )

class StdConv1d(nn.Conv1d):
    def forward(self, x) -> Any:
        import torch.nn.functional as F

        w = self.weight
        var, mean = torch.var_mean(w, dim=[1, 2], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-5)
        return F.conv1d(
            input=x, weight=w, bias=self.bias, stride=self.stride, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        )