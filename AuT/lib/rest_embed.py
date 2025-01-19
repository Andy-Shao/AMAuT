from typing import Any
import random

import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, marsked_rate:float, width=128, num_layers=[6,8]) -> None:
        super(Embedding, self).__init__()
        ng = 32
        assert width % ng == 0, 'width must be dividable by the num_groups in GroupNorm.'
        self.restnet = RestNet(cin=num_channels, embed_size=embed_size, width=width, ng=ng, num_layers=num_layers)
        self.drop_out = nn.Dropout(p=marsked_rate)
        self.patch_embedding = nn.Conv1d(in_channels=width*8, out_channels=embed_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, channel_num, token_num, token_len = x.size()
        x = self.drop_out(x)
        x = x.reshape(batch_size, -1, token_len)
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)
        # x = self.drop_out(x)

        return x
    
class RestNet(nn.Module):
    def __init__(self, cin:int, embed_size:int, width:int, ng:int, num_layers:list[int]) -> None:
        super(RestNet, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(ng, width, eps=1e-6),
            nn.ReLU()
        )

        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.layer1 = nn.ModuleList()
        self.layer1.append(RestNetBlock(cin=width, cout=width*4, cmid=width, ng=ng))
        for _ in range(num_layers[0]): self.layer1.append(RestNetBlock(cin=width*4, cout=width*4, cmid=width, ng=ng))

        self.layer2 = nn.ModuleList()
        self.layer2.append(RestNetBlock(cin=width*4, cout=width*8, cmid=width*2, stride=2, ng=ng))
        for _ in range(num_layers[1]): self.layer2.append(RestNetBlock(cin=width*8, cout=width*8, cmid=width*2, ng=ng))

        if len(num_layers) >= 3:
            self.layer3 = nn.ModuleList()
            self.layer3.append(RestNetBlock(cin=width*8, cout=width*16, cmid=width*4, stride=2, ng=ng))
            for _ in range(num_layers[2]): self.layer3.append(RestNetBlock(cin=width*16, cout=width*16, cmid=width*2, ng=ng))


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layer1: x = l(x)
        for l in self.layer2: x = l(x)
        if hasattr(self, 'layer3'):
            for l in self.layer3: x = l(x)
        return x

class RestNetBlock(nn.Module):
    def __init__(self, cin:int, cout:int, cmid:int, ng:int, stride=1) -> None:
        super(RestNetBlock, self).__init__()

        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=ng, num_channels=cmid, eps=1e-6)
        
        self.conv2 = conv3x3(cmid, cmid, stride=stride, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=ng, num_channels=cmid, eps=1e-6)

        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=ng, num_channels=cout, eps=1e-6)
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
    
class RandomMask(nn.Module):
    def __init__(self, w=(16, 16), p=.0, inplace=False):
        super(RandomMask, self).__init__()
        self.w = w
        self.p = p
        self.inplace = inplace

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.inplace: ret = x
        else: ret = torch.clone(x)

        if self.p == .0:
            return ret
        
        batch_size, channels, width, height = x.size()
        max_width = width / self.w[0]
        max_height = height / self.w[1]
        win_num = (width * height) // (self.w[0] * self.w[1])
        mask_num = (width * height * self.p) // (self.w[0] * self.w[1])
        mask_index = random.sample(range(0, win_num), k=mask_num)
        for i in mask_index:
            h = i % max_width
            w = i // h
            ret[:, :, w*self.w[0]:(w+1)*self.w[0], h*self.w[1]:(h+1)*self.w[1]] = 0
        return ret