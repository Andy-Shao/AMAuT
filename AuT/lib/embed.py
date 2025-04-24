from typing import Any
import random

import torch 
import torch.nn as nn

class FCEmbedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, width=128, num_layers=[6,8]) -> None:
        super(FCEmbedding, self).__init__()
        ng = 32
        self.restnet = FCERestNet(cin=num_channels, width=width, ng=ng, num_layers=num_layers)
        self.patch_embedding = nn.Conv1d(in_channels=width*(2**(1+len(num_layers))), out_channels=embed_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x:torch.Tensor):
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)

        return x

class Embedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, marsked_rate:float, width=128, num_layers=[6,8], in_shape=[80,104], arch:str='CT') -> None:
        super(Embedding, self).__init__()
        ng = 32
        assert width % ng == 0, 'width must be dividable by the num_groups in GroupNorm.'
        self.arch = arch
        if arch == 'CTA':
            self.restnet = ResNet(cin=num_channels, embed_size=embed_size, width=width, ng=ng, num_layers=num_layers)
        elif arch == 'CT':
            self.restnet = ResNetCT(cin=num_channels, width=width, ng=ng, num_layers=num_layers)
        self.drop_out = nn.Dropout(p=marsked_rate)
        self.patch_embedding = nn.Conv1d(in_channels=width*(2**(1+len(num_layers))), out_channels=embed_size, kernel_size=1, stride=1, padding=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_shape[0], in_shape[1]))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x:torch.Tensor):
        x = x + self.pos_embed
        x = self.drop_out(x)
        if self.arch == 'CTA':
            x, hidden_attens = self.restnet(x)
        else:
            x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)

        if self.arch == 'CTA':
            return x, hidden_attens
        else:
            return x
    
class FCERestNet(nn.Module):
    def __init__(self, cin:int, width:int, ng:int, num_layers:list[int]):
        super(FCERestNet, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=14, stride=2, bias=False, padding=6),
            nn.BatchNorm1d(num_features=width, eps=1e-6),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool1d(kernel_size=6, stride=2, padding=2)

        w = width
        self.layers = nn.ModuleList()
        for i, layer_num in enumerate(num_layers):
            if i == 0:
                self.layers.append(FCEResNetBlock(cin=w, cout=w*4, cmid=w, ng=ng))
                for _ in range(layer_num): self.layers.append(FCEResNetBlock(cin=w*4, cout=w*4, cmid=w, ng=ng))
                w = w * 4
            else:
                self.layers.append(FCEResNetBlock(cin=w, cout=w*2, cmid=w//2, stride=2, ng=ng))
                for _ in range(layer_num): self.layers.append(FCEResNetBlock(cin=w*2, cout=w*2, cmid=w//2, ng=ng))
                w = w * 2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layers: x = l(x)
        return x
    
class ResNetCT(nn.Module):
    def __init__(self, cin:int, width:int, ng:int, num_layers:list[int]):
        super(ResNetCT, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3),
            # nn.GroupNorm(ng, width, eps=1e-6),
            nn.BatchNorm1d(num_features=width, eps=1e-6),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        w = width
        self.layers = nn.ModuleList()
        for i, layer_num in enumerate(num_layers):
            if i == 0:
                self.layers.append(ResNetBlock(cin=w, cout=w*4, cmid=w, ng=ng))
                for _ in range(layer_num): self.layers.append(ResNetBlock(cin=w*4, cout=w*4, cmid=w, ng=ng))
                w = w * 4
            else:
                self.layers.append(ResNetBlock(cin=w, cout=w*2, cmid=w//2, stride=2, ng=ng))
                for _ in range(layer_num): self.layers.append(ResNetBlock(cin=w*2, cout=w*2, cmid=w//2, ng=ng))
                w = w * 2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layers: x = l(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, cin:int, embed_size:int, width:int, ng:int, num_layers:list[int]) -> None:
        super(ResNet, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3),
            # nn.GroupNorm(ng, width, eps=1e-6),
            nn.BatchNorm1d(num_features=width, eps=1e-6),
            nn.ReLU()
        )

        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)

        self.layer1 = nn.ModuleList()
        self.layer1.append(ResNetBlock(cin=width, cout=width*4, cmid=width, ng=ng))
        for _ in range(num_layers[0]): self.layer1.append(ResNetBlock(cin=width*4, cout=width*4, cmid=width, ng=ng))

        self.layer2 = nn.ModuleList()
        self.layer2.append(ResNetBlock(cin=width*4, cout=width*8, cmid=width*2, stride=2, ng=ng))
        for _ in range(num_layers[1]): self.layer2.append(ResNetBlock(cin=width*8, cout=width*8, cmid=width*2, ng=ng))

        if len(num_layers) >= 3:
            self.layer3 = nn.ModuleList()
            self.layer3.append(ResNetBlock(cin=width*8, cout=width*16, cmid=width*4, stride=2, ng=ng))
            for _ in range(num_layers[2]): self.layer3.append(ResNetBlock(cin=width*16, cout=width*16, cmid=width*2, ng=ng))


    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, token_num, token_len = x.size()
        hidden_attens = []
        x = self.root(x)
        hidden_attens.append(x)

        x = self.maxPool(x)
        for l in self.layer1: x = l(x)
        
        right_tl = token_len // 4
        hidden_attens.append(self.format_attens(x, right_tl))
        for l in self.layer2: x = l(x)

        if hasattr(self, 'layer3'):
            right_tl = right_tl // 2
            hidden_attens.append(self.format_attens(x, right_tl))
            for l in self.layer3: x = l(x)
            hidden_attens.append(x)

        return x, hidden_attens[::-1]
    
    def format_attens(self, x:torch.Tensor, right_token_len:int) -> torch.Tensor:
        import torch.nn.functional as F
        pad = right_token_len - x.shape[2]
        assert pad < 3 and pad >= 0, f'hidden attention is incorrect: {right_token_len} - {x.shape[2]}'
        if pad == 0: 
            hidden_atten = x
        else:
            # low efficient code:
            # hidden_atten = torch.zeros(x.shape[0], x.shape[1], right_token_len).to(x.device)
            # hidden_atten[:, :, 0:x.shape[2]] = x[:]
            
            hidden_atten = F.pad(x, (0, 1, 0, 0), mode='constant', value=0)
        return hidden_atten
    
class FCEResNetBlock(nn.Module):
    def __init__(self, cin:int, cout:int, cmid:int, ng:int, stride=1) -> None:
        super(FCEResNetBlock, self).__init__()

        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.norm1 = nn.BatchNorm1d(num_features=cmid, eps=1e-6)
        
        self.conv2 = conv7(cmid, cmid, stride=stride, bias=False)
        self.norm2 = nn.BatchNorm1d(num_features=cmid, eps=1e-6)

        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.norm3 = nn.BatchNorm1d(num_features=cout, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride=stride, bias=False)
            self.ds_gn = nn.BatchNorm1d(num_features=cout)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.ds_gn(residual)

        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.norm3(self.conv3(y))

        y = self.relu(residual + y)
        return y

class ResNetBlock(nn.Module):
    def __init__(self, cin:int, cout:int, cmid:int, ng:int, stride=1) -> None:
        super(ResNetBlock, self).__init__()

        self.conv1 = conv1x1(cin, cmid, bias=False)
        # self.gn1 = nn.GroupNorm(num_groups=ng, num_channels=cmid, eps=1e-6)
        self.gn1 = nn.BatchNorm1d(num_features=cmid, eps=1e-6)
        
        self.conv2 = conv3x3(cmid, cmid, stride=stride, bias=False)
        # self.gn2 = nn.GroupNorm(num_groups=ng, num_channels=cmid, eps=1e-6)
        self.gn2 = nn.BatchNorm1d(num_features=cmid, eps=1e-6)

        self.conv3 = conv1x1(cmid, cout, bias=False)
        # self.gn3 = nn.GroupNorm(num_groups=ng, num_channels=cout, eps=1e-6)
        self.gn3 = nn.BatchNorm1d(num_features=cout, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride=stride, bias=False)
            # self.ds_gn = nn.GroupNorm(num_groups=cout, num_channels=cout)
            self.ds_gn = nn.BatchNorm1d(num_features=cout)

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

def conv1x1(cin:int, cout:int, stride=1, bias=False) -> nn.Conv1d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=1, stride=stride, padding=0, bias=bias
    )

def conv3x3(cin:int, cout:int, stride=1, groups=1, bias=False) -> nn.Conv1d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=3, stride=stride, padding=1, bias=bias,
        groups=groups
    )

def conv7(cin:int, cout:int, stride=1, groups=1, bias=False) -> nn.Conv1d:
    return StdConv1d(
        in_channels=cin, out_channels=cout, kernel_size=7, stride=stride, padding=3, bias=bias, 
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