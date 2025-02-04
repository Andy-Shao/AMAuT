from ml_collections import ConfigDict

import torch
from torch import nn

class FreqEmbedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, marsked_rate:float, width=128, num_layers=[6,8], in_shape=[80,104]):
        from AuT.lib.rest_embed import RestNetCT
        super(FreqEmbedding, self).__init__()
        ng = 32
        assert width % ng == 0, 'width must be dividable by the num_groups in GroupNorm.'
        self.restnet = RestNetCT(cin=num_channels, width=width, ng=ng, num_layers=num_layers)
        self.drop_out = nn.Dropout(p=marsked_rate)
        self.patch_embedding = nn.Conv1d(in_channels=width*(2**(1+len(in_shape))), out_channels=embed_size, kernel_size=1, stride=1, padding=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_shape[0], in_shape[1]))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        x = self.drop_out(x)
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)
        return x
    
    def build(cfg:ConfigDict, tag:str):
        c = cfg[tag]
        return FreqEmbedding(
            num_channels=c.num_channels, embed_size=c.embed_size, marsked_rate=c.marsked_rate,
            width=c.width, num_layers=c.num_layers, in_shape=c.in_shape
        )
    
class VisionTransformerEmbedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, marsked_rate:float, width=64, num_layers=[6,8], in_shape=[80,100]):
        super(VisionTransformerEmbedding, self).__init__()
        ng = 32
        assert width % ng == 0, 'width must be dividable by the num_groups in GroupNorm.'
        self.restnet = RestNet2d(cin=num_channels, width=width, ng=ng, num_layers=num_layers)
        self.drop_out = nn.Dropout(p=marsked_rate)
        self.patch_embedding = nn.Conv2d(in_channels=width*(2**(1+len(in_shape))), out_channels=embed_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_shape[0], in_shape[1]))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        x = self.drop_out(x)
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)
        return x
    
    def build(cfg:ConfigDict, tag:str):
        c = cfg[tag]
        return VisionTransformerEmbedding(
            num_channels=c.num_channels, embed_size=c.embed_size, marsked_rate=c.marsked_rate, 
            width=c.width, num_layers=c.num_layers, in_shape=c.in_shape
        )

class RestNet2d(nn.Module):
    def __init__(self, cin:int, width:int, ng:int, num_layers:list[int]):
        from MsT.lib.restnet import RestNetBlock, StdConv2d

        super(RestNet2d, self).__init__()
        self.root = nn.Sequential(
            StdConv2d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(ng, width, eps=1e-6),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        w = width
        self.layers = nn.ModuleList()
        for i, layer_num in enumerate(num_layers):
            if i == 0:
                self.layers.append(RestNetBlock(cin=w, cout=w*4, cmid=w, ng=ng))
                for _ in range(layer_num): self.layers.append(RestNetBlock(cin=w*4, cout=w*4, cmid=w, ng=ng))
                w = w * 4
            else:
                self.layers.append(RestNetBlock(cin=w, cout=w*2, cmid=w//2, stride=2, ng=ng))
                for _ in range(layer_num): self.layers.append(RestNetBlock(cin=w*2, cout=w*2, cmid=w//2, ng=ng))
                w = w * 2

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layers: x = l(x)
        return x