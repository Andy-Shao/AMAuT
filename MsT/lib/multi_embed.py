import torch
from torch import nn

from AuT.lib.rest_embed import RestNetBlock, StdConv1d

class FreqEmbedding(nn.Module):
    def __init__(self, num_channels:int, embed_size:int, marsked_rate:float, width=128, num_layers=[6,8], in_shape=[80,104]):
        super(FreqEmbedding, self).__init__()
        ng = 32
        assert width % ng == 0, 'width must be dividable by the num_groups in GroupNorm.'
        self.restnet = RestNet1d(cin=num_channels, width=width, ng=ng, num_layers=num_layers)
        self.drop_out = nn.Dropout(p=marsked_rate)
        self.patch_embedding = nn.Conv1d(in_channels=width*8, out_channels=embed_size, kernel_size=1, stride=1, padding=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_shape[0], in_shape[1]))
        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        x = self.drop_out(x)
        x = self.restnet(x)
        x = self.patch_embedding(x)
        x = x.transpose(2, 1)
        return x

class RestNet1d(nn.Module):
    def __init__(self, cin:int, width:int, ng:int, num_layers:list[int]):
        super(RestNet1d, self).__init__()
        self.root = nn.Sequential(
            StdConv1d(in_channels=cin, out_channels=width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(ng, width, eps=1e-6),
            nn.ReLU()
        )
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=0)
        
        self.layers = nn.ModuleList()
        # layer1
        self.layers.append(RestNetBlock(cin=width, cout=width*4, cmid=width, ng=ng))
        for _ in range(num_layers[0]): self.layers.append(RestNetBlock(cin=width*4, cout=width*4, cmid=width, ng=ng))
        # layer2
        self.layers.append(RestNetBlock(cin=width*4, cout=width*8, cmid=width*2, stride=2, ng=ng))
        for _ in range(num_layers[1]): self.layers.append(RestNetBlock(cin=width*8, cout=width*8, cmid=width*2, ng=ng))
        # layer3
        if len(num_layers) >= 3:
            self.layers.append(RestNetBlock(cin=width*8, cout=width*16, cmid=width*4, stride=2, ng=ng))
            for _ in range(num_layers[2]): self.layers.append(RestNetBlock(cin=width*16, cout=width*16, cmid=width*2, ng=ng))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.root(x)
        x = self.maxPool(x)
        for l in self.layers: x = l(x)
        return x
