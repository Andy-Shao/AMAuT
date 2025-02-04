from ml_collections import ConfigDict

import torch
from torch import nn

from AuT.lib.model import AttentionBlock

class BiEmbedTransformer(nn.Module):
    def __init__(self, cfg: ConfigDict, embed1:nn.Module, embed2:nn.Module):
        super(BiEmbedTransformer, self).__init__()
        embed_size = cfg.embedding['embed_size']
        self.embed1 = embed1
        self.embed2 = embed2
        self.sep = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.tf_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.layers = nn.ModuleList([AttentionBlock(cfg) for _ in range(cfg.transform['layer_num'])])

    def forward(self, x1:torch.Tensor, x2:torch.Tensor) -> torch.Tensor:
        x1, x2 = self.embed1(x1), self.embed2(x2)
        sep = self.sep.expand(x1.shape[0], -1, -1)
        x = torch.cat([x1, sep, x2], dim=1)
        for l in self.layers: x = l(x)
        x = self.tf_norm(x)
        return x