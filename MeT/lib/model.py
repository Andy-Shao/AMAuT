from ml_collections import ConfigDict

import torch
from torch import nn

from .multi_embed import Embedding
from AuT.lib.model import AttentionBlock

class MultiEmbedTransformer(nn.Module):
    def __init__(self, cfg: ConfigDict):
        super(MultiEmbedTransformer, self).__init__()
        embed_size = cfg.embedding['embed_size']
        self.embed1 = Embedding(
            num_channels=cfg.embed1['num_channels'], embed_size=embed_size, marsked_rate=cfg.embed1['marsked_rate'],
            width=cfg.embed1['width'], num_layers=cfg.embed1['num_layers'], in_shape=cfg.embed1['in_shape']
        )
        self.embed2 = Embedding(
            num_channels=cfg.embed2['num_channels'], embed_size=embed_size, marsked_rate=cfg.embed2['marsked_rate'],
            width=cfg.embed2['width'], num_layers=cfg.embed2['num_layers'], in_shape=cfg.embed2['in_shape']
        )
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