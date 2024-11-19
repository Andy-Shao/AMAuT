import torch 
import torch.nn as nn

class AudioTokenTransformer(nn.Module):
    def __init__(self) -> None:
        super(AudioTokenTransformer, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)