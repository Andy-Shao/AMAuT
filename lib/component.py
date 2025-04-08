import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    " Copyright: https://github.com/meta-llama/llama.git
    " https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
    """
    def __init__(self, dim:int, eps:float=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))

    def __norm__(self, x:torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.__norm__(x)
        x = x * self.weight
        return x