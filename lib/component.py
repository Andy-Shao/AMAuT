import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    Copyright: https://github.com/meta-llama/llama.git
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L34
     
    Original research: Root Mean Square Layer Normalization (https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)
    """
    def __init__(self, normalized_shape:list[int], eps:float=1e-6, elementwise_affine:bool=True):
        super(RMSNorm, self).__init__()
        self.eps = eps 
        
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))

    def __norm__(self, x:torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.__norm__(x)
        if self.elementwise_affine:
            x = x * self.weight
        return x