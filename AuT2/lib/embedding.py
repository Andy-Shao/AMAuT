import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self) -> None:
        super(Embedding, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass

class StdLinear(nn.Linear):
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn import functional as F
        w = self.weight
        v, m = torch.var_mean(w, dim=[1], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.linear(input=x, weight=w, bias=self.bias)
