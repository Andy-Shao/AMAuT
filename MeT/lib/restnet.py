import torch 
from torch import nn

class RestNetBlock(nn.Module):
    def __init__(self, cin:int, cout:int, cmid:int, ng:int, stride=1):
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

class StdConv2d(nn.Conv2d):
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        w = self.weight
        var, mean = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-5)

        return F.conv2d(
            x, weight=w, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, 
            groups=self.groups
        )

def conv1x1(in_channels:int, out_channels:int, stride=1, groups=1, bias=False) -> StdConv2d:
    return StdConv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=bias
    )

def conv3x3(in_channels:int, out_channels:int, stride:int=1, groups:int=1, bias:bool=False) -> StdConv2d:
    return StdConv2d(
        in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=bias, groups=groups
    )