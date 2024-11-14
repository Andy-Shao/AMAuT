import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_channels:int, token_len:int) -> None:
        super(Embedding, self).__init__()
        if num_channels > 1:
            self.blocks = nn.ModuleList([MlpBlock(fin=token_len, fout=token_len) for i in range(num_channels)])

        self.ol1 = MlpBlock(fin=token_len, fout=token_len*2)
        self.ol2 = MlpBlock(fin=token_len*2, fout=token_len*4)
        self.ol3 = MlpBlock(fin=token_len*4, fout=1024)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, channels, token_num, token_len = x.size()
        if channels > 1:
            for i in range(channels):
                if i == 0:
                    out = self.blocks[i](x[:, i-1:i, :, :])
                else:
                    out += self.blocks[i](x[:, i-1:i, :, :])
        else: out = x
        x = out.squeeze_(dim=1)
        x = self.ol1(x)
        x = self.ol2(x)
        x = self.ol3(x)
        return x

class MlpBlock(nn.Module):
    def __init__(self, fin:int, fout:int, num_channels:int):
        super(MlpBlock, self).__init__()
        self.l1 = StdLine(in_features=fin, out_features=fin, bias=False)
        self.nm1 = nn.InstanceNorm2d(num_features=num_channels, affine=True)
        self.l2 = StdLine(in_features=fin, out_features=fout, bias=True)
        self.nm2 = nn.InstanceNorm2d(num_features=num_channels, affine=True)
        self.l3 = StdLine(in_features=fout, out_features=fout, bias=False)
        self.nm3 = nn.InstanceNorm2d(num_features=num_channels, affine=True)
        self.gelu = nn.functional.gelu

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.nm1(self.l1(x)))
        x = self.gelu(self.nm2(self.l2(x)))
        x = self.nm3(self.l3(x))
        return x        

class StdLine(nn.Linear):
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w = self.weight
        var, mean = torch.var_mean(w, dim=(1,2,3), keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-6)
        x = nn.functional.linear(input=x, weight=w, bias=self.bias)
        return x