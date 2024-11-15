import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_channels:int, token_len:int, embed_size:int) -> None:
        super(Embedding, self).__init__()
        if num_channels > 1:
            self.blocks = nn.ModuleList([MlpBlock(fin=token_len, fout=token_len) for i in range(num_channels)])

        self.ol1 = nn.ModuleList()
        for i in range(3):
            if i == 0:
                self.ol1.append(MlpBlock(fin=token_len, fout=token_len*2))
            else:
                self.ol1.append(MlpBlock(fin=token_len*2, fout=token_len*2))
        self.ol2 = nn.ModuleList()
        for i in range(4):
            if i == 0:
                self.ol2.append(MlpBlock(fin=token_len*2, fout=token_len*4))
            else:
                self.ol2.append(MlpBlock(fin=token_len*4, fout=token_len*4))
        self.ol3 = nn.ModuleList()
        for i in range(9):
            if i == 0:
                self.ol3.append(MlpBlock(fin=token_len*4, fout=embed_size))
            else:
                self.ol3.append(MlpBlock(fin=embed_size, fout=embed_size))

        self.drop_out = nn.Dropout(p=.1)

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
        for ol in self.ol1:
            x = ol(x)
        for ol in self.ol2:
            x = ol(x)
        for ol in self.ol3:
            x = ol(x)
        x = self.drop_out(x)
        return x

class MlpBlock(nn.Module):
    def __init__(self, fin:int, fout:int):
        super(MlpBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.l1 = StdLine(in_features=fin, out_features=fin, bias=False)
        self.nm1 = nn.LayerNorm(fin, eps=1e-6)
        self.l2 = StdLine(in_features=fin, out_features=fout, bias=False)
        self.nm2 = nn.LayerNorm(fout, eps=1e-6)
        self.l3 = StdLine(in_features=fout, out_features=fout, bias=False)
        self.nm3 = nn.LayerNorm(fout, eps=1e-6)
        self.gelu = nn.GELU()

        self.init_weight()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.gelu(self.nm1(self.l1(x)))
        y = self.gelu(self.nm2(self.l2(y)))
        y = self.nm3(self.l3(y))
        return self.gelu(y) if self.fin != self.fout else self.gelu(y + x)

    def init_weight(self) -> None:
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)     

class StdLine(nn.Linear):
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w = self.weight
        var, mean = torch.var_mean(w, dim=[1], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-6)
        x = nn.functional.linear(input=x, weight=w, bias=self.bias)
        return x