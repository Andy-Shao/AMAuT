from collections import OrderedDict

import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, token_len:int, embed_size:int, marsked_rate:float=.1) -> None:
        super(Embedding, self).__init__()

        width = embed_size // 4

        self.ol1 = nn.Sequential(OrderedDict(
            [('l0', MlpBlock(fin=token_len, fout=width, fmid=token_len))] +
            [(f'l{i:d}', MlpBlock(fin=width, fout=width, fmid=token_len)) for i in range(1,3)]
        ))
        self.ol2 = nn.Sequential(OrderedDict(
            [('l0', MlpBlock(fin=width, fout=width*2, fmid=width))] +
            [(f'l{i:d}', MlpBlock(fin=width*2, fout=width*2, fmid=width)) for i in range(1,6)]
        ))
        self.ol3 = nn.Sequential(OrderedDict(
            [('l0', MlpBlock(fin=width*2, fout=embed_size, fmid=width*2))] +
            [(f'l{i:d}', MlpBlock(fin=embed_size, fout=embed_size, fmid=width*2)) for i in range(1,3)]
        ))

        self.drop_out = nn.Dropout(p=marsked_rate)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = build_sentence_by_channel(x)
        x = self.ol1(x)
        x = self.ol2(x)
        x = self.ol3(x)
        x = self.drop_out(x)
        return x
    
def build_sentence_by_channel(x:torch.Tensor) -> torch.Tensor:
    if len(x.shape) == 4:
        batch_size, channel_num, token_num, token_len = x.size()
        out = []
        for i in range(channel_num):
            out.append(x[:, i, :, :])
            if i > 1:
                out.append(torch.zeros(batch_size, 1, token_len))
        return torch.concat(out, dim=1)
    elif len(x.shape) == 3:
        channel_num, token_num, token_len = x.size()
        out = []
        for i in range(channel_num):
            out.append(x[i, :, :])
            if i>1:
                out.append(torch.zeros(1, token_len))
        return torch.concat(out, dim=1)
    else: raise Exception('No support')

class MlpBlock(nn.Module):
    def __init__(self, fin:int, fout:int, fmid:int):
        super(MlpBlock, self).__init__()
        self.fin = fin
        self.fout = fout
        self.l1 = StdLine(in_features=fin, out_features=fmid, bias=False)
        self.nm1 = nn.LayerNorm(fmid, eps=1e-6)
        self.l2 = StdLine(in_features=fmid, out_features=fmid, bias=True if fin != fout else False)
        self.nm2 = nn.LayerNorm(fmid, eps=1e-6)
        self.l3 = StdLine(in_features=fmid, out_features=fout, bias=False)
        self.nm3 = nn.LayerNorm(fout, eps=1e-6)
        self.gelu = nn.GELU()

        if fin != fout:
            self.upsample = StdLine(in_features=fin, out_features=fout, bias=False)
            self.us_nm = nn.LayerNorm(fout, eps=1e-6)

        self.init_weight()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residural = x
        if hasattr(self, 'upsample'):
            residural = self.upsample(residural)
            residural = self.us_nm(residural)

        y = self.gelu(self.nm1(self.l1(x)))
        y = self.gelu(self.nm2(self.l2(y)))
        y = self.nm3(self.l3(y))
        return self.gelu(residural + y)

    def init_weight(self) -> None:
        nn.init.xavier_uniform_(self.l1.weight)
        nn.init.xavier_uniform_(self.l2.weight)
        nn.init.xavier_uniform_(self.l3.weight)
        if self.fin != self.fout:
            nn.init.normal_(self.l2.bias)     
        if hasattr(self, 'upsample'):
            nn.init.xavier_uniform_(self.upsample.weight)

class StdLine(nn.Linear):
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        w = self.weight
        var, mean = torch.var_mean(w, dim=[1], keepdim=True, unbiased=False)
        w = (w - mean) / torch.sqrt(var + 1e-6)
        x = nn.functional.linear(input=x, weight=w, bias=self.bias)
        return x