import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self) -> None:
        super(Embedding, self).__init__()
        self.block1 = MlpBlock()
        self.block2 = MlpBlock()
        self.block3 = MlpBlock()

        self.out_line = nn.Linear(in_features=-1, out_features=2048)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.out_line(x)

class MlpBlock(nn.Module):
    def __init__(self, fin:int, fout:int, fmid:int):
        super(MlpBlock, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=-1, num_channels=-1, eps=1e-6)
        self.l1 = nn.Linear(in_features=fin, out_features=fmid)
        self.gn2 = nn.GroupNorm(num_groups=-1, num_channels=-1, eps=1e-6)
        self.l2 = nn.Linear(in_features=fmid, out_features=fmid)
        self.gn3 = nn.GroupNorm(num_groups=-1, num_channels=-1, eps=1e-6)
        self.l3 = nn.Linear(in_features=fmid, out_features=fout)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Linear(in_features=fin, out_features=fout)
        self.up_gn = nn.GroupNorm(num_groups=-1, num_channels=-1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        residual = self.up_gn(self.upsample(x))
        y = self.relu(self.gn1(self.l1(x)))
        y = self.relu(self.gn2(self.l2(y)))
        y = self.gn3(self.l3(y))
        return self.relu(y + residual)
    
class TokenConv(nn.Module):
    def __init__(self, kernal_size:int, stride:int, padding:int) -> None:
        super(TokenConv, self).__init__()
        self.lines = [
            nn.Linear(in_features=-1, out_features=-1, bias=False) for i in range(kernal_size)
        ]
        self.kernal_size = kernal_size
        self.stride = stride
        self.padding = padding

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, token_num, token_len = x.size()
        line_outs = [
            self.lines[i](x) for i in range(self.kernal_size)
        ]
        padding_head = [
            torch.zeros_like(line_outs[0]).to(x.device) for i in range(self.padding)
        ]
        padding_tail = [
            torch.zeros_like(line_outs[0]).to(x.device) for i in range(self.padding)
        ]
        line_outs = padding_head + line_outs + padding_tail

        batch_id = 0
        line_outs[0][batch_id][0] + line_outs[1][batch_id][1] + line_outs[2][batch_id][2]
        line_outs[0][batch_id][1] + line_outs[1][batch_id][2] + line_outs[2][batch_id][3]

        line_outs[0][batch_id][:] + line_outs[1][batch_id][1:] + line_outs[2][batch_id][2:]