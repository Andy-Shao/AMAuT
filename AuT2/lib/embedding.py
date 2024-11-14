import torch 
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_channels:int, token_len:int) -> None:
        super(Embedding, self).__init__()
        if num_channels > 1:
            self.blocks = nn.ModuleList([MlpBlock(num_features=token_len) for i in range(num_channels)])

        self.out_line = MlpBlock(num_features=token_len)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, channels, token_num, token_len = x.size()
        if channels > 1:
            for i in range(channels):
                if i == 0:
                    out = self.blocks[i](x[:, i-1:i, :, :])
                else:
                    out += self.blocks[i](x[:, i-1:i, :, :])
        else: out = x
        out = self.out_line(out)
        return torch.squeeze(input=out, dim=1)

class MlpBlock(nn.Module):
    def __init__(self, num_features:int):
        super(MlpBlock, self).__init__()
        self.l1 = nn.Linear(in_features=num_features, out_features=num_features, bias=True)
        self.relu = nn.GELU()
        self.l2 = nn.Linear(in_features=num_features, out_features=num_features, bias=True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.l2(self.relu(self.l1(x)))