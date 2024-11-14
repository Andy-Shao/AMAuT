import math

import torch 
import torch.nn as nn
from ml_collections import  ConfigDict

class AudioTransform(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(AudioTransform, self).__init__()
        embed_size = config.transform['embed_size']
        self.tf_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.transform['layer_num'])])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.tf_norm(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(AttentionBlock, self).__init__()
        embed_size = config.transform['embed_size']
        self.attention_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.attention = MultiHeadAttention(
            d=embed_size, h=config.transform['head_num'], at_dp=config.transform['atten_drop_rate'])
        self.ffn = MultilayerPerceptron(
            fin=config.transform['mlp_in'], fmid=config.transform['mlp_mid'], fout=config.transform['mlp_out'],
            dp_rt=config.transform['mlp_dp_rt']
        )
        self.ffn_norm = nn.LayerNorm(embed_size, eps=1e-6)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = self.attention_norm(x)
        y = self.attention(y)
        x = y + x

        y = self.ffn_norm(x)
        y = self.ffn(x)
        x = y + x
        return x

class MultilayerPerceptron(nn.Module):
    def __init__(self, fin:int, fmid:int, fout:int, dp_rt:float=.5) -> None:
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(in_features=fin, out_features=fmid)
        self.fc2 = nn.Linear(in_features=fmid, out_features=fout)
        self.act_fn = nn.functional.gelu
        self.drop_out = nn.Dropout(p=dp_rt)

        self.init_weight()

    def init_weight(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        x = self.drop_out(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d:int, h:int, at_dp:float=.5) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d % h == 0, 'it is not divisible'
        self.d = d
        self.h = h
        self.d_k, self.d_v = d/h, d/h

        self.q = nn.Linear(in_features=d, out_features=d)
        self.k = nn.Linear(in_features=d, out_features=d)
        self.v = nn.Linear(in_features=d, out_features=d)
        self.atten_drop = nn.Dropout(p=at_dp)

        self.o = nn.Linear(in_features=d, out_features=d)

    def forward(self, x:torch.Tensor, mask=None) -> torch.Tensor:
        from torch.nn import functional as F
        batch_size, seq_length, embed_size = x.size()

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.h, self.d_v).transpose(1, 2)

        # attention = self.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        attention = F.scaled_dot_product_attention(
            query=Q, key=K, value=V, is_causal=True
        )
        attention = self.atten_drop(attention)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        return self.o(attention)

    # def scaled_dot_product_attention(self, Q, K, V, mask=None) -> torch.Tensor:
    #     scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    #     if mask is not None:
    #         scores = scores.masked_fill(mask= mask == 0, value=float('-inf'))
        
    #     attention = torch.softmax(scores, dim=-1)

    #     return torch.matmul(attention, V)
