import math

import torch 
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d:int, h:int) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d % h == 0, 'it is not divisible'
        self.d = d
        self.h = h
        self.d_k, self.d_v = d/h, d/h

        self.q = nn.Linear(in_features=d, out_features=d)
        self.k = nn.Linear(in_features=d, out_features=d)
        self.v = nn.Linear(in_features=d, out_features=d)

        self.o = nn.Linear(in_features=d, out_features=d)

    def forward(self, x:torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, channel_size, seq_length, embed_size = x.size()

        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)

        Q = Q.view(batch_size, channel_size, seq_length, self.h, self.d_k).transpose(2, 3)
        K = K.view(batch_size, channel_size, seq_length, self.h, self.d_k).transpose(2, 3)
        V = V.view(batch_size, channel_size, seq_length, self.h, self.d_v).transpose(2, 3)

        attention = self.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)
        attention = attention.transpose(2, 3).contiguous().view(batch_size, channel_size, seq_length, embed_size)

        return self.o(attention)

    def scaled_dot_product_attention(self, Q, K, V, mask=None) -> torch.Tensor:
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask= mask == 0, value=float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)

        return torch.matmul(attention, V)
