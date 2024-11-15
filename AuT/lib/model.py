import math

import torch 
import torch.nn as nn
from ml_collections import  ConfigDict

from .embedding import Embedding

def init_weights(m: nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1., .02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class AudioClassifier(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(AudioClassifier, self).__init__()
        embed_size = config.transform['embed_size']
        extend_size = config.classifier['extend_size']
        convergent_size = config.classifier['convergent_size']
        self.fc1 = nn.Linear(in_features=embed_size, out_features=extend_size, bias=True)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc2 = nn.Linear(in_features=extend_size, out_features=convergent_size)
        self.bn = nn.BatchNorm1d(num_features=convergent_size, affine=True)
        self.fc2.apply(init_weights)
        self.fc3 = nn.utils.parametrizations.weight_norm(
            module=nn.Linear(in_features=convergent_size, out_features=config.classifier['class_num']), name='weight')
        self.fc3.apply(init_weights)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, token_num, token_len = x.size()
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = x.contiguous().view(batch_size, token_len)
        x = self.fc1(x)

        x = self.fc2(x)
        x = self.bn(x)
        x = self.fc3(x)

        return x

class AudioTransform(nn.Module):
    def __init__(self, config:ConfigDict) -> None:
        super(AudioTransform, self).__init__()
        embed_size = config.transform['embed_size']
        self.embedding = Embedding(
            num_channels=config.embedding['channel_num'], token_len=config.embedding['in_token_len'],
            embed_size=embed_size
        )
        self.tf_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.transform['layer_num'])])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
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
            fin=embed_size, fmid=config.transform['mlp_mid'], fout=config.transform['mlp_out'],
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
        self.act_fn = nn.GELU()
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
        self.d_k, self.d_v = int(d/h), int(d/h)

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
