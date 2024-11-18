import torch 
import torch.nn as nn

class AudioTokenTransformer(nn.Module):
    def __init__(self) -> None:
        super(AudioTokenTransformer, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2)
    
class Conv1ChannelMerge(nn.Module):
    def __init__(self) -> None:
        super(Conv1ChannelMerge, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        batch_size, channel_num, token_num, token_len = x.size()
        return x.reshape(batch_size, -1, token_len)