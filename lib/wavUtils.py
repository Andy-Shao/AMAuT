import random
from PIL import Image
import numpy as np

import torch 
import torch.nn as nn

class BackgroundNoise(nn.Module):
    def __init__(self, noise_level: float, noise: torch.Tensor, is_random=False):
        super().__init__()
        self.noise_level = noise_level
        self.noise = noise
        self.is_random = is_random

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        import torchaudio.functional as ta_f
        wav_len = wavform.shape[1]
        if self.is_random:
            start_point = np.random.randint(low=0, high=self.noise.shape[1]-wav_len)
            noise_period = self.noise[:, start_point:start_point+wav_len]
        else:
            noise_period = self.noise[:, 0:wav_len]
        noised_wavform = ta_f.add_noise(waveform=wavform, noise=noise_period, snr=torch.tensor([self.noise_level]))
        return noised_wavform

class GuassianNoise(nn.Module):
    def __init__(self, noise_level=.05):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        ## Guassian Noise
        noise = torch.rand_like(wavform) * self.noise_level
        noise_wavform = wavform + noise
        return noise_wavform

class pad_trunc(nn.Module):
    """
    Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    """
    def __init__(self, max_ms: float, sample_rate: int) -> None:
        super().__init__()
        assert max_ms > 0, 'max_ms must be greater than zero'
        assert sample_rate > 0, 'sample_rate must be greater than zeror'
        self.max_ms = max_ms
        self.sample_rate = sample_rate

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        channel_num, wav_len = wavform.shape
        max_len = self.sample_rate // 1000 * self.max_ms

        if (wav_len > max_len):
            wavform = wavform[:, :max_len]
        elif wav_len < max_len:
            head_len = random.randint(0, max_len - wav_len)
            tail_len = max_len - wav_len - head_len

            head_pad = torch.zeros((channel_num, head_len))
            tail_pad = torch.zeros((channel_num, tail_len))

            wavform = torch.cat((head_pad, wavform, tail_pad), dim=1)
        return wavform

class Components(nn.Module):
    def __init__(self, transforms: list) -> None:
        super().__init__()
        self.transforms = transforms

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.transforms is None:
            return None
        for transform in self.transforms:
            wavform = transform(wavform)
        return wavform

class time_shift(nn.Module):
    def __init__(self, shift_limit: float, is_random=True, is_bidirection=False) -> None:
        """
        Time shift data augmentation

        :param shift_limit: shift_limit -> (-1, 1), shift_limit < 0 is left shift
        """
        super().__init__()
        self.shift_limit = shift_limit
        self.is_random = is_random
        self.is_bidirection = is_bidirection

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
        if self.is_random:
            shift_arg = int(random.random() * self.shift_limit * wavform.shape[1])
            if self.is_bidirection:
                shift_arg = int((random.random() * 2 - 1) * self.shift_limit * wavform.shape[1])
        else:
            shift_arg = int(self.shift_limit * wavform.shape[1])
        return wavform.roll(shifts=shift_arg)
    
def display_wavform(waveform: torch.Tensor, title:str='Audio Waveform'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(waveform.numpy().T)
    plt.title(title)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.show()

def display_spectro_gram(waveform: torch.Tensor, title='Mel Spectrogram in channel 0'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,4))
    plt.imshow(waveform[0].detach().numpy(), cmap='viridis', origin='lower', aspect='auto')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

def disply_PIL_image(img: Image, title='Mel Spectrogram in channel 0'):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,4))
    plt.imshow(np.asarray(img), cmap='viridis', origin='lower', aspect='auto')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

class DoNothing(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
    
class AmplitudeToDB(nn.Module):
    def __init__(self, top_db:float, max_out:float) -> None:
        from torchaudio import transforms
        super(AmplitudeToDB, self).__init__()
        self.model = transforms.AmplitudeToDB(top_db=top_db)
        self.max_out = max_out
        self.top_db = top_db

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x) / (self.top_db // self.max_out)