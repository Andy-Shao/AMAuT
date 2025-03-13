import random
from PIL import Image
import numpy as np

import torch 
import torch.nn as nn
import torchaudio

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
        assert transforms is not None, 'No support'
        self.transforms = nn.ModuleList(transforms)

    def forward(self, wavform: torch.Tensor) -> torch.Tensor:
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
    
class MelSpectrogramPadding(nn.Module):
    def __init__(self, target_length):
        super(MelSpectrogramPadding, self).__init__()
        self.target_length = target_length

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        p = self.target_length - x.shape[2]
        if p > 0:
            # padding = nn.ZeroPad1d((0, p, 0, 0))
            # x = padding(x)
            x = pad(x, (0, p, 0, 0), mode='constant', value=0.)
        elif p < 0:
            x = x[:, :, 0:self.target_length]
        return x

class FrequenceTokenTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        c, token_num, token_len = x.size()
        x = x.reshape(-1, token_len)
        return x
    
class VisionTokenTransformer(nn.Module):
    def __init__(self, kernel_size=(16, 20), stride=(8, 10)):
        super(VisionTokenTransformer, self).__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        x = x.transpose(1, 0)
        return x
    
class AudioPadding(nn.Module):
    def __init__(self, max_length:int, sample_rate:int, random_shift:bool=False):
        super(AudioPadding, self).__init__()
        self.max_length = max_length
        self.random_shift = random_shift

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        l = self.max_length - x.shape[1]
        if l > 0:
            if self.random_shift:
                head = random.randint(0, l)
                tail = l - head
            else:
                head = l // 2
                tail = l - head
            x = pad(x, (head, tail), mode='constant', value=0.)
        return x
    
class AudioClip(nn.Module):
    def __init__(self, max_length:int, mode:str='head', is_random:bool=False):
        super(AudioClip, self).__init__()
        assert mode in ['head', 'mid', 'tail']
        self.max_length = max_length
        self.mode = mode
        self.is_random = is_random

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        l = x.shape[1] - self.max_length
        if l > 0:
            if self.is_random:
                start = np.random.randint(low=0, high=l)
            elif self.mode == 'head':
                start = 0
            elif self.mode == 'mid':
                start = int(start/2.)
            elif self.mode == 'tail':
                start = l
            x = x[:, start:start+self.max_length]
        return x

class RandomPitchShift(nn.Module):
    def __init__(
            self, step_rang:list[int], sample_rate:int, bins_per_octave:int=12, n_fft:int=512, win_length:int=None,
            hop_length:int=None
        ):
        from torchaudio.transforms import PitchShift
        super(RandomPitchShift, self).__init__()
        self.shifts = nn.ModuleList([
            PitchShift(
                sample_rate=sample_rate, n_steps=item, bins_per_octave=bins_per_octave, n_fft=n_fft, 
                win_length=win_length, hop_length=hop_length
            ) if item != 0 else DoNothing() for item in step_rang
        ])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        shift = self.shifts[random.randint(0, len(self.shifts)-1)]
        x = shift(x)
        return x
    
class RandomVol(nn.Module):
    def __init__(self, gains:list[float], gain_type:str='amplitude'):
        from torchaudio.transforms import Vol
        super(RandomVol, self).__init__()
        self.vols = nn.ModuleList([Vol(gain=i, gain_type=gain_type) for i in gains])

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        vol = self.vols[random.randint(0, len(self.vols)-1)]
        x = vol(x)
        return x
    
class RandomTimeMask(nn.Module):
    def __init__(self, cfgs:list[dict]):
        super(RandomTimeMask, self).__init__()
        self.cfgs = cfgs
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        from torchaudio.functional import mask_along_axis
        
        for cfg in self.cfgs:
            x = mask_along_axis(specgram=x, mask_param=cfg['mask_param'], mask_value=0., axis=2, p=cfg['p'])
        return x
    
class Fbank(nn.Module):
    def __init__(self, sample_rate:int, num_mel_bins:int, window_type='hanning', dither=0.0, frame_shift=10):
        super(Fbank, self).__init__()
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.windown_type = window_type
        self.dither = dither
        self.frame_shift = frame_shift

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        wavform = wavform - wavform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform=wavform, htk_compat=False, sample_frequency=self.sample_rate, use_energy=False,
            window_type=self.windown_type, num_mel_bins=self.num_mel_bins, dither=self.dither,
            frame_shift=self.frame_shift
        )
        fbank = fbank.transpose(1, 0)
        return fbank

class FbankPadding(nn.Module):
    def __init__(self, target_length):
        super(FbankPadding, self).__init__()
        assert target_length > 0, 'No support'
        self.target_length = target_length

    def forward(self, fbank:torch.Tensor) -> torch.Tensor:
        from torch.nn.functional import pad
        l = self.target_length - fbank.shape[1]
        if l > 0:
            assert l <= 3, 'target_length is too large'
            fbank = pad(fbank, (0, l), mode='constant', value=0.)
        elif l < 0:
            fbank = fbank[:, 0:self.target_length]
        return fbank
    
class RandomSpeed(nn.Module):
    def __init__(self, start_fq:float, end_fq:float, sample_rate:int, max_length:int=-1, step:float=.01, max_it:int=10):
        super().__init__()
        self.speeds = nn.ModuleList()
        self.speed_up_ls = nn.ModuleList()
        fq = start_fq
        while fq < end_fq:
            item = torchaudio.transforms.Speed(orig_freq=sample_rate, factor=fq)
            self.speeds.append(item)
            if fq >= 1.0: self.speed_up_ls.append(item)
            fq += step
        self.max_length = max_length
        self.max_it = max_it

    def __random_speed__(self, wavform:torch.Tensor) -> torch.Tensor:
        if self.max_length != -1 and wavform.shape[1] >= self.max_length:
            if len(self.speed_up_ls) == 0: return wavform
            tf = self.speed_up_ls[np.random.randint(0, len(self.speed_up_ls))]
        else:
            tf = self.speeds[np.random.randint(0, len(self.speeds))]
        new_wav, _ = tf(wavform)
        return new_wav

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        it_num = 0
        while True:
            ret = self.__random_speed__(wavform=wavform)
            if self.max_length == -1 or ret.shape[1] <= self.max_length:
                return ret
            elif it_num >= self.max_it:
                return wavform
            else: 
                it_num += 1
                continue

class BatchTransform(nn.Module):
    def __init__(self, tf:nn.Module):
        super(BatchTransform, self).__init__()
        self.tf = tf

    def forward(self, inputs:torch.Tensor) -> torch.Tensor:
        outputs = []
        for i in range(inputs.shape[0]):
            output = self.tf(inputs[i])
            outputs.append(torch.unsqueeze(output, dim=0))
        outputs = torch.cat(outputs, dim=0)
        return outputs
    
class Stereo2Mono(nn.Module):
    def __init__(self):
        super(Stereo2Mono, self).__init__()

    def forward(self, wavform:torch.Tensor) -> torch.Tensor:
        return wavform.mean(dim=0, keepdim=True)