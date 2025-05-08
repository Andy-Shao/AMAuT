from lib.spDataset import VocalSound
from lib.wavUtils import Components, AudioClip, AudioPadding, AmplitudeToDB
from torchaudio import transforms as a_transforms

sample_rate=16000
max_length = sample_rate * 10
n_mels=64
n_fft=1024
win_length=400
hop_length=154
mel_scale='slaney'
tfs = Components(transforms=[
    AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=max_length),
    AudioClip(max_length=max_length, mode='head', is_random=False),
    a_transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        mel_scale=mel_scale
    ), # 80 x 1026
    AmplitudeToDB(top_db=80., max_out=2.),
])
train_set = VocalSound(
    root_path='/root/data/vocalsound_16k', mode='train', include_rate=False, version='16k', data_tf=tfs
)
print(f'feature shape is: {train_set[0][0].shape}')