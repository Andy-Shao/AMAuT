import os
from typing import Any

from torch.utils.data import Dataset
import torch
import torchaudio

class RandomSpeechCommandsDataset(Dataset):
    test_meta_file = 'rand_testing_list.txt'
    val_meta_file = 'rand_validation_list.txt'
    train_meta_file = 'rand_train_list.txt'

    def __init__(
            self, root_path: str, mode: str, include_rate=True, data_tfs=None, data_type='all', seed:int = 2024, source_mode:str ='train',
            output_path: str = './result/speech-commands-random', refresh=False
        ) -> None:
        super().__init__()
        if refresh:
            try:
                os.remove(os.path.join(output_path), self.test_meta_file)
                os.remove(os.path.join(output_path), self.val_meta_file)
                os.remove(os.path.join(output_path), self.train_meta_file)
            except:
                pass
        self.dataset = SpeechCommandsDataset(root_path=root_path, mode=source_mode, include_rate=include_rate, data_tfs=data_tfs, data_type=data_type)
        self.seed = seed
        assert mode in ['train', 'validation', 'test', 'full', 'test+val'], 'mode type is incorrect'
        self.mode = mode
        self.output_path = output_path
        self.__generate_random_meta_file__(seed=seed)

        if mode == 'train':
            data_list = self.train_indexes
        elif mode == 'validation':
            data_list = self.val_indexes
        elif mode == 'full':
            data_list = self.dataset.data_list
        elif mode == 'test+val':
            data_list = []
            data_list.extend(self.test_indexes)
            data_list.extend(self.val_indexes)
        elif mode == 'test':
            data_list = self.test_indexes
        self.dataset.data_list = data_list

    def __generate_random_meta_file__(self, seed:int) -> None:
        if os.path.exists(os.path.join(self.output_path, self.test_meta_file)):
            with open(os.path.join(self.output_path, self.test_meta_file), 'rt', newline='\n') as f:
                all_data = f.readlines()
            self.test_indexes = [line.rstrip('\n') for line in all_data]
            with open(os.path.join(self.output_path, self.val_meta_file), 'rt', newline='\n') as f:
                all_data = f.readlines()
            self.val_indexes = [line.rstrip('\n') for line in all_data]
            with open(os.path.join(self.output_path, self.train_meta_file), 'rt', newline='\n') as f:
                all_data = f.readlines()
            self.train_indexes = [line.rstrip('\n') for line in all_data]
        else:
            from numpy.random import MT19937, RandomState, SeedSequence
            rs = RandomState(MT19937(SeedSequence(seed)))
            data_list = self.dataset.data_list
            self.test_indexes = rs.choice(data_list, size=int(.3*len(data_list)), replace=False)
            residua = []
            for it in data_list:
                if it in self.test_indexes:
                    continue
                residua.append(it)
            self.train_indexes = rs.choice(residua, size=int(.9*len(residua)), replace=False)
            self.val_indexes = []
            for it in residua:
                if it in self.train_indexes:
                    continue
                self.val_indexes.append(it)
            with open(os.path.join(self.output_path, self.test_meta_file), 'wt', newline='\n') as f:
                for it in self.test_indexes:
                    f.write(it+'\n')
            with open(os.path.join(self.output_path, self.val_meta_file), 'wt', newline='\n') as f:
                for it in self.val_indexes:
                    f.write(it+'\n')
            with open(os.path.join(self.output_path, self.train_meta_file), 'wt', newline='\n') as f:
                for it in self.train_indexes:
                    f.write(it+'\n')

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        return self.dataset[index]

class SpeechCommandsDataset(Dataset):
    test_meta_file = 'testing_list.txt'
    val_meta_file = 'validation_list.txt'
    label_dic = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9., 'bed': 10., 'dog': 11., 'happy': 12., 'marvin': 13., 'off': 14., 
        'right': 15., 'up': 16., 'yes': 17., 'bird': 18., 'down': 19., 'house': 20., 'on': 21., 
        'stop': 22., 'tree': 23., 'cat': 24., 'go': 25., 'left': 26., 'no': 27., 'sheila': 28., 
        'wow': 29.
    }
    commands = {'yes':0., 'no':1., 'up':2., 'down':3., 'left':4., 'right':5., 'on':6., 'off':7., 'stop':8., 'go':9.}
    no_commands = {
        'zero':0., 'one':1., 'two':2., 'three':3., 'four':4., 'five':5., 'six':6., 'seven':7., 'eight':8., 'nine':9., 
        'bed':10., 'dog':11., 'happy':12., 'marvin':13., 'bird':14., 'house':15., 'tree':16., 'cat':17., 'sheila':18., 
        'wow':19.
    }
    numbers = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9.
    }

    def __init__(self, root_path: str, mode: str, include_rate=True, data_tfs=None, data_type='all', normalized:bool=False) -> None:
        super().__init__()
        self.root_path = root_path
        assert mode in ['train', 'validation', 'test', 'full', 'test+val'], 'mode type is incorrect'
        self.mode = mode
        assert data_type in ['all', 'commands', 'no_commands', 'numbers']
        self.data_type = data_type
        self.include_rate = include_rate
        data_list = self.__cal_data_list__(mode=mode)
        self.data_list = self.__filter_data_list__(data_list=data_list)
        self.data_tfs = data_tfs
        self.normalized = normalized
    
    def __filter_data_list__(self, data_list:list[str]) -> list[str]:
        if self.data_type == 'all':
            return data_list
        elif self.data_type == 'commands':
            filter_list = self.commands.keys()
        elif self.data_type == 'no_commands':
            filter_list = self.no_commands.keys()
        elif self.data_type == 'numbers':
            filter_list = self.numbers.keys()
        else:
            raise Exception('No support')
        new_data_list = []
        for it in data_list:
            label = it.strip().split('/')[0]
            if label in filter_list:
                new_data_list.append(it)
        return new_data_list


    def __cal_data_list__(self, mode: str) -> list[str]:
        if mode == 'validation':
            with open(os.path.join(self.root_path, self.val_meta_file), 'rt', newline='\n') as f:
                val_meta_data = f.readlines()
            return [line.rstrip('\n') for line in val_meta_data]
        elif mode == 'test':
            with open(os.path.join(self.root_path, self.test_meta_file), 'rt', newline='\n') as f:
                test_meta_data = f.readlines()
            return [line.rstrip('\n') for line in test_meta_data]
        elif mode == 'full':
            full_meta_data = []
            for k,v in self.label_dic.items():
                base_path = os.path.join(self.root_path, k)
                for p in os.listdir(base_path):
                    if p.endswith('.wav'):
                        full_meta_data.append(f'{k}/{p}')
            return full_meta_data
        elif mode == 'test+val':
            val_meta_data = self.__cal_data_list__(mode='validation')
            test_meta_data = self.__cal_data_list__(mode='test')
            for it in val_meta_data:
                test_meta_data.append(it)
            return test_meta_data
        else:
            val_meta_data = self.__cal_data_list__(mode='validation')
            test_meta_data = self.__cal_data_list__(mode='test')
            full_meta_data = self.__cal_data_list__(mode='full')
            train_meta_data = []
            for it in full_meta_data:
                if it in val_meta_data:
                    continue
                if it in test_meta_data:
                    continue
                train_meta_data.append(it)
            return train_meta_data

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> torch.Tensor:
        audio_path, label = self.__cal_audio_path_label__(self.data_list[index])
        audio, sample_rate = self.__load_wav__(audio_path=audio_path)
        if self.data_tfs is not None:
            audio = self.data_tfs(audio)
        if self.include_rate:
            return audio, label, sample_rate
        return audio, int(label)

    def __cal_audio_path_label__(self, meta_data: str) -> tuple[str, float]:
        label = meta_data.strip().split('/')[0]
        if self.data_type == 'all':
            label = self.label_dic[label]
        elif self.data_type == 'commands':
            label = self.commands[label]
        elif self.data_type == 'no_commands':
            label = self.no_commands[label]
        elif self.data_type == 'numbers':
            label = self.numbers[label]
        else:
            raise Exception('No support')
        audio_path = os.path.join(self.root_path, meta_data)
        return audio_path, label
    
    def __load_wav__(self, audio_path:str) -> tuple[torch.Tensor, int]:
        if self.normalized:
            # import numpy as np
            # from pydub import AudioSegment
            # from pydub.effects import normalize
            # import copy

            # audio = AudioSegment.from_wav(audio_path)
            # normalized_audio = normalize(audio, headroom=.1)
            # raw_data = normalized_audio.raw_data
            # num_channels = normalized_audio.channels
            # sample_width = normalized_audio.sample_width  # in bytes
            # frame_rate = normalized_audio.frame_rate
            # # num_samples = len(normalized_audio.get_array_of_samples()) // num_channels
            # audio_array = np.frombuffer(raw_data, dtype=np.int16 if sample_width == 2 else np.int8)
            # audio_array = audio_array.reshape(num_channels, -1)
            # audio_tensor = torch.from_numpy(copy.deepcopy(audio_array)).to(dtype=torch.float32)
            # return audio_tensor, frame_rate
            wavform, sample_rate = torchaudio.load(audio_path)
            if self.mode == 'train':
                wavform = wavform - (-8.724928e-05) + (-0.00017467)
            return wavform, sample_rate
        else:
            return torchaudio.load(audio_path)
        
class BackgroundNoiseDataset(Dataset):
    base_path = '_background_noise_'

    def __init__(self, root_path: str, data_tf=None, label_tf=None) -> None:
        super().__init__()
        self.root_path = root_path 
        self.data_list = self.__cal_data_list__()
        self.data_tf = data_tf
        self.label_tf = label_tf

    def __cal_data_list__(self) -> list[str]:
        data_list = []
        for it in os.listdir(os.path.join(self.root_path, self.base_path)):
            if it.endswith('.wav'):
                data_list.append(it)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index) -> Any:
        noise_path = os.path.join(self.root_path, self.base_path, self.data_list[index])
        noise, sample_rate = torchaudio.load(noise_path)
        noise_type = self.data_list[index][:-(len('.wav'))]
        if self.data_tf is not None:
            noise = self.data_tf(noise)
        if self.label_tf is not None:
            sample_rate = self.label_tf(sample_rate)
        return noise_type, noise, sample_rate
    
class SpeechCommandsV2(Dataset):
    label_dict = {
        'zero': 0., 'one': 1., 'two': 2., 'three': 3., 'four': 4., 'five': 5., 'six': 6., 'seven': 7., 
        'eight': 8., 'nine': 9., 'bed': 10., 'dog': 11., 'happy': 12., 'marvin': 13., 'off': 14., 
        'right': 15., 'up': 16., 'yes': 17., 'bird': 18., 'down': 19., 'house': 20., 'on': 21., 
        'stop': 22., 'tree': 23., 'cat': 24., 'go': 25., 'left': 26., 'no': 27., 'sheila': 28., 
        'wow': 29., 'Backward': 30., 'Forward': 31., 'Follow': 32., 'Learn': 33., 'Visual': 34.
    }
    def __init__(
            self, root_path:str, folder_in_archive:str, mode:str=None, download:bool = True, 
            data_tf:torch.nn.Module=None, label_tf:torch.nn.Module=None
        ):
        from torchaudio.datasets import SPEECHCOMMANDS
        super(SpeechCommandsV2, self).__init__()
        assert mode in ['training', 'validation', 'testing', None], 'No support'
        self.dataset = SPEECHCOMMANDS(
            root=root_path, url='speech_commands_v0.02', folder_in_archive=folder_in_archive, subset=mode, download=download
        )
        self.data_tf = data_tf
        self.label_tf = label_tf

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        wavform, sample_rate, label, speaker_id, utterance_num = self.dataset[index]
        label = int(self.label_dict[label])
        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)
        return wavform, label