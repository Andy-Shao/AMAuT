import os

import torch 
import torch.nn as nn
from torch.utils.data import Dataset

from lib.scDataset import SpeechCommandsDataset

class VisionTokenTransformer(nn.Module):
    def __init__(self, kernel_size=(16, 20), stride=(8, 10)):
        super(VisionTokenTransformer, self).__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        x = x.transpose(1, 0)
        return x

class FrequenceTokenTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        c, token_num, token_len = x.size()
        x = x.reshape(-1, token_len)
        return x

class AudioTokenTransformer(nn.Module):
    def __init__(self) -> None:
        super(AudioTokenTransformer, self).__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)
    
class FewShotSpeechCommand(Dataset):
    def __init__(
            self, root_path: str, mode: str, include_rate=True, data_tfs=None, data_type='all',
            output_path: str = './result/speech-commands', seed = 2025, fs_num = 10
        ):
        super(FewShotSpeechCommand, self).__init__()
        assert mode in ['train', 'validation', 'test', 'test-fs', 'test-res'], 'mode type is incorrect'
        self.is_fs_mode = True if mode == 'test-fs' or mode == 'test-res' else False
        self.root_dataset = SpeechCommandsDataset(
            root_path=root_path, mode='test' if self.is_fs_mode else mode, 
            include_rate=include_rate, data_tfs=data_tfs, data_type=data_type,
            normalized=False
        )
        self.root_path = root_path
        self.mode = mode
        self.include_rate = include_rate
        self.data_tfs = data_tfs
        self.data_type = data_type
        self.output_path = output_path
        self.seed = seed 
        self.fs_num = fs_num

        if self.is_fs_mode:
            self.__split_test__()
    
    def __split_test__(self, fs_file_name='test-fewshot.txt', res_file_name='test-residual.txt'):
        full_fs_name = os.path.join(self.output_path, fs_file_name)
        full_res_name = os.path.join(self.output_path, res_file_name)
        if os.path.exists(full_fs_name):
            if self.mode == 'test-fs':
                with open(full_fs_name, mode='tr', newline='\n') as f:
                    lines = f.readlines()
            elif self.mode == 'test-res':
                with open(full_res_name, mode='tr', newline='\n') as f:
                    lines = f.readlines()
            else:
                raise Exception('No support')
            self.data_indexes = [line.rstrip('\n') for line in lines]
        else:
            from numpy.random import MT19937, RandomState, SeedSequence
            rs = RandomState(MT19937(SeedSequence(self.seed)))
            data_lists = {}
            if self.include_rate:
                for index, (feature, label, rate) in enumerate(self.root_dataset):
                    if label in data_lists:
                        data_lists[label].append(index)
                    else:
                        data_lists[label] = [index]
            else:
                for index, (feature, label) in enumerate(self.root_dataset):
                    if label in data_lists:
                        data_lists[label].append(index)
                    else:
                        data_lists[label] = [index]
            fs_indexes = []
            res_indexes = []
            for key, value in data_lists.items():
                fs_indexes += [it for it in rs.choice(value, size=self.fs_num, replace=False)]
                res_indexes += [it for it in value if it not in fs_indexes]

            with open(full_fs_name, mode='wt', newline='\n') as f:
                for it in fs_indexes:
                    f.write(str(it)+'\n')
            with open(full_res_name, mode='wt', newline='\n') as f:
                for it in res_indexes:
                    f.write(str(it)+'\n')

            if self.mode == 'test-fs':
                self.data_indexes = fs_indexes
            elif self.mode == 'test-res':
                self.data_indexes = res_indexes
            else:
                raise Exception('No support')

    def __len__(self):
        if self.is_fs_mode:
            return len(self.data_indexes)
        else:
            return len(self.root_dataset)
    
    def __getitem__(self, index):
        if self.is_fs_mode:
            return self.root_dataset[int(self.data_indexes[index])]
        else:
            return self.root_dataset[index]