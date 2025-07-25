import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from typing import Any
import shutil

from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from torch import nn

def dataset_tag(dataset_name:str) -> str:
    if dataset_name == 'speech-commands':
        return 'SC'
    elif dataset_name == 'speech-commands_v2':
        return 'SC2'
    elif dataset_name == 'AudioMNIST':
        return 'AM'
    elif dataset_name == 'AudioMNIST2':
        return 'AM2'
    elif dataset_name == 'VocalSound':
        return 'VS'
    elif dataset_name == 'CochlScene':
        return 'CS'
    else:
        raise Exception('No support')

def multi_process_store_to(loader: DataLoader, root_path: str, index_file_name: str, data_transf=None, label_transf=None) -> None:
    print(f'Store dataset into {root_path}, meta file is: {index_file_name}')
    data_index = pd.DataFrame(columns=['data_path', 'label'])
    try: 
        if os.path.exists(root_path): shutil.rmtree(root_path)
        os.makedirs(root_path)
    except:
        print('remove directory has an error.')
    for i, (features, labels) in tqdm(enumerate(loader), total=len(loader)):
        for k in range(features.shape[0]):
            feature, label = features[k].clone(), labels[k].clone()
            if data_transf is not None:
                feature = data_transf(feature)
            if label_transf is not None:
                label = label_transf(label)
            data_path = f'{i}_{k}_{label}.dt'
            data_index.loc[len(data_index)] = [data_path, label.item()]
            torch.save(feature, os.path.join(root_path, data_path))
    data_index.to_csv(os.path.join(root_path, index_file_name))

def store_to(dataset: Dataset, root_path: str, index_file_name: str, data_transf=None, label_transf=None) -> None:
    print(f'Store dataset into {root_path}, meta file is: {index_file_name}')
    data_index = pd.DataFrame(columns=['data_path', 'label'])
    try: 
        if os.path.exists(root_path): shutil.rmtree(root_path)
        os.makedirs(root_path)
    except:
        print('remove directory has an error.')
    for index, (feature, label) in tqdm(enumerate(dataset), total=len(dataset)):
        if data_transf is not None:
            feature = data_transf(feature)
        if label_transf is not None:
            label = data_transf(feature)
        data_path = f'{index}_{label}.dt'
        data_index.loc[len(data_index)] = [data_path, label]
        torch.save(feature, os.path.join(root_path, data_path))
    data_index.to_csv(os.path.join(root_path, index_file_name))

def load_from(root_path: str, index_file_name: str, data_tf=None, label_tf=None) -> Dataset:
    class LoadDs(Dataset):
        def __init__(self) -> None:
            super().__init__()
            data_index = pd.read_csv(os.path.join(root_path, index_file_name))
            self.data_meta = []
            for idx in range(len(data_index)):
                self.data_meta.append([data_index['data_path'][idx], data_index['label'][idx]]) 
        
        def __len__(self):
            return len(self.data_meta)
        
        def __getitem__(self, index) -> Any:
            data_path = self.data_meta[index][0]
            feature = torch.load(os.path.join(root_path, data_path))
            label = self.data_meta[index][1]
            if data_tf is not None:
                feature = data_tf(feature)
            if label_tf is not None:
                label = label_tf(label)
            return feature, int(label)
    return LoadDs()

class ClipDataset(Dataset):
    def __init__(self, dataset: Dataset, rate: float) -> None:
        super().__init__()      
        assert rate > 0. and rate <= 1., 'rate is the out range'
        self.dataset = dataset
        self.data_size = int(len(dataset) * rate)
        self.indexes = np.random.randint(len(dataset), size=self.data_size)

    def __len__(self):
        return self.data_size
    
    def __getitem__(self, index) -> Any:
        return self.dataset[self.indexes[index]]
    
class TransferDataset(Dataset):
    def __init__(self, dataset: Dataset, data_tf:nn.Module=None, label_tf:nn.Module=None, device='cpu', keep_cpu=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_tf = data_tf if device == 'cpu' or data_tf is None else data_tf.to(device=device)
        self.label_tf = label_tf if device == 'cpu' or label_tf is None else label_tf.to(device=device)
        self.device = device
        self.keep_cpu = keep_cpu

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        feature, label = self.dataset[index]
        if self.device != 'cpu':
            feature = feature.to(self.device)
            label = label.to(self.device) if isinstance(label, torch.Tensor) else label
        if self.data_tf is not None:
            feature = self.data_tf(feature)
        if self.label_tf is not None:
            label = self.label_tf(label)

        if self.device != 'cpu' and self.keep_cpu:
            return feature.cpu(), label.cpu() if isinstance(label, torch.Tensor) else label
        else:
            return feature, label

class TwoTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tf1:nn.Module=None, tf2:nn.Module=None):
        super(TwoTFDataset, self).__init__()
        self.dataset = dataset
        self.tf1, self.tf2 = tf1, tf2

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        x1, x2 = item.clone(), item.clone()
        if self.tf1 is not None:
            x1 = self.tf1(x1)
        if self.tf2 is not None:
            x2 = self.tf2(x2)
        return x1, x2, label

class MultiTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tfs:list[nn.Module]):
        super(MultiTFDataset, self).__init__()
        assert tfs is not None, 'No support'
        self.dataset = dataset
        self.tfs = tfs

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        ret = [item.clone() for _ in range(len(self.tfs))]
        for i, tf in enumerate(self.tfs):
            if tf is not None:
                ret[i] = tf(ret[i])
        ret.append(label)
        return tuple(ret)
    
class FewShotDataset(Dataset):
    def __init__(
            self, dataset:Dataset, output_path: str, mode:str, seed = 2025, fs_num = 10,
            fs_file_name='test-fewshot.txt', res_file_name='test-residual.txt'
        ):
        super(FewShotDataset, self).__init__()
        assert mode in ['fewshot', 'residual'], 'No support'
        self.mode = mode
        self.dataset = dataset
        self.output_path = output_path
        self.fs_file_name = fs_file_name
        self.res_file_name = res_file_name
        self.seed = seed
        self.fs_num = fs_num
        self.__split_test__()

    def __split_test__(self):
        full_fs_name = os.path.join(self.output_path, self.fs_file_name)
        full_res_name = os.path.join(self.output_path, self.res_file_name)
        if os.path.exists(full_fs_name):
            if self.mode == 'fewshot':
                with open(full_fs_name, mode='tr', newline='\n') as f:
                    lines = f.readlines()
            elif self.mode == 'residual':
                with open(full_res_name, mode='tr', newline='\n') as f:
                    lines = f.readlines()
            else:
                raise Exception('No support')
            self.data_index = [line.rstrip('\n') for line in lines]
        else:
            from numpy.random import MT19937, RandomState, SeedSequence
            rs = RandomState(MT19937(SeedSequence(self.seed)))
            data_lists = {}
            for index, (feature, label) in enumerate(self.dataset):
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
            if self.mode == 'fewshot':
                self.data_index = fs_indexes
            elif self.mode == 'residual':
                self.data_index = res_indexes
            else:
                raise Exception('No support')
            
    def __len__(self):
        return len(self.data_index)
    
    def __getitem__(self, index):
        index = self.data_index[index]
        return self.dataset[int(index)]

class Dataset_Idx(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int, int]:
        feature, label = self.dataset[index]
        return feature, label, index
    
class MergeDataset(Dataset):
    def __init__(self, set1:Dataset, set2:Dataset):
        super(MergeDataset, self).__init__()
        self.set1 = set1
        self.set2 = set2

    def __len__(self):
        return len(self.set1) + len(self.set2)

    def __getitem__(self, index):
        if index < len(self.set1):
            return self.set1[index]
        else:
            index -= len(self.set1)
            return self.set2[index]

class PLDataset(Dataset):
    def __init__(self, dataset:Dataset, pseudo_labels:torch.Tensor=None):
        super(PLDataset, self).__init__()
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels

    def set_pl(self, pseudo_labels:torch.Tensor) -> None:
        assert pseudo_labels is not None, 'No support'
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        feature, label = self.dataset[index]
        if self.pseudo_labels is None:
            return feature, label
        else:
            return feature, int(self.pseudo_labels[index].cpu().item())