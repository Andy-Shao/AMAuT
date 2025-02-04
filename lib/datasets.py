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

class FilterAudioMNIST(Dataset):
    def __init__(self, root_path: str, filter_fn, data_tsf=None, include_rate=True):
        super(FilterAudioMNIST, self).__init__()
        data_pathes = load_datapath(root_path=root_path, filter_fn=filter_fn)
        self.dataset = AudioMINST(data_paths=data_pathes, data_trainsforms=data_tsf, include_rate=include_rate)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        return self.dataset[index]

class AudioMINST(Dataset):
    def __init__(self, data_paths: list[str], data_trainsforms=None, include_rate=True):
        super(AudioMINST, self).__init__()
        self.data_paths = data_paths
        self.data_trainsforms = data_trainsforms
        self.include_rate = include_rate

    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index) -> tuple[object, float]:
        (wavform, sample_rate) = torchaudio.load(self.data_paths[index])
        label = self.data_paths[index].split('/')[-1].split('_')[0]
        if self.data_trainsforms is not None:
            wavform = self.data_trainsforms(wavform)
        if self.include_rate:
            return (wavform, sample_rate), int(label)
        else:
            return wavform, int(label)
    
def load_datapath(root_path: str, filter_fn) -> list[str]:
    dataset_list = []
    meta_file_path = root_path + '/audioMNIST_meta.txt'
    with open(meta_file_path) as f:
        meta_data = json.load(f)
    for k,v in meta_data.items():
        if filter_fn(v):
            data_path = f'{root_path}/{k}'
            for it in os.listdir(data_path):
                if it.endswith('wav'):
                    dataset_list.append(f'{data_path}/{it}')
    return dataset_list

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
    def __init__(self, dataset: Dataset, data_tf=None, label_tf=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.data_tf = data_tf
        self.label_tf = label_tf

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        feature, label = self.dataset[index]
        if self.data_tf is not None:
            feature = self.data_tf(feature)
        if self.label_tf is not None:
            label = self.label_tf(label)
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