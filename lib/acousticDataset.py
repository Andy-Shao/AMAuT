import os
from dataclasses import dataclass

import torchaudio
from torch.utils.data import Dataset
from torch import nn

class CochlScene(Dataset):
    """
    CochlScene is an acoustic scene classification dataset created using crowdsourcing. 
    It comprises approximately 76,000 audio samples across 13 distinct acoustic scenes. 
    The dataset was collected from 831 participants and features audio recordings of varied lengths, 
    with the majority ranging between 60 and 70 seconds.

    Sample rate: 44100, 10 seconds audio, mono audio

    The 13 acoustic scenes included in CochlScene are:
    Bus, Car, Elevator, Park, Restaurant, Street, SubwayStation, Cafe, CrowdedIndoor, Kitchen,
    ResidentialArea, Restroom, Subway
    """
    @dataclass
    class SampleMeta:
        sample_path: str
        label_name: str
        label_id: int
    label_dic = {
        'Bus': 0, 'Car': 1, 'Elevator': 2, 'Park': 3, 'Restaurant': 4, 'Street': 5, 'SubwayStation': 6,
        'Cafe': 7, 'CrowdedIndoor': 8, 'Kitchen': 9, 'ResidentialArea': 10, 'Restroom': 11, 'Subway': 12
    }
    mode_mapping = {
        'train': 'Train', 'validation': 'Val', 'test': 'Test'
    }
    def __init__(self, root_path:str, mode:str, data_tf:nn.Module=None, label_tf:nn.Module=None, include_rate:bool=False):
        super(CochlScene).__init__()
        assert mode in ['train', 'validation', 'test'], 'No support'
        self.root_path = root_path
        self.mode = mode
        self.data_tf = data_tf
        self.label_tf = label_tf
        self.include_rate = include_rate

        self.data_list = self.__cal_data_list__()

    def __cal_data_list__(self) -> list[SampleMeta]:
        ret = []
        base_path = os.path.join(self.root_path, self.mode_mapping[self.mode])

        for k,v in self.label_dic.items():
            parent_path = os.path.join(base_path, k)
            for file_path in os.listdir(parent_path):
                if not file_path.endswith('.wav'):
                    continue
                meta_info = self.SampleMeta(
                    sample_path=os.path.join(parent_path, file_path),
                    label_name=k,
                    label_id=v
                )
                ret.append(meta_info)
        return ret

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        meta_info = self.data_list[index]
        wavform, sample_rate = torchaudio.load(meta_info.sample_path)
        label = meta_info.label_id

        if self.data_tf is not None:
            wavform = self.data_tf(wavform)
        if self.label_tf is not None:
            label = self.label_tf(label)

        if self.include_rate:
            return wavform, label, sample_rate
        return wavform, label