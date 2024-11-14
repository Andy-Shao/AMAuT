import argparse
import os
import numpy as np
import random
import wandb

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import Dataset, DataLoader

from lib.toolkit import print_argparse
from lib.wavUtils import pad_trunc, Components
from lib.scDataset import SpeechCommandsDataset
from AuT2.lib.embedding import Embedding

def build_dataest(args:argparse.Namespace, tsf:list, mode:str) -> Dataset:
    if args.dataset == 'speech-commands-random':
        pass
    else:
        dataset = SpeechCommandsDataset(
            root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=Components(transforms=tsf),
            data_type=args.dataset_type
        )
    return dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands-purity', 'speech-commands-random', 'speech-commands-numbers'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='speech-commands')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')
    # ap.add_argument('--test_rate', type=float, default=.3)

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=50, help='interval')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')

    args = ap.parse_args()
    if args.dataset == 'speech-commands' or args.dataset == 'speech-commands-random':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands-purity':
        args.class_num = 10
        args.dataset_type = 'commands'
    elif args.dataset == 'speech-commands-numbers':
        args.class_num = 10
        args.dataset_type = 'numbers'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT2', 'pre_train')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    wandb_run = wandb.init(
        project='AC Pre-Training (CoNMix)', name=args.dataset, mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    max_ms=1000
    sample_rate=16000
    n_mels=128
    n_fft=1024
    hop_length=256
    tf_array = [
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length),
        a_transforms.AmplitudeToDB(top_db=80)
    ]

    train_dataset = build_dataest(args=args, tsf=tf_array, mode='train')
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    embedding = Embedding(num_channels=1, token_len=128).to(device=args.device)

    for features, labels in train_loader:
        features = torch.permute(features, dims=(0, 1, 3, 2))
        batch_size, channels, token_num, token_len = features.size()
        features, labels = features.to(args.device), labels.to(args.device)
        outputs = embedding(features)
        print(f'outputs shape is:{outputs.shape}')        
        break