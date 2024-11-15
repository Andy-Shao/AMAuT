import argparse
import os
import numpy as np
import random
import wandb

import torch 
from torchaudio import transforms as a_transforms
from torchvision import transforms as v_transforms
from torch.utils.data import Dataset, DataLoader
from ml_collections import ConfigDict
import torch.nn as nn

from lib.toolkit import print_argparse
from lib.wavUtils import pad_trunc, Components, AmplitudeToDB
from lib.scDataset import SpeechCommandsDataset
from AuT.lib.model import AudioTransform, AudioClassifier

def build_model(args:argparse.Namespace) -> tuple[nn.Module, nn.Module]:
    config = ConfigDict()
    config.embedding = ConfigDict()
    config.embedding.in_token_len = 128
    config.embedding.channel_num = 1
    config.transform = ConfigDict()
    config.transform.embed_size = 1024
    config.transform.layer_num = 24
    config.transform.head_num = 16
    config.transform.atten_drop_rate = .1
    config.transform.mlp_mid = 1024
    config.transform.mlp_out = 1024
    config.transform.mlp_dp_rt = .1
    config.classifier = ConfigDict()
    config.classifier.class_num = args.class_num
    config.classifier.extend_size = 2048
    config.classifier.convergent_size = 256

    auTmodel = AudioTransform(config=config).to(device=args.device)
    clsmodel = AudioClassifier(config=config).to(device=args.device)

    return auTmodel, clsmodel

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
    # ap.add_argument('--normalized', action='store_true')
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
        AmplitudeToDB(top_db=80., max_out=2.)
    ]

    train_dataset = build_dataest(args=args, tsf=tf_array, mode='train')
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    auTmodel, clsmodel = build_model(args=args)

    for features, labels in train_loader:
        features = torch.permute(features, dims=(0, 1, 3, 2))
        batch_size, channels, token_num, token_len = features.size()
        features, labels = features.to(args.device), labels.to(args.device)
        outputs = clsmodel(auTmodel(features))
        print(f'outputs shape is:{outputs.shape}')        
        break