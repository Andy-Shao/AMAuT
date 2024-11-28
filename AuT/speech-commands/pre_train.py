import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import Dataset, DataLoader
from ml_collections import ConfigDict
import torch.nn as nn
import torch.optim as optim

from lib.toolkit import print_argparse, store_model_structure_to_txt, relative_path, count_ttl_params
from lib.wavUtils import pad_trunc, Components, AmplitudeToDB, DoNothing, time_shift
from lib.scDataset import SpeechCommandsDataset
from AuT.lib.model import AudioTransform, AudioClassifier, cal_model_tag
from AuT.lib.loss import CrossEntropyLabelSmooth
from AuT.lib.dataset import AudioTokenTransformer

def lr_scheduler(optimizer: torch.optim.Optimizer, epoch:int, max_epoch:int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * epoch / max_epoch) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    return optimizer

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def build_optimizer(args: argparse.Namespace, auT:nn.Module, auC:nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in auT.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * .1}]
    for k, v in auC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def build_model(args:argparse.Namespace) -> tuple[nn.Module, nn.Module]:
    def transformer_cfg(args:argparse.Namespace, cfg:ConfigDict) -> None:
        cfg.transform = ConfigDict()
        cfg.transform.layer_num = 24
        cfg.transform.head_num = 16
        cfg.transform.atten_drop_rate = .0
        cfg.transform.mlp_mid = 1024
        cfg.transform.mlp_out = 1024
        cfg.transform.mlp_dp_rt = .1
    
    def classifier_cfg(args:argparse.Namespace, cfg:ConfigDict) -> None:
        cfg.classifier = ConfigDict()
        cfg.classifier.class_num = args.class_num
        cfg.classifier.extend_size = 2048
        cfg.classifier.convergent_size = 256

    config = ConfigDict()
    config.embedding = ConfigDict()
    if args.embed_mode == 'linear':
        config.embedding.in_token_len = 81
        config.embedding.in_token_num = 80
        config.embedding.channel_num = 1
    elif args.embed_mode == 'restnet':
        config.embedding.channel_num = 80
    config.embedding.marsked_rate = .15
    config.embedding.embed_size = 1024
    config.embedding.mode = args.embed_mode # restnet or linear

    transformer_cfg(args, config)
    classifier_cfg(args, config)

    auTmodel = AudioTransform(config=config).to(device=args.device)
    clsmodel = AudioClassifier(config=config).to(device=args.device)

    return auTmodel, clsmodel

def build_dataest(args:argparse.Namespace, tsf:nn.Module, mode:str) -> Dataset:
    if args.dataset == 'speech-commands-random':
        pass
    else:
        dataset = SpeechCommandsDataset(
            root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=tsf,
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

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval_num', type=int, default=50, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--embed_mode', type=str, default='linear', choices=['restnet', 'linear'])

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
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'pre_train')
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
        project='AC-PT (AuT)', name=cal_model_tag(dataset_tag=args.dataset, embed_mode=args.embed_mode), mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    max_ms=1000
    sample_rate=16000
    n_mels=80
    n_fft=1024
    hop_length=200
    tf_array = Components(transforms=[
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length), # 80 x 81
        AmplitudeToDB(top_db=80., max_out=2.),
        AudioTokenTransformer() if args.embed_mode == 'linear' else DoNothing()
    ])

    train_dataset = build_dataest(args=args, tsf=tf_array, mode='train')
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    tf_array = Components(transforms=[
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        a_transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length),
        AmplitudeToDB(top_db=80., max_out=2.),
        AudioTokenTransformer() if args.embed_mode == 'linear' else DoNothing()
    ])
    val_dataset = build_dataest(args=args, tsf=tf_array, mode='test')
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    interval = args.max_epoch // args.interval_num

    auTmodel, clsmodel = build_model(args=args)
    store_model_structure_to_txt(model=auTmodel, output_path=relative_path(args, 'auTmodel.txt'))
    store_model_structure_to_txt(model=clsmodel, output_path=relative_path(args, 'clsmodel.txt'))
    print(
        f'auT weight number:{count_ttl_params(auTmodel)}',
        f', auC weight number:{count_ttl_params(clsmodel)}',
        f', total weight number:{count_ttl_params(auTmodel) + count_ttl_params(clsmodel)}')
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel)

    max_val_accu = 0.
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")

        print("Training...")
        ttl_train_size = 0.
        ttl_train_corr = 0.
        ttl_train_loss = 0.
        auTmodel.train()
        clsmodel.train()
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            optimizer.zero_grad()
            outputs = clsmodel(auTmodel(features))
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            ttl_train_size += labels.shape[0]
            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_train_corr += (preds == labels).sum().cpu().item()
            ttl_train_loss += loss.cpu().item()
        print(f'Training size:{ttl_train_size:.0f}, accuracy:{ttl_train_corr/ttl_train_size * 100.:.2f}%') 

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % interval == 0 or epoch == args.max_epoch - 1:
            lr_scheduler(optimizer=optimizer, epoch=epoch, max_epoch=args.max_epoch)
        
        print("Validation...")
        ttl_val_size = 0.
        ttl_val_corr = 0.
        auTmodel.eval()
        clsmodel.eval()
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            with torch.no_grad():
                outputs = clsmodel(auTmodel(features))
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_val_size += labels.shape[0]
            ttl_val_corr += (preds == labels).sum().cpu().item()
        ttl_val_accu = ttl_val_corr/ttl_val_size * 100.
        print(f'Validation size:{ttl_val_size:.0f}, accuracy:{ttl_val_accu:.2f}%')
        if max_val_accu < ttl_val_accu:
            max_val_accu = ttl_val_accu
            torch.save(auTmodel.state_dict(), relative_path(args, 'AuT.pt'))
            torch.save(clsmodel.state_dict(), relative_path(args, 'AuT-Cls.pt'))

        wandb.log({
            'Train/Accu': ttl_train_corr/ttl_train_size * 100.,
            'Train/Loss': ttl_train_loss/ttl_train_size,
            'Train/LR': learning_rate,
            'Val/Accu': ttl_val_corr/ttl_val_size * 100.,
        }, step=epoch, commit=True)

        if args.early_stop >= 0:
            if args.early_stop == epoch+1: exit()