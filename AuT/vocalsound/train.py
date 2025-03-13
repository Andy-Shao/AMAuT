import argparse
import random
import numpy as np
import os
import wandb
from tqdm import tqdm

import torch
from torch import optim
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, relative_path, store_model_structure_to_txt
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, time_shift, FrequenceTokenTransformer, AudioClip
from lib.datasets import dataset_tag
from lib.spDataset import VocalSound
from AuT.lib.model import AudioTransform, AudioClassifier
from AuT.speech_commands.train import build_optimizer
from AuT.lib.loss import CrossEntropyLabelSmooth
from AuT.lib.config import CT_base

def lr_scheduler(optimizer: torch.optim.Optimizer, epoch:int, lr_cardinality:int, gamma=10, power=0.75, lr_offset=1) -> optim.Optimizer:
    if epoch >= lr_cardinality-lr_offset:
        return optimizer
    decay = (1 + gamma * epoch / lr_cardinality) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    return optimizer

def build_model(args:argparse.Namespace) -> tuple[AudioTransform, AudioClassifier]:
    if args.arch_level == 'base':
        config = CT_base(class_num=args.class_num, n_mels=args.n_mels)
        config.embedding.in_shape = [args.n_mels, args.target_length]
        config.embedding.num_layers = [6, 12]
        auTmodel = AudioTransform(config=config).to(device=args.device)
        clsmodel = AudioClassifier(config=config).to(device=args.device)

    return auTmodel, clsmodel

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--file_name_suffix', type=str, default='')
    ap.add_argument('--validation_mode', type=str, default='validation', choices=['validation', 'test'])

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_offset', type=int, default=1)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--arch_level', type=str, default='base')

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'train')
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
        project='AuT-Train', name=f'{args.arch}-{dataset_tag(args.dataset)}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])

    sample_rate=16000
    max_length = sample_rate * 3
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=301
    mel_scale='slaney'
    args.target_length=160
    tf_array = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=max_length),
        AudioClip(max_length=max_length, mode='head', is_random=False),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 160
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ])
    train_dataset = VocalSound(root_path=args.dataset_root_path, mode='train', include_rate=False, data_tf=tf_array, version='16k')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=max_length),
        AudioClip(max_length=max_length, mode='head', is_random=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 160
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ])
    val_dataset = VocalSound(root_path=args.dataset_root_path, mode=args.validation_mode, include_rate=False, data_tf=tf_array, version='16k')
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    auTmodel, clsmodel = build_model(args)
    store_model_structure_to_txt(model=auTmodel, output_path=relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-auT{args.file_name_suffix}.txt'))
    store_model_structure_to_txt(model=clsmodel, output_path=relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-cls{args.file_name_suffix}.txt'))
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel, auD=None)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

    max_accu = 0.
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
            outputs, _ = clsmodel(auTmodel(features))
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            ttl_train_size += labels.shape[0]
            _, preds = torch.max(outputs.detach(), dim=1)
            ttl_train_corr += (preds == labels).sum().cpu().item()
            ttl_train_loss += loss.cpu().item()
        print(f'Training size:{ttl_train_size:.0f}, accuracy:{ttl_train_corr/ttl_train_size * 100.:.2f}%')

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, lr_offset=args.lr_offset)

        print("Validating...")
        ttl_val_size = 0.
        ttl_val_corr = 0.
        auTmodel.eval()
        clsmodel.eval()
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                attens = auTmodel(features)
                outputs, _ = clsmodel(attens)
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_val_size += labels.shape[0]
            ttl_val_corr += (preds == labels).sum().cpu().item()
        ttl_val_accu = ttl_val_corr/ttl_val_size * 100.
        print(f'Validation size:{ttl_val_size:.0f}, accuracy:{ttl_val_accu:.2f}%')
        if ttl_val_accu >= max_accu:
            max_accu = ttl_val_accu
            torch.save(auTmodel.state_dict(), relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-auT{args.file_name_suffix}.pt'))
            torch.save(clsmodel.state_dict(), relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-cls{args.file_name_suffix}.pt'))

        wandb.log({
            'Train/Accu': ttl_train_corr/ttl_train_size * 100.,
            'Train/Loss': ttl_train_loss/ttl_train_size,
            'Train/LR': learning_rate,
            'Val/Accu': ttl_val_corr/ttl_val_size * 100.,
        }, step=epoch, commit=True)

        if args.early_stop >= 0:
            if args.early_stop == epoch+1: exit()