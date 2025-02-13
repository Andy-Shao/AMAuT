import argparse
import random
import numpy as np
import os
import wandb
from ml_collections import ConfigDict
from tqdm import tqdm

import torch
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

from lib.toolkit import print_argparse, relative_path, store_model_structure_to_txt
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, time_shift, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.datasets import FilterAudioMNIST, ClipDataset
from AuT.lib.model import AudioTransform, AudioClassifier, AudioDecoder
from AuT.lib.config import transformer_cfg, classifier_cfg
from AuT.speech_commands.pre_train import includeAutoencoder, decoder_cfg, op_copy, lr_scheduler
from AuT.lib.loss import CrossEntropyLabelSmooth

def build_optimizer(args: argparse.Namespace, auT:nn.Module, auC:nn.Module, auD:AudioDecoder) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in auT.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    for k, v in auC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    if includeAutoencoder(args):
        for k, v in auD.named_parameters():
            param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def build_model(args:argparse.Namespace) -> tuple[AudioTransform, AudioClassifier, AudioDecoder]:
    config = ConfigDict()
    config.embedding = ConfigDict()
    config.embedding.channel_num = args.n_mels
    config.embedding.marsked_rate = .15
    config.embedding.embed_size = args.embed_size
    config.embedding.in_shape = [args.n_mels, args.target_length]
    config.embedding.arch = args.arch

    transformer_cfg(cfg=config, embed_size=args.embed_size)
    classifier_cfg(cfg=config, class_num=args.class_num)
    auTmodel = AudioTransform(config=config).to(device=args.device)
    clsmodel = AudioClassifier(config=config).to(device=args.device)

    if includeAutoencoder(args):
        decoder_cfg(cfg=config, embed_size=args.embed_size, n_mels=args.n_mels)
        auDecoder = AudioDecoder(config=config).to(device=args.device)
    else:
        auDecoder = None
    
    return auTmodel, clsmodel, auDecoder

def cal_model_tag(dataset_tag:str, pre_tag:str) -> str:
    tag = pre_tag
    if dataset_tag == 'AudioMNIST':
        tag += '-AM'
    else:
        raise Exception('No Support')
    return tag

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='AudioMNIST', choices=['AudioMNIST'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='AudioMNIST')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--model_topology', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_dec', type=float, default=1.25)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--embed_size', type=int, default=768, choices=[768, 1024])

    args = ap.parse_args()
    if args.dataset == 'AudioMNIST':
        args.class_num = 10
    else:
        raise Exception('No support!')
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
        project='AC-PT (AuT)', name=cal_model_tag(dataset_tag=args.dataset, pre_tag=args.arch), 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])

    max_ms=1000
    sample_rate=48000
    args.n_mels=80
    n_fft=2048
    win_length=800
    hop_length=302
    mel_scale='slaney'
    args.target_length=160
    tf_array = Components(transforms=[
        AudioPadding(max_ms=max_ms, sample_rate=sample_rate, random_shift=True),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 159
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])
    train_dataset = FilterAudioMNIST(root_path=args.dataset_root_path, data_tsf=tf_array, include_rate=False, filter_fn=lambda x: x['accent'] == 'German')
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = Components(transforms=[
        AudioPadding(max_ms=max_ms, sample_rate=sample_rate, random_shift=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 159
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])
    # val_dataset = FilterAudioMNIST(root_path=args.dataset_root_path, data_tsf=tf_array, include_rate=False, filter_fn=lambda x: x['accent'] != 'German')
    val_dataset = ClipDataset(dataset=train_dataset, rate=.3)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    auTmodel, clsmodel, _ = build_model(args)
    store_model_structure_to_txt(model=auTmodel, output_path=relative_path(args, 'auTmodel.txt'))
    store_model_structure_to_txt(model=clsmodel, output_path=relative_path(args, 'clsmodel.txt'))
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel, auD=None)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

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
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        print("Validation...")
        ttl_val_size = 0.
        ttl_val_corr = 0.
        auTmodel.eval()
        clsmodel.eval()
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                attens = auTmodel(features)
                outputs = clsmodel(attens)
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_val_size += labels.shape[0]
            ttl_val_corr += (preds == labels).sum().cpu().item()
        ttl_val_accu = ttl_val_corr/ttl_val_size * 100.
        print(f'Validation size:{ttl_val_size:.0f}, accuracy:{ttl_val_accu:.2f}%')
        if max_val_accu <= ttl_val_accu:
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