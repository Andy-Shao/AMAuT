import argparse
import random
import numpy as np
import os
import wandb
from tqdm import tqdm

import torch
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, relative_path, store_model_structure_to_txt
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, time_shift, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.wavUtils import GuassianNoise, BackgroundNoise
from lib.datasets import dataset_tag, MultiTFDataset
from lib.acousticDataset import CochlScene
from lib.spDataset import BackgroundNoiseDataset
from AuT.lib.model import FCETransform, FCEClassifier
from AuT.speech_commands.train import lr_scheduler, build_optimizer
from AuT.lib.loss import CrossEntropyLabelSmooth
from AuT.lib.config import CT_base

def background_noise(args:argparse.Namespace) -> dict[str, torch.Tensor]:
    dataset = BackgroundNoiseDataset(root_path=args.background_path)
    ret = {}
    for noise_type, noise, sample_rate in dataset:
        ret[noise_type] = noise
    return ret

def build_model(args:argparse.Namespace) -> tuple[FCETransform, FCEClassifier]:
    if args.arch_level == 'base':
        config = CT_base(class_num=args.class_num, n_mels=args.n_mels)
        config.embedding.in_shape = [args.n_mels, args.target_length]
        config.embedding.num_layers = [6, 4, 12]
        config.embedding.width = 64
        config.embedding.embed_num = 92
        auTmodel = FCETransform(config=config).to(device=args.device)
        clsmodel = FCEClassifier(config=config).to(device=args.device)

    return auTmodel, clsmodel

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='CochlScene', choices=['CochlScene'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--background_path', type=str)
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
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'FCE'])
    ap.add_argument('--arch_level', type=str, default='base')

    args = ap.parse_args()
    if args.dataset == 'CochlScene':
        args.class_num = 13
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
    
    background_noises = background_noise(args=args)

    sample_rate=44100
    args.n_mels=64
    n_fft=2048
    win_length=800
    hop_length=300
    mel_scale='slaney'
    args.target_length=1472
    if args.dataset == 'CochlScene':
        train_dataset = CochlScene(root_path=args.dataset_root_path, mode='train', include_rate=False)
    train_dataset = MultiTFDataset(
        dataset=train_dataset,
        tfs=[
            Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=sample_rate*10),
                time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 1471
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                GuassianNoise(noise_level=.015),
                AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=sample_rate*10),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 1471
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                BackgroundNoise(noise_level=50, noise=background_noises['dude_miaowing'], is_random=True),
                AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=sample_rate*10),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 1471
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                BackgroundNoise(noise_level=50, noise=background_noises['pink_noise'], is_random=True),
                AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=sample_rate*10),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 1471
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        ]
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate*10),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 1471
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])
    if args.dataset == 'CochlScene':
        val_dataset = CochlScene(root_path=args.dataset_root_path, mode=args.validation_mode, data_tf=tf_array, include_rate=False)
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
        for fs in tqdm(train_loader):
            labels = fs[-1].to(args.device)
            optimizer.zero_grad()
            for i in range(len(fs) - 1):
                features = fs[i].to(args.device)
                outputs = clsmodel(auTmodel(features)[1])
                if i == 0: loss = loss_fn(outputs, labels)
                else: loss += loss_fn(outputs, labels)
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

        print("Validating...")
        ttl_val_size = 0.
        ttl_val_corr = 0.
        auTmodel.eval()
        clsmodel.eval()
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                outputs = clsmodel(auTmodel(features)[1])
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