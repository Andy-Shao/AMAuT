import argparse
import wandb
from ml_collections import ConfigDict
import os
import numpy as np
import random
from tqdm import tqdm

import torch
from torch import optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse, relative_path
from lib.wavUtils import Components, AudioPadding, time_shift, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer, VisionTokenTransformer
from lib.scDataset import SpeechCommandsDataset
from lib.datasets import TwoTFDataset
from AuT.lib.model import cal_model_tag, AudioClassifier
from AuT.speech_commands.pre_train import build_dataest, op_copy, lr_scheduler
from AuT.lib.config import transformer_cfg, classifier_cfg
from AuT.lib.loss import CrossEntropyLabelSmooth
from MsT.lib.multi_embed import FreqEmbedding
from MsT.lib.model import BiEmbedTransformer

def build_optim(args:argparse.Namespace, auT:BiEmbedTransformer, auC:AudioClassifier) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k,v in auT.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    for k,v in auC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def build_model(args:argparse.Namespace) -> tuple[BiEmbedTransformer, AudioClassifier]:
    cfg = ConfigDict()
    transformer_cfg(embed_size=args.embed_size, cfg=cfg)
    classifier_cfg(class_num=args.class_num, cfg=cfg)
    cfg.embedding = ConfigDict()
    cfg.embedding.embed_size = args.embed_size

    embed1 = FreqEmbedding(
        num_channels=args.n_mels, embed_size=args.embed_size, marsked_rate=.15, width=128, 
        num_layers=[6, 8], in_shape=[args.n_mels, args.target_length]
    )
    embed2 = FreqEmbedding(
        num_channels=81, embed_size=args.embed_size, marsked_rate=.15, width=128, 
        num_layers=[6, 8], in_shape=[81, 320]
    )
    auT = BiEmbedTransformer(cfg=cfg, embed1=embed1, embed2=embed2).to(device=args.device)
    auC = AudioClassifier(config=cfg).to(device=args.device)
    return auT, auC


if __name__ == '__main__':
    ap = argparse.ArgumentParser('MeT')
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='speech-commands')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--model_topology', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='MeT', choices=['MeT'])
    ap.add_argument('--embed_size', type=int, default=768, choices=[768, 1024])
    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'MeT', 'pre_train')
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
    #########################################

    wandb_run = wandb.init(
        project='AC-PT (AuT)', name=cal_model_tag(dataset_tag=args.dataset, pre_tag=args.arch), 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    max_ms=1000
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=163
    mel_scale='slaney'
    args.target_length=100
    tf_array = Components(transforms=[
        AudioPadding(max_ms=max_ms, sample_rate=sample_rate, random_shift=True),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 100
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
    ])
    train_dataset = TwoTFDataset(
        dataset=build_dataest(args=args, tsf=tf_array, mode='train'), 
        tf1=FrequenceTokenTransformer(),
        tf2=VisionTokenTransformer()
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    tf_array = Components(transforms=[
        AudioPadding(max_ms=max_ms, sample_rate=sample_rate, random_shift=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
    ])
    val_dataset = TwoTFDataset(
        dataset=build_dataest(args=args, tsf=tf_array, mode='test'),
        tf1=FrequenceTokenTransformer(),
        tf2=VisionTokenTransformer()
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    auTModel, clsModel = build_model(args=args)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)
    optimizer = build_optim(args=args, auT=auTModel, auC=clsModel)

    max_val_accu = 0.
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")

        print("Training...")
        ttl_train_size = 0.
        ttl_train_corr = 0.
        ttl_train_loss = 0.
        auTModel.train()
        clsModel.train()
        for fs1, fs2, labels in tqdm(train_loader):
            fs1, fs2, labels = fs1.to(args.device), fs2.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            outputs = clsModel(auTModel(fs1, fs2))
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
        auTModel.eval()
        clsModel.eval()
        for fs1, fs2, labels in tqdm(val_loader):
            fs1, fs2, labels = fs1.to(args.device), fs2.to(args.device), labels.to(args.device)

            with torch.no_grad():
                outputs = clsModel(auTModel(fs1, fs2))
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_val_size += labels.shape[0]
            ttl_val_corr += (preds == labels).sum().cpu().item()
        ttl_val_accu = ttl_val_corr/ttl_val_size * 100.
        print(f'Validation size:{ttl_val_size:.0f}, accuracy:{ttl_val_accu:.2f}%')
        if max_val_accu < ttl_val_accu:
            max_val_accu = ttl_val_accu
            torch.save(auTModel.state_dict(), relative_path(args, 'MeT.pt'))
            torch.save(auTModel.state_dict(), relative_path(args, 'MeT-Cls.pt'))

        wandb.log({
            'Train/Accu': ttl_train_corr/ttl_train_size * 100.,
            'Train/Loss': ttl_train_loss/ttl_train_size,
            'Train/LR': learning_rate,
            'Val/Accu': ttl_val_corr/ttl_val_size * 100.,
        }, step=epoch, commit=True)

        if args.early_stop >= 0:
            if args.early_stop == epoch+1: exit()
