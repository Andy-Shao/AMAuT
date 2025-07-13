import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from lib.toolkit import print_argparse, store_model_structure_to_txt, relative_path, count_ttl_params
from lib.wavUtils import AudioPadding, Components, AmplitudeToDB, time_shift, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.spDataset import SpeechCommandsDataset, SpeechCommandsV2
from lib.datasets import dataset_tag
from AuT.lib.model import AudioTransform, AudioClassifier, AudioDecoder
from AuT.lib.loss import CrossEntropyLabelSmooth, CosineSimilarityLoss
from AuT.lib.config import decoder_cfg, CT_base

def print_weight_num(auT:AudioTransform, auC:AudioClassifier, auD:AudioDecoder, args:argparse.Namespace) -> None:
    if includeAutoencoder(args):
        print(
            f'auT weight number:{count_ttl_params(auT)}',
            f', auC weight number:{count_ttl_params(auC)}',
            f', auD weight number:{count_ttl_params(auD)}',
            f', total weight number:{count_ttl_params(auT) + count_ttl_params(auC) + count_ttl_params(auD)}')
    else:
        print(
            f'auT weight number:{count_ttl_params(auT)}',
            f', auC weight number:{count_ttl_params(auC)}',
            f', total weight number:{count_ttl_params(auT) + count_ttl_params(auC)}')

def includeAutoencoder(args:argparse.Namespace) -> bool:
    return args.arch == 'CTA'

def time_masking(features:torch.Tensor, cfgs:list[dict]) -> torch.Tensor:
    import torchaudio.functional as F
    ret = features
    for cfg in cfgs:
        ret = F.mask_along_axis_iid(
            specgrams=ret, mask_param=cfg['mask_param'], mask_value=.0, axis=2, p=cfg['p']
        )
    return ret

def store_model_structure_by_tb(tModel: nn.Module, cModel:nn.Module, input_tensor:torch.Tensor, log_dir:str) -> None:
    from torch.utils.tensorboard.writer import SummaryWriter
    import shutil

    try:
        if os.path.exists(log_dir): shutil.rmtree(log_dir)
    except:
        pass

    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model=tModel, input_to_model=input_tensor)
    tmp = tModel(input_tensor)
    writer.add_graph(model=cModel, input_to_model=tmp)
    writer.close()

def lr_scheduler(optimizer: torch.optim.Optimizer, epoch:int, lr_cardinality:int, gamma=10, power=0.75) -> optim.Optimizer:
    if epoch >= lr_cardinality-1:
        return optimizer
    decay = (1 + gamma * epoch / lr_cardinality) ** (-power)
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
    if args.arch_level == 'base':
        config = CT_base(class_num=args.class_num, n_mels=args.n_mels)
        config.embedding.in_shape = [args.n_mels, args.target_length]
        config.embedding.arch = args.arch
        if args.dataset == 'speech-commands_v2':
            config.embedding.num_layers = [6, 12]
        auTmodel = AudioTransform(config=config).to(device=args.device)
        clsmodel = AudioClassifier(config=config).to(device=args.device)

        if includeAutoencoder(args):
            decoder_cfg(cfg=config, embed_size=768, n_mels=args.n_mels)
            auDecoder = AudioDecoder(config=config).to(device=args.device)
        else:
            auDecoder = None

    return auTmodel, clsmodel, auDecoder

def build_dataset(args:argparse.Namespace, tsf:nn.Module, mode:str) -> Dataset:
    if args.dataset == 'speech-commands':
        dataset = SpeechCommandsDataset(
            root_path=args.dataset_root_path, mode=mode, include_rate=False, data_tfs=tsf,
            data_type=args.dataset_type
        )
    elif args.dataset == 'speech-commands_v2':
        mode_dict = {
            'train':'training',
            'validation':'validation',
            'test':'testing'
        }
        mode = mode_dict[mode]
        dataset = SpeechCommandsV2(
            root_path=args.dataset_root_path, mode=mode, data_tf=tsf, folder_in_archive='speech-commands_v2'
        )
    return dataset

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands_v2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--file_name_suffix', type=str, default='')

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
    ap.add_argument('--arch_level', type=str, default='base')

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands_v2':
        args.class_num = 35
        data_path = os.path.join(args.dataset_root_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
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
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
    tf_array = Components(transforms=[
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=True),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 104
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])

    train_dataset = build_dataset(args=args, tsf=tf_array, mode='train')
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    tf_array = Components(transforms=[
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])
    val_dataset = build_dataset(args=args, tsf=tf_array, mode='validation')
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    auTmodel, clsmodel, auDecoder = build_model(args=args)
    store_model_structure_to_txt(model=auTmodel, output_path=relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-auT{args.file_name_suffix}.txt'))
    store_model_structure_to_txt(model=clsmodel, output_path=relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-cls{args.file_name_suffix}.txt'))
    if includeAutoencoder(args):
        store_model_structure_to_txt(model=auDecoder, output_path=relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-auD{args.file_name_suffix}.txt'))
    print_weight_num(auT=auTmodel, auC=clsmodel, auD=auDecoder, args=args)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)
    decoder_loss_fn = nn.MSELoss(reduction='mean').to(device=args.device)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel, auD=auDecoder)

    if args.model_topology:
        features, labels = next(iter(train_loader))
        features = features.to(args.device)
        store_model_structure_by_tb(
            tModel=auTmodel, cModel=clsmodel, input_tensor=features, log_dir=relative_path(args, 'model_topology'))
        exit()

    max_val_accu = 0.
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")

        print("Training...")
        ttl_train_size = 0.
        ttl_train_corr = 0.
        ttl_train_loss = 0.
        auTmodel.train()
        clsmodel.train()
        if includeAutoencoder(args): auDecoder.train()
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            org_fts = torch.clone(features).detach().to(args.device)

            optimizer.zero_grad()
            if includeAutoencoder(args):
                attens, hidden_attens = auTmodel(features)
                outputs, _ = clsmodel(attens)
                gen_fts = auDecoder(attens, hidden_attens)
                loss = loss_fn(outputs, labels) + args.lr_dec * decoder_loss_fn(gen_fts, org_fts)
            else:
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
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)
        
        print("Validation...")
        ttl_val_size = 0.
        ttl_val_corr = 0.
        auTmodel.eval()
        clsmodel.eval()
        for features, labels in tqdm(val_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            with torch.no_grad():
                if includeAutoencoder(args):
                    attens, _ = auTmodel(features)
                else:
                    attens = auTmodel(features)
                outputs, _ = clsmodel(attens)
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_val_size += labels.shape[0]
            ttl_val_corr += (preds == labels).sum().cpu().item()
        ttl_val_accu = ttl_val_corr/ttl_val_size * 100.
        print(f'Validation size:{ttl_val_size:.0f}, accuracy:{ttl_val_accu:.2f}%')
        if max_val_accu <= ttl_val_accu:
            max_val_accu = ttl_val_accu
            torch.save(auTmodel.state_dict(), relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-auT{args.file_name_suffix}.pt'))
            torch.save(clsmodel.state_dict(), relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-cls{args.file_name_suffix}.pt'))
            if includeAutoencoder(args):
                torch.save(auDecoder.state_dict(), relative_path(args, f'{args.arch}-{dataset_tag(args.dataset)}-auD{args.file_name_suffix}.pt'))

        wandb.log({
            'Train/Accu': ttl_train_corr/ttl_train_size * 100.,
            'Train/Loss': ttl_train_loss/ttl_train_size,
            'Train/LR': learning_rate,
            'Val/Accu': ttl_val_corr/ttl_val_size * 100.,
        }, step=epoch, commit=True)

        if args.early_stop >= 0:
            if args.early_stop == epoch+1: exit()