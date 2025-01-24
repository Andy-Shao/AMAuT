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
from lib.wavUtils import pad_trunc, Components, AmplitudeToDB, time_shift, MelSpectrogramPadding
from lib.scDataset import SpeechCommandsDataset
from AuT.lib.model import AudioTransform, AudioClassifier, cal_model_tag, AudioDecoder
from AuT.lib.loss import CrossEntropyLabelSmooth, CosineSimilarityLoss
from AuT.lib.dataset import FrequenceTokenTransformer

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
    return args.embed_mode == 'CTA'

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
    def transformer_cfg(args:argparse.Namespace, cfg:ConfigDict) -> None:
        cfg.transform = ConfigDict()
        cfg.transform.layer_num = 12 if args.embed_size == 768 else 24
        cfg.transform.head_num = 12 if args.embed_size == 768 else 16
        cfg.transform.atten_drop_rate = .0
        cfg.transform.mlp_mid = 3072 if args.embed_size == 768 else 4096
        cfg.transform.mlp_dp_rt = .0
    
    def classifier_cfg(args:argparse.Namespace, cfg:ConfigDict) -> None:
        cfg.classifier = ConfigDict()
        cfg.classifier.class_num = args.class_num
        cfg.classifier.extend_size = 2048
        cfg.classifier.convergent_size = 256

    def decoder_cfg(args:argparse.Namespace, cfg:ConfigDict) -> None:
        decoder = ConfigDict()
        decoder.in_channels = [512, 128, 128]
        decoder.out_channels = [128, 128, args.n_mels]
        decoder.skip_channels = [512, 128, 0]
        decoder.hidden_size = args.embed_size
        cfg.decoder = decoder

    config = ConfigDict()
    config.embedding = ConfigDict()
    config.embedding.channel_num = args.n_mels
    config.embedding.marsked_rate = .15
    config.embedding.embed_size = args.embed_size
    config.embedding.mode = args.embed_mode
    config.embedding.in_shape = [args.n_mels, 104]

    transformer_cfg(args, config)
    classifier_cfg(args, config)
    auTmodel = AudioTransform(config=config).to(device=args.device)
    clsmodel = AudioClassifier(config=config).to(device=args.device)

    if includeAutoencoder(args):
        decoder_cfg(args, config)
        auDecoder = AudioDecoder(config=config).to(device=args.device)
    else:
        auDecoder = None

    return auTmodel, clsmodel, auDecoder

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
    ap.add_argument('--interval_num', type=int, default=50, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_dec', type=float, default=1.25)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--embed_mode', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--embed_size', type=int, default=768, choices=[768, 1024])

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
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
        project='AC-PT (AuT)', name=cal_model_tag(dataset_tag=args.dataset, pre_tag=args.embed_mode), 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    max_ms=1000
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    target_length=104
    tf_array = Components(transforms=[
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 104
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])

    train_dataset = build_dataest(args=args, tsf=tf_array, mode='train')
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    tf_array = Components(transforms=[
        pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    val_dataset = build_dataest(args=args, tsf=tf_array, mode='test')
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    interval = args.max_epoch // args.interval_num

    auTmodel, clsmodel, auDecoder = build_model(args=args)
    store_model_structure_to_txt(model=auTmodel, output_path=relative_path(args, 'auTmodel.txt'))
    store_model_structure_to_txt(model=clsmodel, output_path=relative_path(args, 'clsmodel.txt'))
    if includeAutoencoder(args):
        store_model_structure_to_txt(model=auDecoder, output_path=relative_path(args, 'auDecoder.txt'))
    print_weight_num(auT=auTmodel, auC=clsmodel, auD=auDecoder, args=args)
    loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)
    decoder_loss_fn = CosineSimilarityLoss(reduction='mean', dim=2).to(args.device)
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
            attens, hidden_attens = auTmodel(features)
            outputs = clsmodel(attens)
            if includeAutoencoder(args):
                gen_fts = auDecoder(attens, hidden_attens)
                loss = loss_fn(outputs, labels) + args.lr_dec * decoder_loss_fn(gen_fts, org_fts)
            else:
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
                attens, _ = auTmodel(features)
                outputs = clsmodel(attens)
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