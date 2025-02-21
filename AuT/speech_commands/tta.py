import argparse
import os
import numpy as np
import random 
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.datasets import dataset_tag
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from AuT.speech_commands.pre_train import build_dataest, build_model, includeAutoencoder, op_copy, lr_scheduler
from AuT.lib.model import AudioClassifier, AudioDecoder, AudioTransform

def inference(auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            if includeAutoencoder(args):
                attens, _ = auT(features)
            else: 
                attens = auT(features)
            outputs, _ = auC(attens)
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]

    return ttl_corr / ttl_size * 100.0

def load_model(args:argparse.Namespace, auT:AudioTransform, auD:AudioDecoder, auC:AudioClassifier):
    auT.load_state_dict(state_dict=torch.load(args.original_auT_weight_path))
    auC.load_state_dict(state_dict=torch.load(args.original_auC_weight_path))
    if includeAutoencoder(args):
        auD.load_state_dict(state_dict=torch.load(args.original_auD_weight_path))

def build_optimizer(args: argparse.Namespace, auT:AudioTransform, auC:AudioClassifier, auD:AudioDecoder) -> optim.Optimizer:
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands_v2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')

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

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)
    ap.add_argument('--original_auD_weight_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands_v2':
        args.class_num = 35
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
        project='AC-PT (AuT)', name=f'{args.arch}-{dataset_tag(args.dataset)}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    target_length=104
    tf_array = Components(transforms=[
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    test_dataset = build_dataest(args=args, tsf=tf_array, mode='test')
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    auTmodel, clsmodel, auDecoder = build_model(args=args)
    load_model(args=args, auT=auTmodel, auC=clsmodel, auD=auDecoder)
    decoder_loss_fn = nn.MSELoss(reduction='mean').to(device=args.device)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel, auD=auDecoder)

    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")

        auTmodel.train()
        clsmodel.train()
        if includeAutoencoder(args): auDecoder.train()
        for inputs, _ in tqdm(test_loader):
            inputs = inputs.to(args.device)
            org_inputs = torch.clone(inputs).detach().to(args.device)

            optimizer.zero_grad()
            attens, hidden_attens = auTmodel(inputs)
            outputs, features = clsmodel(attens)
            gen_fts = auDecoder(attens, hidden_attens)

            loss = decoder_loss_fn(gen_fts, org_inputs)
            loss.backward()
            optimizer.step()

        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
        print(f'accuracy is: {accu:.2f}%, sample size is: {len(test_dataset)}')