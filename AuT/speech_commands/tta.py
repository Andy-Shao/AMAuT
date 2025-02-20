import argparse
import os
import numpy as np
import random 
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.datasets import dataset_tag, TransferDataset, Dataset_Idx, load_from
from lib.wavUtils import Components, MelSpectrogramPadding, AudioPadding, AmplitudeToDB, FrequenceTokenTransformer, time_shift
from lib.wavUtils import RandomPitchShift, RandomSpeed
from AuT.speech_commands.pre_train import build_dataest, build_model, lr_scheduler, includeAutoencoder, op_copy
from AuT.speech_commands.tta_analysis import load_model
from AuT.lib.loss import soft_CE
from AuT.lib.model import AudioTransform, AudioClassifier, AudioDecoder

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

def inference(auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> tuple[float, np.ndarray]:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for idx, (features, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = auC(auT(features))
            _, preds = torch.max(input=outputs.detach(), dim=1)
            if idx == 0:
                all_output = outputs.float().cpu()
            else:
                all_output = torch.cat([all_output, outputs.float().cpu()], dim=0)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    all_output = nn.Softmax(dim=1)(all_output)

    return ttl_corr / ttl_size * 100.0, torch.mean(all_output, dim=0).numpy()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands_v2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--corrupted_data_root_path', type=str)
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

    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')
    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands_v2':
        args.class_num = 35
    else:
        raise Exception('No support!')
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda'
    assert torch.cuda.is_available(), 'No support cpu mode'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'TTA')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    args.meta_file_name = 'speech_commands_meta.csv'
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    wandb_run = wandb.init(
        project='AC-TTA (AuT)', name=f'{args.arch}-{dataset_tag(args.dataset)}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    target_length=104
    # test_set = build_dataest(args=args, mode='test', tsf=None)
    test_set = load_from(root_path=args.corrupted_data_root_path, index_file_name=args.meta_file_name)
    test_loader = DataLoader(
        dataset=TransferDataset(
            dataset=test_set, 
            data_tf=Components(transforms=[
                AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=target_length),
                FrequenceTokenTransformer()
            ])),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    tf_array = Components(transforms=[
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
        # time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    weak_set = TransferDataset(dataset=test_set, data_tf=tf_array, device='cpu', keep_cpu=True)
    weak_set = Dataset_Idx(dataset=weak_set)
    weak_loader = DataLoader(dataset=weak_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = Components(transforms=[
        # RandomSpeed(start_fq=.9, end_fq=1.11, sample_rate=sample_rate, max_length=sample_rate, step=.01, max_it=5),
        RandomPitchShift(step_rang=[0,1,2,3,4], sample_rate=sample_rate),
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    strong_set = TransferDataset(dataset=test_set, data_tf=tf_array, device=args.device, keep_cpu=False)

    auTmodel, clsmodel, _ = build_model(args=args)
    load_model(args=args, auC=clsmodel, auT=auTmodel, mode='original')
    optimizer = build_optimizer(args=args, auC=clsmodel, auT=auTmodel, auD=None)

    max_accu = 0.
    print('Pre-inferencing...')
    accu, mean_all_output = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
    print(f'Test accuracy: {accu:.2f}%, test set sample size is: {len(test_set)}')
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")

        print('Adaptating...')
        auTmodel.train()
        clsmodel.train()
        adatpting_time = len(weak_loader)
        ttl_loss = 0.
        ttl_fbnm_loss = 0.
        ttl_cst_loss = 0.
        for weak_features, _, ids in tqdm(weak_loader):
            batch_size = weak_features.shape[0]
            weak_features = weak_features.to(args.device)
            strong_features = torch.cat([torch.unsqueeze(strong_set[i][0], dim=0) for i in ids], dim=0)
            features = torch.cat([weak_features, strong_features], dim=0)

            optimizer.zero_grad()
            outputs = clsmodel(auTmodel(features))

            # fbnm -> Nuclear-norm Maximization loss
            if args.fbnm_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_output,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_output.shape[0],softmax_output.shape[1])])
                fbnm_loss = args.fbnm_par*fbnm_loss
            else:
                fbnm_loss = torch.tensor(.0).cuda()

            # Consist loss -> soft cross-entropy loss
            if args.const_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                expectation_ratio = mean_all_output/torch.mean(softmax_output[0:batch_size],dim=0)
                with torch.no_grad():
                    soft_label_norm = torch.norm(softmax_output[0:batch_size]*expectation_ratio,dim=1,keepdim=True) #Frobenius norm
                    soft_label = (softmax_output[0:batch_size]*expectation_ratio)/soft_label_norm
                consistency_loss = args.const_par*torch.mean(soft_CE(softmax_output[batch_size:],soft_label))
            else:
                consistency_loss = torch.tensor(.0).cuda()

            loss = fbnm_loss + consistency_loss
            loss.backward()
            optimizer.step()
            ttl_loss += loss
            ttl_fbnm_loss += fbnm_loss.cpu().item()
            ttl_cst_loss += consistency_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        print('Inferencing...')
        accu, mean_all_output = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
        mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
        print(f'Test accuracy: {accu:.2f}%, test set sample size is: {len(test_set)}')
        wandb.log({
            "LOSS/total loss":ttl_loss / adatpting_time,
            "LOSS/consistency loss": ttl_cst_loss / adatpting_time,
            "LOSS/Nuclear-norm Maximization loss":ttl_fbnm_loss / adatpting_time,
            "Accuracy/classifier accuracy": accu
        }, step=epoch, commit=True)