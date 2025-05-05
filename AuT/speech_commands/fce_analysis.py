import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse, count_ttl_params, relative_path
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.wavUtils import time_shift, GuassianNoise
from lib.datasets import MultiTFDataset
from AuT.speech_commands.train import build_dataset
from AuT.lib.model import FCETransform, AudioClassifier
from AuT.speech_commands.fce_train import build_model

def hybrid_inference(
    auT:FCETransform, auC:AudioClassifier, auT2:FCETransform, auC2:AudioClassifier, auT3:FCETransform, auC3:AudioClassifier,
    data_loader: DataLoader, args:argparse.Namespace
) -> float:
    def aug_inference(T:FCETransform, C:AudioClassifier, f1:torch.Tensor, f2:torch.Tensor, f3:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            o1, _ = C(T(f1)[0])
            o2, _ = C(T(f2)[0])
            o3, _ = C(T(f3)[0])
        return merge_outs(o1, o2, o3)
    auT.eval()
    auC.eval()
    auT2.eval()
    auC2.eval()
    auT3.eval()
    auC3.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for aug1, aug2, org, labels in tqdm(data_loader):
        aug1, aug2, org, labels = aug1.to(args.device), aug2.to(args.device), org.to(args.device), labels.to(args.device)

        o1 = aug_inference(T=auT, C=auC, f1=aug1, f2=aug2, f3=org)
        o2 = aug_inference(T=auT2, C=auC2, f1=aug1, f2=aug2, f3=org)
        o3 = aug_inference(T=auT3, C=auC3, f1=aug1, f2=aug2, f3=org)
        o = merge_outs(o1, o2, o3)
        _, preds = torch.max(input=o.detach(), dim=1)
        ttl_size += labels.shape[0]
        ttl_corr += (preds == labels).sum().cpu().item()
    return ttl_corr / ttl_size * 100.

def multi_train_inference(
    auT:FCETransform, auC:AudioClassifier, auT2:FCETransform, auC2:AudioClassifier, auT3:FCETransform, auC3:AudioClassifier,
    data_loader: DataLoader, args:argparse.Namespace
) -> float:
    auT.eval()
    auC.eval()
    auT2.eval()
    auC2.eval()
    auT3.eval()
    auC3.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            o1, _ = auC(auT(features)[0])
            o2, _ = auC2(auT2(features)[0])
            o3, _ = auC3(auT3(features)[0])
            outputs = merge_outs(o1, o2, o3)
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def merge_outs(o1:torch.Tensor, o2:torch.Tensor, o3:torch.Tensor, softmax:bool=False) -> torch.Tensor:
    if softmax:
        from torch.nn import functional as F
        return (F.softmax(o1, dim=1) + F.softmax(o2, dim=1) + F.softmax(o3, dim=1)) / 3.
    else: return (o1 + o2 + o3) / 3.

def aug_elect_inference(auT:FCETransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for aug1, aug2, org, labels in tqdm(data_loader):
        aug1, aug2, org, labels = aug1.to(args.device), aug2.to(args.device), org.to(args.device), labels.to(args.device)

        with torch.no_grad():
            o1, _ = auC(auT(aug1)[0])
            o2, _ = auC(auT(aug2)[0])
            o3, _ = auC(auT(org)[0])
            outputs = merge_outs(o1, o2, o3)
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def inference(auT:FCETransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = auC(auT(features)[0])
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]

    return ttl_corr / ttl_size * 100.0

def load_model(args:argparse.Namespace, auT:FCETransform, auC:AudioClassifier, version=1):
    if version == 1:
        auT_path = args.original_auT_weight_path
        auC_path = args.original_auC_weight_path
    elif version == 2:
        auT_path = args.original_auT2_weight_path
        auC_path = args.original_auC2_weight_path
    elif version == 3:
        auT_path = args.original_auT3_weight_path
        auC_path = args.original_auC3_weight_path
    auT.load_state_dict(state_dict=torch.load(auT_path))
    auC.load_state_dict(state_dict=torch.load(auC_path))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands_v2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')

    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--arch', type=str, default='FCE', choices=['FCE'])
    ap.add_argument('--arch_level', type=str, default='base')

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)
    ap.add_argument('--original_auT2_weight_path', type=str)
    ap.add_argument('--original_auC2_weight_path', type=str)
    ap.add_argument('--original_auT3_weight_path', type=str)
    ap.add_argument('--original_auC3_weight_path', type=str)

    ap.add_argument('--only_origin', action='store_true')

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands_v2':
        args.class_num = 35
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'tta_analysis')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    torch.backends.cudnn.benchmark = True
    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'accuracy', 'error', 'number of weight'])
    
    print_argparse(args)
    ################################################################

    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    args.target_length=104
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
    test_dataset = build_dataset(args=args, tsf=tf_array, mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print('Original')
    auTmodel, clsmodel = build_model(args)
    weigth_num = count_ttl_params(auTmodel) + count_ttl_params(clsmodel)
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    print(f'Original accuracy is: {accu:.4f}%, sample size is: {len(test_dataset)}')
    accu_record.loc[len(accu_record)] = [args.dataset, 'FCE', pd.NA, accu, 100. - accu, weigth_num]

    if args.only_origin:
        exit()

    print('Augmentation election refinement')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    aug_test_set = MultiTFDataset(
        dataset= build_dataset(args=args, mode='test', tsf=AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False)),
        tfs=[
            Components(transforms=[
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                time_shift(shift_limit=-.17, is_random=False, is_bidirection=False),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
            Components(transforms=[
                time_shift(shift_limit=.17, is_random=False, is_bidirection=False),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ]),
        ]
    )
    aug_test_loader = DataLoader(dataset=aug_test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    accu = aug_elect_inference(auT=auTmodel, auC=clsmodel, data_loader=aug_test_loader, args=args)
    print(f'Augmentation election refinement accuracy is: {accu:.4f}%, sample size is: {len(aug_test_set)}')
    accu_record.loc[len(accu_record)] = [args.dataset, 'FCE', 'aug-elect', accu, 100. - accu, weigth_num]

    print('Multi-training election refinement')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    auTmodel2, clsmodel2 = build_model(args=args)
    load_model(args=args, auT=auTmodel2, auC=clsmodel2, version=2)
    auTmodel3, clsmodel3 = build_model(args=args)
    load_model(args=args, auT=auTmodel3, auC=clsmodel3, version=3)
    accu = multi_train_inference(
        auT=auTmodel, auC=clsmodel, auT2=auTmodel2, auC2=clsmodel2, auT3=auTmodel3, auC3=clsmodel3, 
        data_loader=test_loader, args=args
    )
    print(f'Multi-training election refinement accuracy is: {accu:.4f}%, sample size is: {len(test_dataset)}')
    accu_record.loc[len(accu_record)] = [args.dataset, 'FCE', 'multi-train', accu, 100. - accu, weigth_num]

    print('Hybrid election refinement')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    load_model(args=args, auT=auTmodel2, auC=clsmodel2, version=2)
    load_model(args=args, auT=auTmodel3, auC=clsmodel3, version=3)
    accu = hybrid_inference(
        auT=auTmodel, auC=clsmodel, auT2=auTmodel2, auC2=clsmodel2, auT3=auTmodel3, auC3=clsmodel3,
        data_loader=aug_test_loader, args=args
    )
    print(f'Hybrid election refinement accuracy is: {accu:.4f}%, sample size is: {len(aug_test_set)}')
    accu_record.loc[len(accu_record)] = [args.dataset, 'FCE', 'hybrid-elect', accu, 100. - accu, weigth_num]

    accu_record.to_csv(relative_path(args, args.output_csv_name))