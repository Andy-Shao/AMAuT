import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse, count_ttl_params, relative_path
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.wavUtils import AudioClip
from lib.wavUtils import time_shift
from lib.datasets import MultiTFDataset
from lib.spDataset import VocalSound
from AuT.vocalsound.fce_train import build_model
from AuT.speech_commands.fce_analysis import load_model, merge_outs
from AuT.lib.model import FCEClassifier, FCETransform

def hybrid_inference(
    auT:FCETransform, auC:FCEClassifier, auT2:FCETransform, auC2:FCEClassifier, auT3:FCETransform, auC3:FCEClassifier,
    data_loader: DataLoader, args:argparse.Namespace
) -> float:
    def aug_inference(T:FCETransform, C:FCEClassifier, f1:torch.Tensor, f2:torch.Tensor, f3:torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            o1 = C(T(f1)[1])
            o2 = C(T(f2)[1])
            o3 = C(T(f3)[1])
        return merge_outs(o1, o2, o3, softmax=(args.merg_mode == 'softmax'))
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
        o = merge_outs(o1, o2, o3, softmax=(args.merg_mode == 'softmax'))
        _, preds = torch.max(input=o.detach(), dim=1)
        ttl_size += labels.shape[0]
        ttl_corr += (preds == labels).sum().cpu().item()
    return ttl_corr / ttl_size * 100.

def multi_train_inference(
    auT:FCETransform, auC:FCEClassifier, auT2:FCETransform, auC2:FCEClassifier, auT3:FCETransform, auC3:FCEClassifier,
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
            o1 = auC(auT(features)[1])
            o2 = auC2(auT2(features)[1])
            o3 = auC3(auT3(features)[1])
            outputs = merge_outs(o1, o2, o3, softmax=(args.merg_mode == 'softmax'))
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def aug_elect_inference(auT:FCETransform, auC:FCEClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for aug1, aug2, org, labels in tqdm(data_loader):
        aug1, aug2, org, labels = aug1.to(args.device), aug2.to(args.device), org.to(args.device), labels.to(args.device)

        with torch.no_grad():
            o1 = auC(auT(aug1)[1])
            o2 = auC(auT(aug2)[1])
            o3 = auC(auT(org)[1])
            outputs = merge_outs(o1, o2, o3, softmax=(args.merg_mode == 'softmax'))
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def inference(auT:FCETransform, auC:FCEClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = auC(auT(features)[1])
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]

    return ttl_corr / ttl_size * 100.0

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
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
    ap.add_argument('--merg_mode', type=str, default='origin', choices=['origin', 'softmax'])

    args = ap.parse_args()
    if args.dataset == 'VocalSound':
        args.class_num = 6
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
    max_length = sample_rate * 10
    args.n_mels=64
    n_fft=1024
    win_length=400
    hop_length=154
    mel_scale='slaney'
    args.target_length=1040
    tf_array = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=max_length),
        AudioClip(max_length=max_length, mode='head', is_random=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 1039
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])
    test_dataset = VocalSound(root_path=args.dataset_root_path, mode='test', data_tf=tf_array, version='16k', include_rate=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print('Original')
    auTmodel, clsmodel = build_model(args)
    weigth_num = count_ttl_params(auTmodel) + count_ttl_params(clsmodel)
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    print(f'Original accuracy is: {accu:.4f}%, sample size is: {len(test_dataset)}, number of weight: {weigth_num}')
    accu_record.loc[len(accu_record)] = [args.dataset, 'FCE', pd.NA, accu, 100. - accu, weigth_num]

    if args.only_origin:
        exit()

    print('Augmentation election refinement')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    aug_test_set = MultiTFDataset(
        dataset= VocalSound(root_path=args.dataset_root_path, mode='test', data_tf=None, version='16k', include_rate=False),
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