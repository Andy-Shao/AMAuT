import argparse
import os
import pandas as pd

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, count_ttl_params, relative_path
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.wavUtils import time_shift
from lib.spDataset import AudioMINST
from lib.datasets import MultiTFDataset
from AuT.audiomnist.fce_train import build_model
from AuT.speech_commands.fce_analysis import load_model, inference, aug_elect_inference, multi_train_inference, hybrid_inference

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='AudioMNIST', choices=['AudioMNIST'])
    ap.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
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
    if args.dataset == 'AudioMNIST':
        args.class_num = 10
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

    sample_rate=48000
    args.n_mels=80
    n_fft=2048
    win_length=800
    hop_length=302
    mel_scale='slaney'
    args.target_length=160
    tf_array = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 159
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])
    test_set = AudioMINST(
        data_paths=AudioMINST.default_splits(mode='test', fold=args.fold, root_path=args.dataset_root_path),
        data_trainsforms=tf_array, include_rate=False
    )
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print('Original')
    auTmodel, clsmodel = build_model(args)
    weigth_num = count_ttl_params(auTmodel) + count_ttl_params(clsmodel)
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    print(f'Original accuracy is: {accu:.4f}%, sample size is: {len(test_set)}')
    accu_record.loc[len(accu_record)] = [args.dataset, 'FCE', pd.NA, accu, 100. - accu, weigth_num]

    if args.only_origin:
        exit()

    print('Augmentation election refinement')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    aug_test_set = MultiTFDataset(
        dataset= AudioMINST(
            data_paths=AudioMINST.default_splits(mode='test', fold=args.fold, root_path=args.dataset_root_path),
            data_trainsforms=None, include_rate=False
        ),
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
    print(f'Multi-training election refinement accuracy is: {accu:.4f}%, sample size is: {len(test_set)}')
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