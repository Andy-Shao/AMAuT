import argparse
import os
import random 
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, relative_path
from lib.wavUtils import Components, AudioPadding, time_shift, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.spDataset import AudioMINST, FilterAudioMNIST
from AuT.audiomnist.pre_train import build_model
from AuT.speech_commands.pre_train import build_optimizer, lr_scheduler
from AuT.lib.loss import CrossEntropyLabelSmooth

def analyze_AM2(args:argparse.Namespace, train_tf:torch.nn.Module, test_tf:torch.nn.Module):
    records = pd.DataFrame(columns=['dataset', 'accuracy', 'training_set_size', 'test_set_size'])
    train_set = FilterAudioMNIST(root_path=args.dataset_root_path, data_tsf=train_tf, include_rate=False, filter_fn=lambda x: x['accent'] == 'German')
    test_set = FilterAudioMNIST(root_path=args.dataset_root_path, data_tsf=test_tf, include_rate=False, filter_fn=lambda x: x['accent'] != 'German')
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    auTmodel, clsmodel = build_model(args)
    auTmodel.load_state_dict(state_dict=torch.load(args.original_auT_weight_path))
    clsmodel.load_state_dict(state_dict=torch.load(args.original_auC_weight_path))

    ttl_test_size = 0.
    ttl_test_corr = 0.
    auTmodel.eval()
    clsmodel.eval()
    for features, labels in tqdm(test_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = clsmodel(auTmodel(features))
            _, preds = torch.max(outputs.detach(), dim=1)
        ttl_test_size += labels.shape[0]
        ttl_test_corr += (preds == labels).sum().cpu().item()
    ttl_accu = ttl_test_corr / ttl_test_size * 100.
    records.loc[len(records)] = [args.dataset, ttl_accu, len(train_set), len(test_set)]
    records.to_csv(relative_path(args, args.output_csv_name), sep=',')

def analyze_AM(args:argparse.Namespace, train_tf:torch.nn.Module, test_tf:torch.nn.Module):
    records = pd.DataFrame(columns=['dataset', 'fold', 'accuracy', 'training_set_size', 'test_set_size'])
    for fold in range(5):
        train_list = AudioMINST.default_splits(mode='train', fold=fold, root_path=args.dataset_root_path)
        train_list += AudioMINST.default_splits(mode='validate', fold=fold, root_path=args.dataset_root_path)
        train_set = AudioMINST(data_paths=args.dataset_root_path, data_trainsforms=train_tf, include_rate=False)
        train_loader = DataLoader(
            dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
        )

        test_list = AudioMINST.default_splits(mode='test', fold=fold, root_path=args.dataset_root_path)
        test_set = AudioMINST(data_paths=test_list, data_trainsforms=test_tf, include_rate=False)
        test_loader = DataLoader(
            dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
        )

        auTmodel, clsmodel = build_model(args)
        optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel, auD=None)
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.class_num, use_gpu=torch.cuda.is_available(), epsilon=args.smooth)

        ttl_train_size = 0.
        ttl_train_corr = 0.
        ttl_train_loss = 0.
        auTmodel.train()
        clsmodel.train()
        for epoch in range(args.max_epoch):
            print(f'Fold: {fold}, Epoch {epoch+1}/{args.max_epoch}')

            for features, labels in tqdm(train_loader):
                features, labels = features.to(args.device), labels.to(args.device)

                optimizer.zero_grad()
                outputs, _ = clsmodel(auTmodel(features))
                loss = loss_fn(outputs, labels)
                optimizer.step()

                ttl_train_size += labels.shape[0]
                _, preds = torch.max(outputs.detach(), dim=1)
                ttl_train_corr += (preds == labels).sum().cpu().item()
                ttl_train_loss += loss.cpu().item()
            print(f'Training size:{ttl_train_size:.0f}, train accuracy:{ttl_train_corr/ttl_train_size * 100.:.2f}%')

            if epoch % args.interval == 0:
                lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        auTmodel.eval()
        clsmodel.eval()
        ttl_test_size = 0.
        ttl_test_corr = 0.
        for features, labels in tqdm(test_loader):
            features, labels = features.to(args.device), labels.to(args.device)

            with torch.no_grad():
                outputs, _ = clsmodel(auTmodel(features))
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_test_size += labels.shape[0]
            ttl_test_corr += (preds == labels).sum().cpu().item()
        test_accuracy = ttl_test_corr/ttl_test_size * 100.
        print(f'Testing size: {ttl_test_size:.0f}, testing accuracy:{test_accuracy:.2f}%')
        records.loc[len(records)] = [args.dataset, fold, test_accuracy, len(train_set), len(test_set)]

    records.to_csv(relative_path(args, args.output_csv_name), sep=',')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='AudioMNIST', choices=['AudioMNIST', 'AudioMNIST2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='accuracy_record.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='AudioMNIST')

    ap.add_argument('--seed', type=int, default=2025, help='random seed')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--arch_level', type=str, default='base')

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)

    args = ap.parse_args()
    if args.dataset == 'AudioMNIST':
        args.class_num = 10
    elif args.dataset == 'AudioMNIST2':
        args.class_num = 10
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'accu_analysis')
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

    sample_rate=48000
    args.n_mels=80
    n_fft=2048
    win_length=800
    hop_length=302
    mel_scale='slaney'
    args.target_length=160
    train_tf = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=True, max_length=sample_rate),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 159
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])

    test_tf = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 159
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ])

    if args.dataset == 'AudioMNIST':
        analyze_AM(args=args, train_tf=train_tf, test_tf=test_tf)
    elif args.dataset == 'AudioMNIST2':
        analyze_AM2(args=args, train_tf=train_tf, test_tf=test_tf)