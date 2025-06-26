import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse, make_unless_exits
from lib.datasets import dataset_tag, MultiTFDataset
from lib.spDataset import AudioMINST
from lib.wavUtils import Components, AudioPadding, time_shift, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from AuT.speech_commands.tta import build_optimizer, lr_scheduler, nucnm, g_entropy, entropy, load_weight
from AuT.speech_commands.fce_analysis import inference
from AuT.audiomnist.fce_train import build_model

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='AudioMNIST', choices=['AudioMNIST'])
    ap.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3, 4])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--cache_path', type=str)
    ap.add_argument('--file_suffix', type=str, default='')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_gamma', type=int, default=10)
    ap.add_argument('--lr_threshold', type=int, default=1)
    ap.add_argument('--auT_lr_decay', type=float, default=1.)
    ap.add_argument('--auC_lr_decay', type=float, default=1.)
    ap.add_argument('--nucnm_rate', type=float, default=1.)
    ap.add_argument('--ent_rate', type=float, default=1.)
    ap.add_argument('--gent_rate', type=float, default=1.)
    ap.add_argument('--gent_q', type=float, default=.9)
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--origin_auT_weight', type=str)
    ap.add_argument('--origin_cls_weight', type=str)
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--arch', type=str, default='FCE', choices=['FCE'])
    ap.add_argument('--arch_level', type=str, default='base')

    args = ap.parse_args()
    if args.dataset == 'AudioMNIST':
        args.class_num = 10
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'TTA', str(args.fold))
    make_unless_exits(args.output_path)
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    arch = 'AuT'
    wandb_run = wandb.init(
        project=f'{arch}-TTA', 
        name=f'{arch}-{dataset_tag(dataset_name=args.dataset)}-F{args.fold}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'Test-time Adaptation'])
    
    sample_rate=48000
    args.n_mels=80
    n_fft=2048
    win_length=800
    hop_length=302
    mel_scale='slaney'
    args.target_length=160
    if args.dataset == 'AudioMNIST':
        test_list = AudioMINST.default_splits(mode='test', fold=args.fold, root_path=args.dataset_root_path)
        shift_set = MultiTFDataset(
            dataset=AudioMINST(
                data_paths=test_list, include_rate=False, 
                data_trainsforms=Components(transforms=[
                    AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                ])
            ),
            tfs=[
                Components(transforms=[
                    time_shift(shift_limit=.17, is_random=True, is_bidirection=False),
                    a_transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        mel_scale=mel_scale
                    ), # 80 x 159
                    AmplitudeToDB(top_db=80., max_out=2.),
                    MelSpectrogramPadding(target_length=args.target_length),
                    FrequenceTokenTransformer()
                ]),
                Components(transforms=[
                    time_shift(shift_limit=-.17, is_random=True, is_bidirection=False),
                    a_transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                        mel_scale=mel_scale
                    ), # 80 x 159
                    AmplitudeToDB(top_db=80., max_out=2.),
                    MelSpectrogramPadding(target_length=args.target_length),
                    FrequenceTokenTransformer()
                ]),
            ]
        )
    shift_loader = DataLoader(
        dataset=shift_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )

    if args.dataset == 'AudioMNIST':
        test_set = AudioMINST(
            data_paths=test_list, include_rate=False, 
            data_trainsforms=Components(transforms=[
                AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=sample_rate),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ), # 80 x 159
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=args.target_length),
                FrequenceTokenTransformer()
            ])
        )
    test_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=False, drop_last=False, pin_memory=True,
        num_workers=args.num_workers
    )

    auTmodel, clsmodel = build_model(args)
    load_weight(args=args, auT=auTmodel, auC=clsmodel)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel)

    print('Pre-adaptation')
    accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
    print(f'Accurayc is: {accuracy:.4f}%, sample size is: {len(test_set)}')

    max_accu = 0.
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Adaptating...')
        auTmodel.train()
        clsmodel.train()
        ttl_size = 0.
        ttl_loss = 0.
        ttl_nucnm_loss = 0.
        ttl_ent_loss = 0.
        ttl_gent_loss = 0.
        for fs1, fs2, _ in tqdm(shift_loader):
            fs1, fs2 = fs1.to(args.device), fs2.to(args.device)

            optimizer.zero_grad()
            os1, _ = clsmodel(auTmodel(fs1)[0])
            os2, _ = clsmodel(auTmodel(fs2)[0])

            nucnm_loss = nucnm(args, os1) + nucnm(args, os2)
            ent_loss = entropy(args, os1) + entropy(args, os2)
            gent_loss = g_entropy(args, os1, q=args.gent_q) + g_entropy(args, os2, q=args.gent_q)

            loss = nucnm_loss + ent_loss + gent_loss
            loss.backward()
            optimizer.step()

            ttl_size += fs1.shape[0]
            ttl_loss += loss.cpu().item()
            ttl_nucnm_loss += nucnm_loss.cpu().item()
            ttl_ent_loss += ent_loss.cpu().item()
            ttl_gent_loss += gent_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality, gamma=args.lr_gamma, threshold=args.lr_threshold)

        print('Inferencing...')
        accuracy = inference(args=args, auT=auTmodel, auC=clsmodel, data_loader=test_loader)
        print(f'Accuracy is: {accuracy:.4f}%, sample size is: {len(shift_set)}')
        if accuracy >= max_accu:
            max_accu = accuracy
        torch.save(auTmodel.state_dict(), os.path.join(args.output_path, f'{arch}-{dataset_tag(args.dataset)}-auT{args.file_suffix}.pt'))
        torch.save(clsmodel.state_dict(), os.path.join(args.output_path, f'{arch}-{dataset_tag(args.dataset)}-cls{args.file_suffix}.pt'))

        wandb_run.log(
            data={
                'Loss/ttl_loss': ttl_loss / ttl_size,
                'Loss/Nuclear-norm loss': ttl_nucnm_loss / ttl_size,
                'Loss/Entropy loss': ttl_ent_loss / ttl_size,
                'Loss/G-entropy loss': ttl_gent_loss / ttl_size,
                'Adaptation/accuracy': accuracy,
                'Adaptation/LR': learning_rate,
                'Adaptation/max_accu': max_accu,
            }, step=epoch, commit=True
        )