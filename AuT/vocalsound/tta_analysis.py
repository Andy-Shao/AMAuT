import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse, count_ttl_params, relative_path
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, AudioClip, FrequenceTokenTransformer
from lib.wavUtils import BatchTransform, time_shift
from lib.spDataset import VocalSound
from AuT.vocalsound.train import build_model
from AuT.speech_commands.tta_analysis import load_model, inference, elect_inference, aug_inference, merge_outs

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='VocalSound', choices=['VocalSound'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')

    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--arch_level', type=str, default='base')

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)
    ap.add_argument('--original_auT2_weight_path', type=str)
    ap.add_argument('--original_auC2_weight_path', type=str)
    ap.add_argument('--original_auT3_weight_path', type=str)
    ap.add_argument('--original_auC3_weight_path', type=str)

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
    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'accuracy', 'error', 'number of weight'])
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ##########################################

    sample_rate=16000
    max_length = sample_rate * 3
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=301
    mel_scale='slaney'
    args.target_length=160
    tf_array = Components(transforms=[
        AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=max_length),
        AudioClip(max_length=max_length, mode='mid', is_random=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 160
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ])
    test_set = VocalSound(root_path=args.dataset_root_path, mode='test', include_rate=False, data_tf=tf_array, version='16k')
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    auTmodel, clsmodel = build_model(args)
    num_weight = count_ttl_params(auTmodel) + count_ttl_params(clsmodel)

    print('Origin')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, pd.NA, accu, 100.0-accu, num_weight]
    print(f'Original testing -- accuracy: {accu:.2f}%, sample size: {len(test_set)}')

    print('Augmentation election inference')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    org_test_set = VocalSound(
        root_path=args.dataset_root_path, mode='test', include_rate=False,
        data_tf=Components(transforms=[
            AudioPadding(sample_rate=sample_rate, random_shift=False, max_length=max_length),
            AudioClip(max_length=max_length, mode='mid', is_random=False),
        ]),
        version='16k'
    )
    org_test_loader = DataLoader(dataset=org_test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    ls = BatchTransform(tf=Components(transforms=[
        time_shift(shift_limit=-.17, is_random=False, is_bidirection=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 160
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ]))
    rs = BatchTransform(tf=Components(transforms=[
        time_shift(shift_limit=.17, is_random=False, is_bidirection=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 160
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ]))
    ns = BatchTransform(tf=Components(transforms=[
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 160
        AmplitudeToDB(top_db=80., max_out=2.),
        FrequenceTokenTransformer()
    ]))
    ttl_adpt_curr = elect_inference(
        auT=auTmodel, auC=clsmodel, data_loader=org_test_loader, args=args, aug1=ls, aug2=rs, no_aug=ns
    )
    print(f'Election accuracy is: {ttl_adpt_curr:.2f}%, samples size is: {len(org_test_set)}')
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, 'Aug_elect', ttl_adpt_curr, 100.-ttl_adpt_curr, num_weight]

    print('Multi-training election inference')
    load_model(args=args, auT=auTmodel, auC=clsmodel)
    auT2, cls2, = build_model(args=args)
    load_model(args=args, auT=auT2, auC=cls2, version=2)
    auT3, cls3, = build_model(args=args)
    load_model(args=args, auT=auT3, auC=cls3, version=3)
    ttl_test_size = 0.
    ttl_test_curr = 0.
    auTmodel.eval()
    clsmodel.eval()
    auT2.eval()
    cls2.eval()
    auT3.eval()
    cls3.eval()
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        with torch.no_grad():
            o1, _ = clsmodel(auTmodel(inputs))
            o2, _ = cls2(auT2(inputs))
            o3, _ = cls3(auT3(inputs))
            o = merge_outs(o1, o2, o3)
            _, preds = torch.max(o.detach(), dim=1)
        ttl_test_curr += (preds == labels).sum().cpu().item()
        ttl_test_size += labels.shape[0]
    ttl_test_accu = ttl_test_curr / ttl_test_size * 100.
    print(f'Accuracy is: {ttl_test_accu:.2f}%, sample size is:{ttl_test_size:.0f}')
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, 'Mlt-train_elect', ttl_test_accu, 100.-ttl_test_accu, num_weight]

    print('Hybrid election inference')
    ttl_test_size = 0.
    ttl_test_curr = 0.
    auTmodel.eval()
    clsmodel.eval()
    auT2.eval()
    cls2.eval()
    auT3.eval()
    cls3.eval()
    for inputs, labels in tqdm(org_test_loader):
        labels = labels.to(args.device)

        o1 = aug_inference(auT=auTmodel, auC=clsmodel, args=args, aug1=ls, aug2=rs, no_aug=ns, raw_input=inputs)
        o2 = aug_inference(auT=auT2, auC=cls2, args=args, aug1=ls, aug2=rs, no_aug=ns, raw_input=inputs)
        o3 = aug_inference(auT=auT3, auC=cls3, args=args, aug1=ls, aug2=rs, no_aug=ns, raw_input=inputs)
        o = merge_outs(o1, o2, o3)
        _, preds = torch.max(o.detach(), dim=1)
        ttl_test_curr += (preds == labels).sum().cpu().item()
        ttl_test_size += labels.shape[0]
    ttl_test_accu = ttl_test_curr / ttl_test_size * 100.
    print(f'Hybrid election accuracy is: {ttl_test_accu:.2f}%, sample size is:{ttl_test_size:.0f}')
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, 'Hybrid_elect', ttl_test_accu, 100.-ttl_test_accu, num_weight]

    accu_record.to_csv(relative_path(args, args.output_csv_name))