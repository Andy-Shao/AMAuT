import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, relative_path, count_ttl_params
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer, time_shift
from lib.wavUtils import BatchTransform
from AuT.speech_commands.train import build_dataset, build_model
from AuT.lib.model import AudioTransform, AudioClassifier

def merge_outs(o1:torch.Tensor, o2:torch.Tensor, o3:torch.Tensor) -> torch.Tensor:
    from torch.nn import functional as F
    return (F.softmax(o1, dim=1) + F.softmax(o2, dim=1) + F.softmax(o3, dim=1)) / 3.

def aug_inference(
    auT:AudioTransform, auC:AudioClassifier, args:argparse.Namespace, aug1:torch.nn.Module, 
    aug2:torch.nn.Module, no_aug:torch.nn.Module, raw_input:torch.Tensor
) -> torch.Tensor:
    ins1 = aug1(raw_input).to(args.device)
    ins2 = aug2(raw_input).to(args.device)
    ins3 = no_aug(raw_input).to(args.device)

    with torch.no_grad():
        o1, _ = auC(auT(ins1))
        o2, _ = auC(auT(ins2))
        o3, _ = auC(auT(ins3))
        
    return merge_outs(o1, o2, o3)

def elect_inference(
    auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace, aug1:torch.nn.Module, 
    aug2:torch.nn.Module, no_aug:torch.nn.Module
) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for inputs, labels in tqdm(data_loader):
        labels = labels.to(args.device)
        ins1 = aug1(inputs).to(args.device)
        ins2 = aug2(inputs).to(args.device)
        ins3 = no_aug(inputs).to(args.device)
        
        with torch.no_grad():
            o1, _ = auC(auT(ins1))
            o2, _ = auC(auT(ins2))
            o3, _ = auC(auT(ins3))
            outputs = merge_outs(o1, o2, o3)
            _, preds = torch.max(outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    return ttl_corr / ttl_size * 100.

def inference(auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, _ = auC(auT(features))
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]

    return ttl_corr / ttl_size * 100.0

def load_model(args:argparse.Namespace, auT:AudioTransform, auC:AudioClassifier, version=1):
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
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--arch_level', type=str, default='base')

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)
    ap.add_argument('--original_auT2_weight_path', type=str)
    ap.add_argument('--original_auC2_weight_path', type=str)
    ap.add_argument('--original_auT3_weight_path', type=str)
    ap.add_argument('--original_auC3_weight_path', type=str)

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

    auTmodel, clsmodel, _ = build_model(args=args)
    num_weight = count_ttl_params(model=auTmodel) + count_ttl_params(model=clsmodel)

    print('Origin')
    load_model(args=args, auT=auTmodel, auC=clsmodel)
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, pd.NA, accu, 100.0-accu, num_weight]
    print(f'Original testing -- accuracy: {accu:.2f}%, sample size: {len(test_dataset)}')

    print('Augmentation election inference')
    load_model(args=args, auT=auTmodel, auC=clsmodel)
    org_test_set = build_dataset(args=args, tsf=AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False), mode='test')
    org_test_loader = DataLoader(
        dataset=org_test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    ls = BatchTransform(tf=Components(transforms=[
        time_shift(shift_limit=-.17, is_random=False, is_bidirection=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ]))
    rs = BatchTransform(tf=Components(transforms=[
        time_shift(shift_limit=.17, is_random=False, is_bidirection=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ]))
    ns = BatchTransform(tf=Components(transforms=[
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=args.target_length),
        FrequenceTokenTransformer()
    ]))
    ttl_adpt_curr = elect_inference(
        auT=auTmodel, auC=clsmodel, data_loader=org_test_loader, args=args, aug1=ls, aug2=rs, no_aug=ns
    )
    print(f'Election accuracy is: {ttl_adpt_curr:.2f}%, samples size is: {len(org_test_set)}')
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, 'Aug_elect', ttl_adpt_curr, 100.-ttl_adpt_curr, num_weight]

    print('Multi-training election inference')
    load_model(args=args, auT=auTmodel, auC=clsmodel)
    auT2, cls2, _ = build_model(args=args)
    load_model(args=args, auT=auT2, auC=cls2, version=2)
    auT3, cls3, _ = build_model(args=args)
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