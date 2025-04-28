import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from AuT.speech_commands.train import build_dataset
from AuT.lib.model import FCETransform, FCEClassifier
from AuT.speech_commands.fce_train import build_model

def inference(auT:FCETransform, auC:FCEClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = auC(auT(features)[0])
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]

    return ttl_corr / ttl_size * 100.0

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
    auTmodel.load_state_dict(state_dict=torch.load(args.original_auT_weight_path))
    clsmodel.load_state_dict(state_dict=torch.load(args.original_auC_weight_path))
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    print(f'Original accuracy is: {accu}%, sample size is: {len(test_dataset)}')