import argparse
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse
from lib.wavUtils import Components, AudioPadding, AudioClip, AmplitudeToDB, MelSpectrogramPadding
from lib.wavUtils import FrequenceTokenTransformer
from lib.spDataset import VocalSound
from AuT.vocalsound.fce_train import build_model
from AuT.speech_commands.fce_analysis import load_model

def collect_preds(args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, dataloader:DataLoader) -> torch.Tensor:
    aut.eval(); clsf.eval()
    for idx, (features, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs = clsf(aut(features)[1])
        _, preds = torch.max(outputs.cpu().detach(), dim=1)
        if idx == 0:
            ttl_preds = preds
        else:
            ttl_preds = torch.concat([ttl_preds, preds], dim=0)
    return ttl_preds

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

    auTmodel, clsmodel = build_model(args)
    print('Collecting prediction 01')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=1)
    pred1 = collect_preds(args=args, aut=auTmodel, clsf=clsmodel, dataloader=test_loader)
    print('Collecting prediction 02')
    load_model(args=args, auT=auTmodel, auC=clsmodel, version=2)
    pred2 = collect_preds(args=args, aut=auTmodel, clsf=clsmodel, dataloader=test_loader)

    cm = confusion_matrix(y_true=pred1.numpy(), y_pred=pred2.numpy())
    agreement_rate = np.trace(cm) / np.sum(cm)
    print(f'The agreement rate from confusion matrix is {agreement_rate:.4f}')