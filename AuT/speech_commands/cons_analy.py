import argparse
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

import torch 
from torch import nn
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from AuT.speech_commands.train import build_dataset
from AuT.speech_commands.fce_train import build_model
from AuT.speech_commands.fce_analysis import load_model

def collect_preds(args:argparse.Namespace, aut:nn.Module, clsf:nn.Module, dataloader:DataLoader) -> torch.Tensor:
    aut.eval(); clsf.eval()
    for idx, (features, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.inference_mode():
            outputs, _ = clsf(aut(features)[0])
        _, preds = torch.max(outputs.cpu().detach(), dim=1)
        if idx == 0:
            ttl_preds = preds
        else:
            ttl_preds = torch.concat([ttl_preds, preds], dim=0)
    return ttl_preds

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