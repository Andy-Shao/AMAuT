import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch 
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, relative_path, count_ttl_params
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from lib.datasets import load_from
from AuT.speech_commands.pre_train import build_dataest, build_model
from AuT.lib.model import AudioTransform, AudioClassifier

def inference(auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for features, labels in tqdm(data_loader):
        features, labels = features.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs = auC(auT(features))
            _, preds = torch.max(input=outputs.detach(), dim=1)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]

    return ttl_corr / ttl_size * 100.0

def load_model(args:argparse.Namespace, auT:AudioTransform, auC:AudioClassifier, mode='original'):
    assert mode in ['original', 'adapted'], 'No support'
    if mode == 'original':
        auT_path = args.original_auT_weight_path
        auC_path = args.original_auC_weight_path
    else:
        auT_path = args.adapted_auT_weight_path
        auC_path = args.adapted_auT_weight_path
    auT.load_state_dict(state_dict=torch.load(auT_path))
    auC.load_state_dict(state_dict=torch.load(auC_path))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--corrupted_data_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')

    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--arch_level', type=str, default='base')

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)
    ap.add_argument('--adapted_auT_weight_path', type=str)
    ap.add_argument('--adapted_auC_weight_path', type=str)
    ap.add_argument('--corruption', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'gaussian_noise'])
    ap.add_argument('--severity_level', type=float, default=.0025)
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    else:
        raise Exception('No support!')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'tta_analysis')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    args.meta_file_name = 'speech_commands_meta.csv'
    torch.backends.cudnn.benchmark = True
    accu_record = pd.DataFrame(columns=['dataset', 'algorithm', 'tta-operation', 'corruption', 'accuracy', 'error', 'severity level', 'number of weight'])
    
    print_argparse(args)
    ################################################################

    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    target_length=104
    tf_array = Components(transforms=[
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    test_dataset = build_dataest(args=args, tsf=tf_array, mode='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    auTmodel, clsmodel, _ = build_model(args=args)
    num_weight = count_ttl_params(model=auTmodel) + count_ttl_params(model=clsmodel)
    ttl_test_size = 0.
    ttl_test_curr = 0.

    print('Origin')
    load_model(args=args, auT=auTmodel, auC=clsmodel, mode='original')
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, pd.NA, pd.NA, accu, 100.0-accu, 0., num_weight]
    print(f'Original testing -- accuracy: {accu:.2f}%, sample size: {len(test_dataset)}')

    print('Corruption')
    corrupted_dataset = load_from(root_path=args.corrupted_data_root_path, index_file_name=args.meta_file_name, data_tf=tf_array)
    corrupted_loader = DataLoader(dataset=corrupted_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    load_model(args=args, auT=auTmodel, auC=clsmodel, mode='original')
    accu = inference(auT=auTmodel, auC=clsmodel, data_loader=corrupted_loader, args=args)
    accu_record.loc[len(accu_record)] = [args.dataset, args.arch, pd.NA, args.corruption, accu, 100.-accu, args.severity_level, num_weight]
    print(f'Corrupted testing -- accuracy: {accu:.2f}%, sample size: {len(corrupted_dataset)}')

    # Adaptation
    # TODO

    accu_record.to_csv(relative_path(args, args.output_csv_name))