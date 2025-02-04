import argparse
import wandb
from ml_collections import ConfigDict

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as a_transforms

from lib.toolkit import print_argparse
from lib.wavUtils import Components, AudioPadding, time_shift, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer, VisionTokenTransformer
from lib.scDataset import SpeechCommandsDataset
from lib.datasets import TwoTFDataset
from AuT.lib.model import cal_model_tag
from AuT.speech_commands.pre_train import build_dataest
from AuT.lib.config import transformer_cfg
from MsT.lib.model import BiEmbedTransformer

def build_model(args:argparse.Namespace) -> BiEmbedTransformer:
    cfg = ConfigDict()
    transformer_cfg(embed_size=args.embed_size, cfg=cfg)



if __name__ == '__main__':
    ap = argparse.ArgumentParser('MeT')
    args = ap.parse_args()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='speech-commands')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--model_topology', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='MeT', choices=['MeT'])
    ap.add_argument('--embed_size', type=int, default=768, choices=[768, 1024])

    print_argparse(args)
    #########################################

    wandb_run = wandb.init(
        project='AC-PT (AuT)', name=cal_model_tag(dataset_tag=args.dataset, pre_tag=args.arch), 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    max_ms=1000
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=163
    mel_scale='slaney'
    target_length=100
    tf_array = Components(transforms=[
        AudioPadding(max_ms=max_ms, sample_rate=sample_rate, random_shift=True),
        time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ), # 80 x 100
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
    ])
    train_dataset = TwoTFDataset(
        dataset=build_dataest(args=args, tsf=tf_array, mode='train'), 
        tf1=FrequenceTokenTransformer(),
        tf2=VisionTokenTransformer()
    )
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers
    )

    tf_array = Components(transforms=[
        AudioPadding(max_ms=max_ms, sample_rate=sample_rate, random_shift=False),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    val_dataset = TwoTFDataset(
        dataset=build_dataest(args=args, tsf=tf_array, mode='test'),
        tf1=FrequenceTokenTransformer(),
        tf2=VisionTokenTransformer()
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
