# containminate dataset
import argparse

import torch
from torchaudio import transforms

from lib.toolkit import print_argparse
from lib.wavUtils import Components, BackgroundNoise, AudioPadding, time_shift
from lib.spDataset import BackgroundNoiseDataset
from lib.datasets import load_from, TransferDataset
from AuT.speech_commands.train import build_dataest

def store_to(dataset: torch.utils.data.Dataset, root_path:str, index_file_name:str, args:argparse.Namespace, data_transf=None, label_transf=None, parallel=False) -> None:
    from lib.datasets import store_to as single_store_to, multi_process_store_to
    if parallel:
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=args.num_workers)
        multi_process_store_to(loader=data_loader, root_path=root_path, index_file_name=index_file_name, data_transf=data_transf, label_transf=label_transf)
    else:
        single_store_to(dataset=dataset, root_path=root_path, index_file_name=index_file_name, data_transf=data_transf, label_transf=label_transf)

def find_background_noise(args: argparse.Namespace) -> tuple[str, torch.Tensor]:
    background_noise_dataset = BackgroundNoiseDataset(root_path=args.dataset_root_path)
    for noise_type, noise, _ in background_noise_dataset:
        if args.corruption == noise_type:
            return noise_type, noise
    return ()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--output_path', type=str)
    ap.add_argument('--data_type', type=str, choices=['raw', 'final'], default='final')
    ap.add_argument('--rand_bg', action='store_true')

    ap.add_argument('--corruption', type=str, choices=['doing_the_dishes', 'dude_miaowing', 'exercise_bike', 'pink_noise', 'running_tap', 'white_noise', 'guassian_noise'])
    ap.add_argument('--severity_level', type=float, default=20)
    ap.add_argument('--cal_strong', action='store_true')
    ap.add_argument('--parallel', action='store_true')
    ap.add_argument('--num_workers', type=int, default=16)

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
        args.sample_rate=16000
    else:
        raise Exception('No support!')
    args.meta_file_name = 'speech_commands_meta.csv'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True

    print_argparse(args)
    ###############################################
    
    noise_type, noise = find_background_noise(args)
    audio_tsf = Components(transforms=[
        # AudioPadding(max_length=args.sample_rate, sample_rate=args.sample_rate, random_shift=True),
        BackgroundNoise(noise_level=args.severity_level, noise=noise, is_random=args.rand_bg)
    ])
    origin_dataset = build_dataest(args=args, tsf=audio_tsf, mode='test')
    print('Generate containminated dataset')
    store_to(dataset=origin_dataset, root_path=args.output_path, index_file_name=args.meta_file_name, args=args, parallel=args.parallel)

    # origin_dataset = load_from(root_path=args.output_path, index_file_name=args.meta_file_name)
    # print('Weak augmentation')
    # weak_path = f'{args.output_path}-weak'
    # store_to(dataset=origin_dataset, root_path=weak_path, index_file_name=args.meta_file_name, args=args, parallel=args.parallel)

    # print('Strong augmentation')
    # strong_path = f'{args.output_path}-strong'
    # strong_tf = Components(transforms=[
    #     transforms.PitchShift(sample_rate=args.sample_rate, n_steps=4, n_fft=512), # replace with random pitch shift
    #     time_shift(shift_limit=.17, is_random=True, is_bidirection=True)
    # ])
    # strong_dataset = TransferDataset(dataset=origin_dataset, data_tf=strong_tf, device=args.device)
    # store_to(dataset=strong_dataset, root_path=strong_path, index_file_name=args.meta_file_name, args=args, parallel=False)