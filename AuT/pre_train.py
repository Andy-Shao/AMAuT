import argparse
import os
import numpy as np
import random
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as v_transforms
import torchaudio.transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse, cal_norm, store_model_structure_to_txt
from lib.wavUtils import Components, pad_trunc, time_shift
from lib.datasets import load_datapath, AudioMINST, ClipDataset
from AuT.lib.models import AuT, AudioClassifier
from CoNMix.lib.loss import CrossEntropyLabelSmooth

def lr_scheduler(optimizer: torch.optim.Optimizer, epoch:int, max_epoch:int, gamma=10, power=0.75) -> optim.Optimizer:
    decay = (1 + gamma * epoch / max_epoch) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = .9
        param_group['nestenv'] = True
    return optimizer

def op_copy(optimizer: optim.Optimizer) -> optim.Optimizer:
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def build_optimizer(args: argparse.Namespace, auT:nn.Module, auC:nn.Module) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in auT.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate * .1}]
    for k, v in auC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def load_models(args: argparse.Namespace) -> tuple[nn.Module, nn.Module]:
    auT = AuT(in_channels=1).to(device=args.device)
    auC = AudioClassifier(
        type=args.classifier, feature_dim=auT.out_features, bottleneck_dim=args.bottleneck, cls_type=args.layer,
        class_num=args.class_num
    )
    auC = auC.to(device=args.device)
    return auT, auC

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='audio-mnist', choices=['audio-mnist'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')
    ap.add_argument('--output_csv_name', type=str, default='training_records.csv')
    ap.add_argument('--output_weight_prefix', type=str, default='audio-mnist')
    ap.add_argument('--temporary_path', type=str)
    ap.add_argument('--wandb', action='store_true')

    ap.add_argument('--seed', type=int, default=2024, help='random seed')
    ap.add_argument('--normalized', action='store_true')
    ap.add_argument('--test_rate', type=float, default=.3)

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_gamma', type=float, default=10)
    ap.add_argument('--bottleneck', type=int, default=256)
    ap.add_argument('--epsilon', type=float, default=1e-5)
    ap.add_argument('--layer', type=str, default='wn', choices=['linear', 'wn'])
    ap.add_argument('--classifier', type=str, default='bn', choices=['ori', 'bn'])
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--trte', type=str, default='val', choices=['full', 'val'])

    args = ap.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'pre_train')
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

    wandb_run = wandb.init(
        project='Audio Classification Pre-Training (AuT)', name=args.dataset, mode='online' if args.wandb else 'disabled',
        config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    if args.dataset == 'audio-mnist':
        args.class_num = 10
        max_ms=1000
        sample_rate=48000
        n_mels=80
        n_fft = 4096
        hop_length= n_fft // 2
        train_tf = [
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            time_shift(shift_limit=.25, is_random=True, is_bidirection=True),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
        ]
        train_data_paths = load_datapath(root_path=args.dataset_root_path, filter_fn=lambda x: x['accent'] == 'German')
        if args.normalized:
            print('calculate train dataset mean and standard deviation')
            train_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=train_tf), include_rate=False)
            mean_vals, std_vals = cal_norm(loader=DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, drop_last=False))
            train_tf.append(v_transforms.Normalize(mean=mean_vals, std=std_vals))
        train_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=train_tf), include_rate=False)
        test_tf = [
            pad_trunc(max_ms=max_ms, sample_rate=sample_rate),
            a_transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length),
            a_transforms.AmplitudeToDB(top_db=80),
        ]
        if args.normalized:
            print('calculat test dataset mean and standard deviation')
            test_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=test_tf), include_rate=False)
            mean_vals, std_vals = cal_norm(loader=DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, drop_last=False))
            test_tf.append(v_transforms.Normalize(mean=mean_vals, std=std_vals))
        test_dataset = AudioMINST(data_paths=train_data_paths, data_trainsforms=Components(transforms=test_tf), include_rate=False)
        test_dataset = ClipDataset(dataset=test_dataset, rate=args.test_rate)
    else:
        raise Exception('No support')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    auT, auC = load_models(args=args)
    store_model_structure_to_txt(model=auT, output_path=os.path.join(args.full_output_path, 'AuT_structure.txt'))
    store_model_structure_to_txt(model=auC, output_path=os.path.join(args.full_output_path, 'AuC_structure.txt'))
    optimizer = build_optimizer(args, auT=auT, auC=auC)
    classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth, use_gpu=torch.cuda.is_available())

    best_accuracy = 0.
    ttl_train_loss = 0.
    ttl_train_num = 0
    ttl_train_corr = 0
    for epoch in range(args.max_epoch):
        print(f'Epoch {epoch+1}/{args.max_epoch}')
        print('Training...')
        auT.train()
        auC.train()
        for features, labels in tqdm(train_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            outputs = auC(auT(features))
            loss = classifier_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs.detach(), dim=1)   
            ttl_train_corr += (preds == labels).sum().cpu().item()
            ttl_train_loss += loss.cpu().item()
            ttl_train_num += labels.shape[0]
            features = None
            labels = None
            loss = None
            outputs = None

        if epoch % args.interval == 0 or epoch == args.max_epoch-1:
            lr_scheduler(optimizer=optimizer, epoch=epoch, max_epoch=args.max_epoch, gamma=args.lr_gamma)

        auT.eval()
        auC.eval()
        ttl_corr = 0
        ttl_size = 0
        print('Validating...')
        for features, labels in tqdm(test_loader):
            features, labels = features.to(args.device), labels.to(args.device)
            with torch.no_grad():
                outputs = auC(auT(features))
                _, preds = torch.max(outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
        curr_accu = ttl_corr / ttl_size * 100.
        wandb_run.log({'Train/Accuracy': ttl_train_corr/ttl_train_num*100.}, step=epoch)
        wandb_run.log({'Val/Accuracy': curr_accu}, step=epoch)
        wandb_run.log({'Train/classifier_loss': ttl_train_loss/ttl_train_num}, step=epoch)
        wandb_run.log({'Train/learning_rate': optimizer.param_groups[0]['lr']}, step=epoch, commit=True)
        print(f'train accu: {ttl_train_corr/ttl_train_num*100.:.2f}%, val accu: {curr_accu:.2f}%')
        if curr_accu > best_accuracy:
            best_accuracy = curr_accu
            best_auT = auT.state_dict()
            best_auC = auC.state_dict()
            torch.save(best_auT, os.path.join(args.full_output_path, f'{args.output_weight_prefix}_best_auT.pt'))
            torch.save(best_auC, os.path.join(args.full_output_path, f'{args.output_weight_prefix}_best_auC.pt'))
        features = None
        labels = None
        outputs = None