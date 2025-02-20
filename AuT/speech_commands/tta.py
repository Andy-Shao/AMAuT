import argparse
import os
import numpy as np
import random 
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.datasets import dataset_tag, TransferDataset, Dataset_Idx, load_from
from lib.wavUtils import Components, MelSpectrogramPadding, AudioPadding, AmplitudeToDB, FrequenceTokenTransformer, time_shift
from lib.wavUtils import RandomPitchShift, RandomSpeed
from AuT.speech_commands.pre_train import build_dataest, build_model, lr_scheduler, includeAutoencoder, op_copy
from AuT.speech_commands.tta_analysis import load_model
from AuT.lib.loss import soft_CE, SoftCrossEntropyLoss
from AuT.lib.model import AudioTransform, AudioClassifier, AudioDecoder
from AuT.lib.plr import plr

def build_optimizer(args: argparse.Namespace, auT:nn.Module, auC:nn.Module, auD:AudioDecoder) -> optim.Optimizer:
    param_group = []
    learning_rate = args.lr
    for k, v in auT.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate*.1}]
    for k, v in auC.named_parameters():
        param_group += [{'params':v, 'lr':learning_rate}]
    if includeAutoencoder(args):
        for k, v in auD.named_parameters():
            param_group += [{'params':v, 'lr':learning_rate}]
    optimizer = optim.SGD(params=param_group)
    optimizer = op_copy(optimizer)
    return optimizer

def inference(auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace):
    from scipy.spatial.distance import cdist
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    for idx, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs, labels = inputs.to(args.device), labels.to(args.device)

        with torch.no_grad():
            outputs, features = auC(auT(inputs))
            _, preds = torch.max(input=outputs.detach(), dim=1)
            if idx == 0:
                all_output = outputs.float().cpu()
                all_feature = features.float().cpu()
            else:
                all_output = torch.cat([all_output, outputs.float().cpu()], dim=0)
                all_feature = torch.cat([all_feature, features.float().cpu()], dim=0)
        ttl_corr += (preds == labels).sum().cpu().item()
        ttl_size += labels.shape[0]
    all_output = nn.Softmax(dim=1)(all_output)
    mean_all_output = torch.mean(all_output, dim=0).numpy()
    _, predict = torch.max(all_output, dim=1)

    # find centroid per class
    if args.distance == 'cosine': 
        ######### Not Clear (looks like feature normalization though)#######
        all_feature = torch.cat((all_feature, torch.ones(all_feature.size(0), 1)), dim=1)
        all_feature = (all_feature.t() / torch.norm(all_feature, p=2, dim=1)).t() # here is L2 norm
    ### all_fea: extractor feature [bs,N]. all_feature is g_t in paper
    all_feature = all_feature.float().cpu().numpy()
    K = all_output.size(1) # number of classes
    aff = all_output.float().cpu().numpy() ### aff: softmax output [bs,c]

    # got the initial normalized centroid (k*(d+1))
    initc = aff.transpose().dot(all_feature)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    cls_count = np.eye(K)[predict].sum(axis=0) # total number of prediction per class
    labelset = np.where(cls_count >= args.threshold) ### index of classes for which same sampeled have been detected # returns tuple
    labelset = labelset[0] # index of classes for which samples per class greater than threshold
    # labelset == [0, 1, 2, ..., 29]

    # dd is the data distance between data and central point.
    # dd = dict(all_feature, initc[labelset], args.distance)
    dd = all_feature @ initc[labelset].T # <g_t, initc>
    dd = np.exp(dd) # amplify difference
    pred_label = dd.argmax(axis=1) # predicted class based on the minimum distance
    pred_label = labelset[pred_label] # this will be the actual class

    for round in range(args.initc_num): # calculate initc and pseduo label multi-times
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_feature)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        # dd = dict(all_feature, initc[labelset], args.distance)
        dd = all_feature @ initc[labelset].T
        dd = np.exp(dd)
        pred_label = dd.argmax(axis=1)
        pred_label = labelset[pred_label]
    dd = nn.functional.softmax(torch.from_numpy(dd), dim=1)

    return pred_label, all_output.cpu().numpy(), dd.numpy().astype(np.float32), mean_all_output, ttl_corr / ttl_size * 100.0

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands_v2'])
    ap.add_argument('--dataset_root_path', type=str)
    ap.add_argument('--corrupted_data_root_path', type=str)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--output_path', type=str, default='./result')

    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--seed', type=int, default=2025, help='random seed')
    ap.add_argument('--model_topology', action='store_true')

    ap.add_argument('--max_epoch', type=int, default=200, help='max epoch')
    ap.add_argument('--interval', type=int, default=1, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--lr_cardinality', type=int, default=40)
    ap.add_argument('--lr_dec', type=float, default=1.25)
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='CT', choices=['CT', 'CTA'])
    ap.add_argument('--arch_level', type=str, default='base')

    ap.add_argument('--original_auT_weight_path', type=str)
    ap.add_argument('--original_auC_weight_path', type=str)

    ap.add_argument('--fbnm_par', type=float, default=4.0, help='lambda 1')
    ap.add_argument('--cls_par', type=float, default=0.2, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--const_par', type=float, default=0.2, help='lambda 3')

    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands_v2':
        args.class_num = 35
    else:
        raise Exception('No support!')
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cuda'
    assert torch.cuda.is_available(), 'No support cpu mode'
    args.full_output_path = os.path.join(args.output_path, args.dataset, 'AuT', 'TTA')
    try:
        os.makedirs(args.full_output_path)
    except:
        pass
    args.meta_file_name = 'speech_commands_meta.csv'
    torch.backends.cudnn.benchmark = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print_argparse(args)
    ##########################################

    wandb_run = wandb.init(
        project='AC-TTA (AuT)', name=f'{args.arch}-{dataset_tag(args.dataset)}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
    sample_rate=16000
    args.n_mels=80
    n_fft=1024
    win_length=400
    hop_length=155
    mel_scale='slaney'
    target_length=104
    # test_set = build_dataest(args=args, mode='test', tsf=None)
    test_set = load_from(root_path=args.corrupted_data_root_path, index_file_name=args.meta_file_name)
    test_loader = DataLoader(
        dataset=TransferDataset(
            dataset=test_set, 
            data_tf=Components(transforms=[
                AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
                a_transforms.MelSpectrogram(
                    sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                    mel_scale=mel_scale
                ),
                AmplitudeToDB(top_db=80., max_out=2.),
                MelSpectrogramPadding(target_length=target_length),
                FrequenceTokenTransformer()
            ])),
        batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    tf_array = Components(transforms=[
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=False),
        # time_shift(shift_limit=.17, is_random=True, is_bidirection=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    weak_set = TransferDataset(dataset=test_set, data_tf=tf_array, device='cpu', keep_cpu=True)
    weak_set = Dataset_Idx(dataset=weak_set)
    weak_loader = DataLoader(dataset=weak_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)

    tf_array = Components(transforms=[
        # RandomSpeed(start_fq=.9, end_fq=1.11, sample_rate=sample_rate, max_length=sample_rate, step=.01, max_it=5),
        RandomPitchShift(step_rang=[0,1,2,3,4], sample_rate=sample_rate),
        AudioPadding(max_length=sample_rate, sample_rate=sample_rate, random_shift=True),
        a_transforms.MelSpectrogram(
            sample_rate=sample_rate, n_mels=args.n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            mel_scale=mel_scale
        ),
        AmplitudeToDB(top_db=80., max_out=2.),
        MelSpectrogramPadding(target_length=target_length),
        FrequenceTokenTransformer()
    ])
    strong_set = TransferDataset(dataset=test_set, data_tf=tf_array, device=args.device, keep_cpu=False)

    auTmodel, clsmodel, _ = build_model(args=args)
    load_model(args=args, auC=clsmodel, auT=auTmodel, mode='original')
    optimizer = build_optimizer(args=args, auC=clsmodel, auT=auTmodel, auD=None)

    max_accu = 0.
    print('Pre-inferencing...')
    mem_label, soft_output, dd, mean_all_output, accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    print(f'Test accuracy: {accu:.2f}%, test set sample size is: {len(test_set)}')
    if args.plr:
        prev_mem_label = mem_label
        mem_label = dd
    else:
        mem_label = dd

    mem_label = torch.from_numpy(mem_label).to(args.device)
    dd = torch.from_numpy(dd).to(args.device)
    mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")

        print('Adaptating...')
        auTmodel.train()
        clsmodel.train()
        adatpting_time = len(weak_loader)
        ttl_loss = 0.
        ttl_fbnm_loss = 0.
        ttl_cst_loss = 0.
        for weak_features, _, ids in tqdm(weak_loader):
            batch_size = weak_features.shape[0]
            weak_features = weak_features.to(args.device)
            strong_features = torch.cat([torch.unsqueeze(strong_set[i][0], dim=0) for i in ids], dim=0)
            features = torch.cat([weak_features, strong_features], dim=0)

            optimizer.zero_grad()
            outputs = clsmodel(auTmodel(features))

            # Pseudo-label cross-entropy loss
            if args.cls_par > 0:
                with torch.no_grad():
                    pred = mem_label[ids]
                if args.cls_mode == 'logsoft_ce':
                    classifier_loss = SoftCrossEntropyLoss(outputs[0:batch_size], pred)
                    classifier_loss = torch.mean(classifier_loss)
                elif args.cls_mode == 'soft_ce':
                    softmax_output = nn.Softmax(dim=1)(outputs[0:batch_size])
                    classifier_loss = nn.CrossEntropyLoss()(softmax_output, pred)
                elif args.cls_mode == 'logsoft_nll':
                    softmax_output = nn.LogSoftmax(dim=1)(outputs[0:batch_size])
                    _, pred = torch.max(pred, dim=1)
                    classifier_loss = nn.NLLLoss(reduction='mean')(softmax_output, pred)
                classifier_loss = args.cls_par * classifier_loss
            else:
                classifier_loss = torch.tensor(.0).cuda()

            # fbnm -> Nuclear-norm Maximization loss
            if args.fbnm_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                list_svd,_ = torch.sort(torch.sqrt(torch.sum(torch.pow(softmax_output,2),dim=0)), descending=True)
                fbnm_loss = - torch.mean(list_svd[:min(softmax_output.shape[0],softmax_output.shape[1])])
                fbnm_loss = args.fbnm_par*fbnm_loss
            else:
                fbnm_loss = torch.tensor(.0).cuda()

            # Consist loss -> soft cross-entropy loss
            if args.const_par > 0:
                softmax_output = nn.Softmax(dim=1)(outputs)
                expectation_ratio = mean_all_output/torch.mean(softmax_output[0:batch_size],dim=0)
                with torch.no_grad():
                    soft_label_norm = torch.norm(softmax_output[0:batch_size]*expectation_ratio,dim=1,keepdim=True) #Frobenius norm
                    soft_label = (softmax_output[0:batch_size]*expectation_ratio)/soft_label_norm
                consistency_loss = args.const_par*torch.mean(soft_CE(softmax_output[batch_size:],soft_label))
            else:
                consistency_loss = torch.tensor(.0).cuda()

            loss = classifier_loss + fbnm_loss + consistency_loss
            loss.backward()
            optimizer.step()
            ttl_loss += loss
            ttl_fbnm_loss += fbnm_loss.cpu().item()
            ttl_cst_loss += consistency_loss.cpu().item()

        learning_rate = optimizer.param_groups[0]['lr']
        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)

        print('Inferencing...')
        mem_label, soft_output, dd, mean_all_output, accu = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
        if args.plr:
            mem_label = plr(prev_mem_label, mem_label, dd, args.class_num, alpha = args.alpha)
            prev_mem_label = mem_label.argmax(axis=1).astype(int)
        else:
            mem_label = dd
        mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
        mem_label = torch.from_numpy(mem_label).to(args.device)
        dd = torch.from_numpy(dd).to(args.device)
        print(f'Test accuracy: {accu:.2f}%, test set sample size is: {len(test_set)}')
        wandb.log({
            "LOSS/total loss":ttl_loss / adatpting_time,
            "LOSS/consistency loss": ttl_cst_loss / adatpting_time,
            "LOSS/Nuclear-norm Maximization loss":ttl_fbnm_loss / adatpting_time,
            "Accuracy/classifier accuracy": accu
        }, step=epoch, commit=True)