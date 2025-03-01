import argparse
import os
import numpy as np
import random 
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torchaudio import transforms as a_transforms
from torch.utils.data import DataLoader

from lib.toolkit import print_argparse
from lib.datasets import dataset_tag, Dataset_Idx
from lib.wavUtils import Components, AudioPadding, AmplitudeToDB, MelSpectrogramPadding, FrequenceTokenTransformer
from AuT.speech_commands.train import build_dataest, build_model, includeAutoencoder, op_copy, lr_scheduler
from AuT.lib.model import AudioClassifier, AudioDecoder, AudioTransform
from AuT.lib.plr import plr
from AuT.lib.loss import SoftCrossEntropyLoss

def inference(auT:AudioTransform, auC:AudioClassifier, data_loader:DataLoader, args:argparse.Namespace) -> float:
    import torch.nn.functional as F
    auT.eval()
    auC.eval()
    ttl_size = 0.
    ttl_corr = 0.
    with torch.no_grad():
        for idx, (inputs, labels) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs, features = auC(auT(inputs))
            _, preds = torch.max(input=outputs.detach(), dim=1)
            ttl_corr += (preds == labels).sum().cpu().item()
            ttl_size += labels.shape[0]
            if idx == 0:
                all_feature = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
            else:
                all_feature = torch.cat([all_feature, features.float().cpu()], dim=0)
                all_output = torch.cat([all_output, outputs.float().cpu()], dim=0)
                all_label = torch.cat([all_label, labels.float().cpu()], dim=0)
        inputs = None
        features = None
        outputs = None
        preds = None
    ############################### inference ################################

    # all_output = nn.Softmax(dim=1)(all_output)
    all_output = F.softmax(all_output, dim=1)
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

    dd = all_feature @ initc[labelset].T # <g_t, initc>
    dd = np.exp(dd) # amplify difference
    pred_label = dd.argmax(axis=1) # predicted class based on the minimum distance
    pred_label = labelset[pred_label] # this will be the actual class

    for round in range(1): # calculate initc and pseduo label multi-times
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_feature)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = all_feature @ initc[labelset].T
        dd = np.exp(dd)
        pred_label = dd.argmax(axis=1)
        pred_label = labelset[pred_label]

    # pseduo-label accuracy
    pl_acc = np.sum(pred_label == all_label.float().numpy()) / len(all_feature) * 100.

    dd = F.softmax(torch.from_numpy(dd), dim=1)
    return pred_label, all_output.cpu().numpy(), dd.numpy().astype(np.float32), mean_all_output, ttl_corr / ttl_size * 100.0, pl_acc

def load_model(args:argparse.Namespace, auT:AudioTransform, auD:AudioDecoder, auC:AudioClassifier):
    auT.load_state_dict(state_dict=torch.load(args.original_auT_weight_path))
    auC.load_state_dict(state_dict=torch.load(args.original_auC_weight_path))
    if includeAutoencoder(args):
        auD.load_state_dict(state_dict=torch.load(args.original_auD_weight_path))

def build_optimizer(args: argparse.Namespace, auT:AudioTransform, auC:AudioClassifier, auD:AudioDecoder) -> optim.Optimizer:
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

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, default='speech-commands', choices=['speech-commands', 'speech-commands_v2'])
    ap.add_argument('--dataset_root_path', type=str)
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
    ap.add_argument('--original_auD_weight_path', type=str)

    ap.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--plr', help='Pseudo-label refinement', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.9)
    ap.add_argument('--cls_par', type=float, default=1.0, help='lambda 2 | Pseudo-label loss capable')
    ap.add_argument('--cls_mode', type=str, default='soft_ce', choices=['logsoft_ce', 'soft_ce', 'logsoft_nll'])

    args = ap.parse_args()
    if args.dataset == 'speech-commands':
        args.class_num = 30
        args.dataset_type = 'all'
    elif args.dataset == 'speech-commands_v2':
        args.class_num = 35
    else:
        raise Exception('No support!')
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
        project='AC-PT (AuT)', name=f'{args.arch}-{dataset_tag(args.dataset)}', 
        mode='online' if args.wandb else 'disabled', config=args, tags=['Audio Classification', args.dataset, 'AuT'])
    
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
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    weak_dataset = Dataset_Idx(dataset=test_dataset)
    weak_loader = DataLoader(
        dataset=weak_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )

    auTmodel, clsmodel, auDecoder = build_model(args=args)
    load_model(args=args, auT=auTmodel, auC=clsmodel, auD=auDecoder)
    decoder_loss_fn = nn.MSELoss(reduction='mean').to(device=args.device)
    optimizer = build_optimizer(args=args, auT=auTmodel, auC=clsmodel, auD=auDecoder)

    print('Pre-predicating...')
    mem_label, soft_output, dd, mean_all_output, accu, pl_acc = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
    if args.plr:
        prev_mem_label = mem_label
        mem_label = dd
    else:
        mem_label = dd
    mem_label = torch.from_numpy(mem_label).to(args.device)
    dd = torch.from_numpy(dd).to(args.device)
    mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
    print(f'accuracy is: {accu:.2f}%, sample size is: {len(test_dataset)}, pseudo-label accuracy is: {pl_acc:.2f}%')
    for epoch in range(args.max_epoch):
        print(f"Epoch {epoch+1}/{args.max_epoch}")
        auTmodel.train()
        clsmodel.train()

        for inputs, _, idxes in tqdm(weak_loader):
            inputs = inputs.to(args.device)
            batch_size = inputs.shape[0]
            optimizer.zero_grad()
            outputs, _ = clsmodel(auTmodel(inputs))

            # Pseudo-label cross-entropy loss
            if args.cls_par > 0:
                with torch.no_grad():
                    pred = mem_label[idxes]
                    pred = pred.detach()
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
                classifier_loss = args.cls_par*classifier_loss
            else:
                classifier_loss = torch.tensor(.0).cuda()

            loss = classifier_loss
            loss.backward()
            optimizer.step()

        if epoch % args.interval == 0:
            lr_scheduler(optimizer=optimizer, epoch=epoch, lr_cardinality=args.lr_cardinality)
                
        mem_label, soft_output, dd, mean_all_output, accu, pl_acc = inference(auT=auTmodel, auC=clsmodel, data_loader=test_loader, args=args)
        if args.plr:
            mem_label = plr(prev_mem_label, mem_label, dd, args.class_num, alpha = args.alpha)
            prev_mem_label = mem_label.argmax(axis=1).astype(int)
        else:
            mem_label = dd
        mem_label = torch.from_numpy(mem_label).to(args.device)
        dd = torch.from_numpy(dd).to(args.device)
        mean_all_output = torch.from_numpy(mean_all_output).to(args.device)
        print(f'accuracy is: {accu:.2f}%, sample size is: {len(test_dataset)}, pseudo-label accuracy is: {pl_acc:.2f}%')