import argparse
import os

import torch

from lib.models import TransUNet
from lib.toolkits import print_model

if __name__ == '__main__':
    ap = argparse.ArgumentParser('TransUnet')
    ap.add_argument('--output_path', type=str, default='./output')

    args = ap.parse_args()

    try:
        if not os.path.exists(args.output_path): os.makedirs(args.output_path)
    except:
        pass
    ##############################################################

    transUNet = TransUNet()
    print_model(model=transUNet, output_path=os.path.join(args.output_path, 'TransUNet.txt'))

    features = torch.ones(32, 3, 224, 224)
    outputs = transUNet(features)
    print('outputs shape is:', outputs.shape) # 32, 100, 224, 224