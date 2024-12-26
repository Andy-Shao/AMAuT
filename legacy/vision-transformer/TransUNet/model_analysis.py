import argparse
import os

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