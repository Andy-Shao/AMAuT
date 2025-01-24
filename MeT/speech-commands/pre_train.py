import argparse

from torch import nn
from torch.utils.data import Dataset

from lib.toolkit import print_argparse

class TwoTFDataset(Dataset):
    def __init__(self, dataset:Dataset, tf1:nn.Module=None, tf2:nn.Module=None):
        super(TwoTFDataset, self).__init__()
        self.dataset = dataset
        self.tf1, self.tf2 = tf1, tf2

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item, label = self.dataset[index]
        x1, x2 = item, item
        if self.tf1 is not None:
            x1 = self.tf1(x1)
        if self.tf2 is not None:
            x2 = self.tf2(x2)
        return x1, x2, label

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
    ap.add_argument('--interval_num', type=int, default=50, help='interval number')
    ap.add_argument('--batch_size', type=int, default=64, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    ap.add_argument('--smooth', type=float, default=.1)
    ap.add_argument('--early_stop', type=int, default=-1)
    ap.add_argument('--arch', type=str, default='MeT', choices=['MeT'])
    ap.add_argument('--embed_size', type=int, default=768, choices=[768, 1024])

    print_argparse(args)
    #########################################