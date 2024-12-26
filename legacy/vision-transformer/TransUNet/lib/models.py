import numpy as np

import torch.nn as nn

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def init_weights(m: nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1., .02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class TransUNet(nn.Module):
    def __init__(self):
        super(TransUNet, self).__init__()
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = 100
        config_vit.n_skip = 3
        config_vit.patches.grid = (int(224 / 16), int(224 / 16))
        config_vit.pretrained_path = './model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
        self.feature_extractor = ViT_seg(config_vit, img_size=[224, 224], num_classes=config_vit.n_classes)
        self.feature_extractor.load_from(weights=np.load(config_vit.pretrained_path))
        self.in_features = 2048
    
    def forward(self, x):
        _, feat = self.feature_extractor(x)
        return feat