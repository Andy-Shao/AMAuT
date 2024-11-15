import torch.nn as nn
import torch
from ml_collections import ConfigDict

from .aut_model import AudioTransformer

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

class AuT(nn.Module):
    def __init__(self, config: ConfigDict, mel_spec_shape=[224,224], in_channels=3) -> None:
        super(AuT, self).__init__()
        self.feature_extractor = AudioTransformer(config, mel_spec_shape=mel_spec_shape, in_channels=in_channels)
        self.out_features = self.feature_extractor.output_format.fc.out_features

    def forward(self, x):
        feat = self.feature_extractor(x)
        return feat

class AudioClassifier(nn.Module):
    def __init__(self, feature_dim: int, class_num: int, bottleneck_dim=256, type="ori", cls_type="linear", ) -> None:
        super(AudioClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=.5)
        self.bottleneck = nn.Linear(in_features=feature_dim, out_features=bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

        self.cls_type = cls_type
        if cls_type == 'wn':
            self.fc = nn.utils.parametrizations.weight_norm(module=nn.Linear(in_features=bottleneck_dim, out_features=class_num), name='weight')
            self.fc.apply(init_weights)
        else: 
            self.fc = nn.Linear(in_features=bottleneck_dim, out_features=class_num)
            self.fc.apply(init_weights)
    
    def forward(self, x: torch.Tensor):
        x = self.bottleneck(x)
        if self.type == 'bn':
            x = self.bn(x)

        x = self.fc(x)
        return x