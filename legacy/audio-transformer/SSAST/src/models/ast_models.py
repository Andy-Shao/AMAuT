import sys
BASE_PATH='/home/andyshao'
sys.path.append(f"{BASE_PATH}/data/sls/scratch/aed-trans/src/models/")
sys.path.append(f"{BASE_PATH}/data/sls/scratch/aed-trans/src/")
from timm.models.layers import to_2tuple

import torch.nn as nn

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x