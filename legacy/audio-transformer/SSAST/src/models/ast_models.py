import sys
BASE_PATH='/home/andyshao'
PROJECT_PATH=BASE_PATH + '/Audio-Transform/legacy/audio-transformer/SSAST'
sys.path.append(f"{BASE_PATH}/data/sls/scratch/aed-trans/src/models/")
sys.path.append(f"{BASE_PATH}/data/sls/scratch/aed-trans/src/")
from timm.models.layers import to_2tuple, trunc_normal_
import numpy as np
import timm
from typing import Optional, Callable, Tuple, Union
import os

import torch
import torch.nn as nn

from src.utils.toolkits import store_model_structure_to_txt, print_attributes

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

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (i // 2) / d_hid) for i in range(d_hid)]
    
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class ASTModel(nn.Module):
    def __init__(self, label_dim=527,
                 fshape=128, tshape=2, fstride=128, tstride=2,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 pretrain_stage=True, load_pretrained_mdl_path=None):
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if pretrain_stage == True:
            if load_pretrained_mdl_path != None:
                raise ValueError('Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError('fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')
            
            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False) # image_size is 384
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

        self.original_num_patches = self.v.patch_embed.num_patches #576
        self.oringal_hw = int(self.original_num_patches ** 0.5) #24
        self.original_embedding_dim = self.v.pos_embed.shape[2] #768

        # SSL Pretraining Code
        self.softmax = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        self.fshape, self.tshape = fshape, tshape
        self.fstride, self.tstride = fstride, tstride
        self.input_fdim, self.input_tdim = input_fdim, input_tdim
        # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
        self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False), nn.Parameter(torch.tensor(input_tdim), requires_grad=False)

        # masked patch classification (discriminative objective) layer
        # we use two layers for pretext task, but using a single layer has similar performance.
        # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
        # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
        self.cpredlayer = nn.Sequential(
            nn.Linear(self.original_embedding_dim, self.original_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(self.original_embedding_dim, 256)
        )
        # masked patch reconstruction (generative objective) layer
        self.gpredlayer = nn.Sequential(
            nn.Linear(self.original_embedding_dim, self.original_embedding_dim), 
            nn.ReLU(), 
            nn.Linear(self.original_embedding_dim, 256)
        )
        self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

        # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
        self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
        self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

        # get the intermediate shape
        self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        num_patches = self.p_f_dim * self.p_t_dim
        self.num_patches = num_patches
        self.v.patch_embed.num_patches = num_patches
        print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
        print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
        print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
        print('pretraining number of patches={:d}'.format(num_patches))

        # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        self.v.patch_embed.proj = new_proj

        # use trainable positional embedding
        new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
        self.v.pos_embed = new_pos_embed
        trunc_normal_(self.v.pos_embed, std=.02)

        store_model_structure_to_txt(model=self.v, output_path=os.path.join(PROJECT_PATH, 'output/pretrain/ASTModel.txt'))
        print_attributes(obj=self.v, atts=[], output_path=os.path.join(PROJECT_PATH, 'output/pretrain/ASTModel_items.txt'))
        print_attributes(
            obj=self.v,
            atts=[
                'num_classes', 'num_features', 'embed_dim', 'cls_token', 'patch_embed', 'pos_embed',
                'pos_drop', 'dist_token', 'norm', 'head', 'head_dist', 'training'
            ],
            output_path=os.path.join(PROJECT_PATH, 'output/pretrain/ASTModel_attributes.txt')
        )
    
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        """get the shape of intermediate representation.
        """
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
            
if __name__ == '__main__':
    # this is an example of how to use the SSAST model

    # pretraining stage
    # suppose you have an unlabled dataset with avg length of 1024 frames (i.e., 10.24s)
    input_tdim = 1024
    # create a 16*16 patch based AST model for pretraining.
    # note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
    ast_mdl = ASTModel(
        fshape=16, tshape=16, fstride=16, tstride=16,
        input_fdim=128, input_tdim=input_tdim, model_size='base',
        pretrain_stage=True)
    # # alternatively, create a frame based AST model
    # ast_mdl = ASTModel(
    #              fshape=128, tshape=2, fstride=128, tstride=2,
    #              input_fdim=128, input_tdim=input_tdim, model_size='base',
    #              pretrain=True)