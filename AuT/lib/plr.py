import numpy as np
import sys

import torch 
import torch.nn.functional as F

np.set_printoptions(threshold=sys.maxsize)

def plr(prev_pseudo_labels, pseudo_labels, soft_output, class_num:int,  alpha=0.9) -> np.ndarray:
    """Pseudo Label Refinement"""
    consensus = torch.zeros((class_num, class_num))
    for i in range(class_num):
        prev_i_label_index = np.where(prev_pseudo_labels == i)[0]
        for j in range(class_num):
            j_label_index = np.where(pseudo_labels == j)[0]
            intersection = np.intersect1d(prev_i_label_index, j_label_index)
            union = np.union1d(prev_i_label_index, j_label_index)
            consensus[i][j] = len(intersection) / (len(union)+1e-8)
    
    consensus = F.softmax(consensus, dim=1)
    prev_pseudo_labels = torch.unsqueeze(torch.from_numpy(prev_pseudo_labels), dim=1)
    pseudo_labels = torch.unsqueeze(torch.from_numpy(pseudo_labels), dim=1)

    prop_prev_pl = torch.matmul(torch.from_numpy(soft_output), consensus)
    refined = torch.add(alpha*pseudo_labels, (1-alpha)*prop_prev_pl)
    refined = F.softmax(refined, dim=1)
    return refined.numpy()