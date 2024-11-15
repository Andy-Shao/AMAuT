import torch 
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, reduction=True, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon 
        self.reduction = reduction 
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.use_gpu = use_gpu
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1) # cross-entropy loss
        if self.reduction:
            return loss.mean()
        else:
            return loss

def Entropy(input_: torch.Tensor, epsilon= 1e-5):
    # epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def SoftCrossEntropyLoss(logit: torch.Tensor, soft_pseudo_label: torch.Tensor) -> torch.Tensor:   # Checked and is correct
    """Pseudo-label cross-entropy loss uses this loss function"""
    percentage = F.log_softmax(logit, dim=1)
    # print(f'left shape: {soft_pseudo_label.shape}, right shape: {percentage.shape}')
    return -(soft_pseudo_label * percentage).sum(dim=1)

def soft_CE(softout: torch.Tensor, soft_label: torch.Tensor, epsilon = 1e-5) -> torch.Tensor:
    """(Consist loss -> soft cross-entropy loss) uses this loss function"""
    # epsilon = 1e-5
    loss = -soft_label * torch.log(softout + epsilon)
    total_loss = torch.sum(loss, dim=1)
    return total_loss