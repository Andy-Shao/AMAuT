import torch 
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    def __init__(self, reduction='mean', c1=1e-8, c2=1e-8):
        super(SSIMLoss, self).__init__()
        assert reduction in ['mean', 'sum'], 'reduction value is incorrect'
        self.reduction = reduction
        self.c1 = c1
        self.c2 = c2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        mu1 = torch.mean(x1, dim=2)
        mu2 = torch.mean(x2, dim=2)
        var1 = torch.var(x1, dim=2)
        var2 = torch.var(x2, dim=2)
        cov = torch.sum((x1 - mu1.unsqueeze(2)) * (x2 - mu2.unsqueeze(2)), dim=2)/(x1.shape[2]-1)

        loss = (2 * mu1 * mu2 + self.c1) * (2 * cov + self.c2) / (torch.pow(mu1, exponent=2) + torch.pow(mu2, exponent=2) + self.c1) * (var1 + var2 + self.c2)
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()

class CosineSimilarityLoss(nn.Module):
    def __init__(self, reduction:str='mean', dim=2):
        super(CosineSimilarityLoss, self).__init__()
        assert reduction in ['mean', 'sum'], 'reduction is incorrect'
        self.cos_sim = nn.CosineSimilarity(dim=dim)
        self.reduction = reduction

    def forward(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        similarity = self.cos_sim(input, target)
        loss = 1 - similarity
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

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

def soft_CE(softout: torch.Tensor, soft_label: torch.Tensor, epsilon = 1e-5) -> torch.Tensor:
    """(Consist loss -> soft cross-entropy loss) uses this loss function"""
    # epsilon = 1e-5
    loss = -soft_label * torch.log(softout + epsilon)
    total_loss = torch.sum(loss, dim=1)
    return total_loss

def SoftCrossEntropyLoss(logit: torch.Tensor, soft_pseudo_label: torch.Tensor) -> torch.Tensor:   # Checked and is correct
    """Pseudo-label cross-entropy loss uses this loss function"""
    percentage = F.log_softmax(logit, dim=1)
    # print(f'left shape: {soft_pseudo_label.shape}, right shape: {percentage.shape}')
    return -(soft_pseudo_label * percentage).sum(dim=1)