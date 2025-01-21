import torch 
import torch.nn as nn
import torch.nn.functional as F

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