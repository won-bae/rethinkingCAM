import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, gamma=1., temp=1., reduction='mean', eps=1e-6):
        super(_Loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.eps = eps

    def forward(self, preds, labels):
        preds = preds / self.temp
        if self.gamma >= 1.:
            loss = F.cross_entropy(
                preds, labels, weight=self.weight, reduction=self.reduction)
        else:
            log_prob = preds - torch.logsumexp(preds, dim=1, keepdim=True)
            log_prob = log_prob * self.gamma
            loss = F.nll_loss(
                log_prob, labels, weight=self.weight, reduction=self.reduction)

        losses = {'loss': loss}
        return losses

class CustomCriterion(_Loss):
    def __init__(self, criterion):
        super(_Loss, self).__init__()
        self.criterion = criterion

    def forward(self, preds, labels):
        losses = self.criterion(preds, labels)

        return losses

