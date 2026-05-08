import torch.nn as nn
import torch.nn.functional as F


class SeparationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return F.l1_loss(pred, target)