import torch.nn.functional as F
import torch.nn as nn


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    loss = nn.CrossEntropyLoss()
    return loss(output, target)
