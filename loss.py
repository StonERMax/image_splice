from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss(x, t, gamma=2, with_weight=False):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,1, ...].
        y: (tensor) sized [N, 1, ...].
    Return:
        (tensor) focal loss.
    '''

    x = x.view(-1)
    t = t.view(-1)

    if with_weight:
        wgt = torch.sum(t) / (t.shape[0])
    else:
        wgt = 0.5

    p = torch.sigmoid(x)
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = (1-wgt)*t + wgt*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w.detach())


def cross_entropy_loss(y, labels, with_logits=True):
    """
        mode in "forge", "source"
    """
    # b, _, h, w = y.shape
    labels = labels.squeeze(1)

    if with_logits:
        loss = - labels * F.log_softmax(y, dim=-3)
    else:
        loss = - labels * torch.log(y)

    loss = loss.mean()
    return loss


def dice_loss(y, labels):
    smooth = 1
    y = torch.sigmoid(y.view(-1))
    lab = labels.view(-1)

    numer = 2 * (y * lab).sum()
    den = y.sum() + lab.sum()

    return 1 - (numer + smooth) / (den + smooth)

def BCE_loss(y, labels, with_weight=False, with_logits=True):
    y = y.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    if not with_weight:
        wgt = torch.ones_like(labels)
    else:
        ind_pos = (labels > 0.9)
        ind_neg = (labels < 0.1)
        ind_total = ind_pos.sum()+ind_neg.sum()
        _w = torch.max(
            (ind_pos.sum() / (ind_total + 1e-8)).float(),
            torch.tensor(0.05).to(y.device)
        )
        wgt = labels * (1 - _w) + _w * (1 - labels)

    if with_logits:
        bce_loss = F.binary_cross_entropy_with_logits(
            y, labels, wgt, reduction='none')
    else:
        bce_loss = F.binary_cross_entropy(y, labels, wgt, reduction='none')

    bce_loss = bce_loss.sum() / (wgt.sum() + 1e-8)

    return bce_loss
