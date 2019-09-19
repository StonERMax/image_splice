from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage
import os
from tqdm import tqdm

# metric
from sklearn import metrics
import utils
from loss import BCE_loss


def add_torch_overlay(im, mask_s, mask_f=None, inv=True, clone=True):
    if clone:
        im = im.clone()
    if inv:
        mean = torch.tensor([0.485, 0.456, 0.406], device=im.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=im.device).view(1, 3, 1, 1)
        im = im * std + mean
    for i in range(im.shape[0]):
        if mask_f is None:
            im[i, 0] += mask_s[i, 0]
        else:
            im[i, 0] += mask_f[i, 0]
            im[i, 1] += mask_s[i, 0]
    im = im.clamp_max(1.0)
    im = im.clamp_min(0)
    return im


def rev_inv(im):
    mean = torch.tensor([0.485, 0.456, 0.406], device=im.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=im.device).view(3, 1, 1)
    im = im * std + mean
    return im


@torch.no_grad()
def test(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    metric = utils.Metric()
    # metric_im = utils.Metric_image()

    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data.load()):
        Xp, Xq, Yp, Yq, labels = ret
        Xp, Xq, Yp, Yq = Xp.to(device), Xq.to(device), Yp.to(device), Yq.to(device)

        predp, predq, pred_det = model(Xp, Xq)

        loss_p = BCE_loss(predp, Yp, with_logits=True)
        loss_q = BCE_loss(predq, Yq, with_logits=True)
        loss = loss_p + loss_q
        print(f"{i}:")

        def fnp(x):
            return x.data.cpu().numpy()

        predp = torch.sigmoid(predp)
        predq = torch.sigmoid(predq)

        metric.update([fnp(Yp), fnp(Yq)], [fnp(predp), fnp(predq)])

        if logger:
            logger.add_scalar("test_loss/total", loss, iteration)
        if num is not None and i >= num:
            break

    out = metric.final()

    return out, loss.data.cpu().numpy()
