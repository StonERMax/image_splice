import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from loss import *


def tval(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()


def train(X, Y, model, optimizer, args, iteration, device,
          logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    X, Y = X.to(device), Y.to(device)

    Y_bw = Y[:, [-1]]
    if args.out_channel == 1:
        Y = Y[:, [0]]
    else:
        Y = Y[:, :3]
    # Y = Y[:, [0]]

    y_det = torch.zeros((Y.shape[0], 1), dtype=torch.float32, device=device)
    for i in range(Y.shape[0]):
        if torch.any(Y[i, 0] > 0.5):
            y_det[i] = 1

    pred, pred_det = model(X)

    if args.out_channel == 3:
        loss1 = cross_entropy_loss(pred, Y)
    else:
        loss1 = BCE_loss(pred, Y, with_logits=True)

    loss_det = F.binary_cross_entropy_with_logits(pred_det, y_det)

    loss_bw = torch.sum(Y_bw * torch.sigmoid(pred)) / Y_bw.shape[0] #/ (torch.sum(Y_bw) + 1e-8) 

    loss = loss1  + args.gamma * loss_det +  loss_bw * args.gamma2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(f"{iteration:5d}: f: {loss1.data.cpu().numpy():.4f} + {tval(loss_det):.4f} "
             f"loss     {loss_val:.4f}")

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val