import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from loss import BCE_loss


def tval(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()


def train(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    Xp, Xq, Yp, Yq, labels = D
    labels = torch.from_numpy(np.array(labels, dtype=np.float32)).to(device)
    Xp, Xq, Yp, Yq = Xp.to(device), Xq.to(device), Yp.to(device), Yq.to(device)

    predp, predq, pred_det = model(Xp, Xq)

    loss_p = BCE_loss(predp, Yp, with_logits=True)
    loss_q = BCE_loss(predq, Yq, with_logits=True)

    loss_det = F.binary_cross_entropy_with_logits(pred_det.squeeze(), labels.squeeze())

    loss = loss_p + loss_q + args.gamma * loss_det

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(f"{iteration:5d}: f: {tval(loss_p):.4f} + {tval(loss_q):.4f} ")

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val
