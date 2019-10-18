import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

from loss import BCE_loss
import kornia


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

    Xs, Xt, Ys, Yt, labels = D
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    labels = labels.float().to(device)
    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

    preds, predt, pred_det = model(Xs, Xt)

    loss_p = BCE_loss(predt, Yt, with_logits=True)
    loss_q = BCE_loss(preds, Ys, with_logits=True)

    if not args.wo_det:
        loss_det = F.binary_cross_entropy_with_logits(
            pred_det.squeeze(), labels.squeeze()
        )
    else:
        loss_det = 0

    loss = loss_p + loss_q + args.gamma * loss_det

    if args.bw:
        gauss = kornia.filters.GaussianBlur2d((7, 7), (5, 5))
        Ys_edge = (kornia.sobel(gauss(Ys)) > 0.01).float()
        Yt_edge = (kornia.sobel(gauss(Yt)) > 0.01).float()

        loss_edge_s = torch.sum(-Ys_edge * F.logsigmoid(preds)) / torch.sum(
            Ys_edge
        )
        loss_edge_t = torch.sum(-Yt_edge * F.logsigmoid(predt)) / torch.sum(
            Yt_edge
        )
        loss_edge = loss_edge_s + loss_edge_t
        loss += args.gamma2 * loss_edge

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()

    _str = (
        f"{iteration:5d}: f(probe+donor+det): {tval(loss_p):.4f} + "
        + f"{tval(loss_q):.4f} + {tval(loss_det):.4f}"
    )

    if args.bw:
        _str += f" + {tval(loss_edge):.4f}"

    print(_str)

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val


def train_dmac(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    Xs, Xt, Ys, Yt, labels = D
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    labels = labels.float().to(device)
    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

    preds, predt, pred_det = model(Xs, Xt)

    if args.model == "dmac":
        criterion = nn.NLLLoss().cuda(device)

        log_op = F.log_softmax(predt, dim=1)
        log_oq = F.log_softmax(preds, dim=1)

        Yq = Ys.squeeze(1).long()
        Yp = Yt.squeeze(1).long()

        loss_p = criterion(log_op, Yp)
        loss_q = criterion(log_oq, Yq)

        loss = loss_p + loss_q
    elif args.model == "dmvn":
        loss_p = BCE_loss(predt, Yt, with_logits=True)
        loss_q = BCE_loss(preds, Ys, with_logits=True)
        loss = loss_p + loss_q

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(
        f"{iteration:5d}: f(probe+donor+det): {tval(loss_p):.4f} + "
        + f"{tval(loss_q):.4f}"
    )

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val
