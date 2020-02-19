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
    if args.eval_bn:
        module.set_bn_to_eval()

    Xs, Xt, Ys, Yt, labels = D
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    labels = labels.float().to(device)
    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

    preds, predt = model(Xs, Xt)

    loss_p = BCE_loss(predt, Yt, with_logits=True)
    loss_q = BCE_loss(preds, Ys, with_logits=True)

    loss = loss_p + loss_q

    if args.bw:
        gauss = kornia.filters.GaussianBlur2d((7, 7), (5, 5))
        Ys_edge = (kornia.sobel(gauss(Ys)) > 0.01).float()
        Yt_edge = (kornia.sobel(gauss(Yt)) > 0.01).float()

        loss_edge_s = torch.sum(-Ys_edge * F.logsigmoid(preds)) / (torch.sum(Ys_edge)+1e-8)
        loss_edge_t = torch.sum(-Yt_edge * F.logsigmoid(predt)) / (torch.sum(Yt_edge)+1e-8)
        loss_edge = loss_edge_s + loss_edge_t
        loss += args.gamma2 * loss_edge

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()

    _str = (
        f"{iteration:5d}: f(probe+donor+det): {tval(loss_p):.4f} + "
        + f"{tval(loss_q):.4f} = {tval(loss):.4f}"
    )

    if args.bw:
        _str += f" + {tval(loss_edge):.4f}"

    print(_str)

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val




def train_cmfd(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    if args.freeze_bn:
        module.set_bn_to_eval()

    if args.mode == "both":

        Xs, Xt, Ys, Yt, labels = D
        Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

        preds, predt = model(Xs, Xt)

        loss_p = BCE_loss(predt, Yt, with_logits=True)
        loss_q = BCE_loss(preds, Ys, with_logits=True)
        loss = loss_p + loss_q
        if args.bw:
            gauss = kornia.filters.GaussianBlur2d((5, 5), (3, 3))
            Ys_edge = (kornia.sobel(gauss(Ys)) > 0.01).float()
            Yt_edge = (kornia.sobel(gauss(Yt)) > 0.01).float()

            loss_edge_s = torch.sum(-Ys_edge * F.logsigmoid(preds)) / (torch.sum(Ys_edge)+1e-8)
            loss_edge_t = torch.sum(-Yt_edge * F.logsigmoid(predt)) / (torch.sum(Yt_edge)+1e-8)
            loss_edge = loss_edge_s + loss_edge_t
            loss += args.gamma2 * loss_edge
        _str = (
            f"{iteration:5d}: f(probe+donor+det): {tval(loss_p):.4f} + "
            + f"{tval(loss_q):.4f}"
        )
    else:
        if len(D) == 2:
            X, Y = D
            Xs, Xt, Ys, Yt = X, X, Y, Y
            if Y.shape[1] == 3:
                Ys, Yt = Y[:, [1]], Y[:, [0]]
        else:
            Xs, Xt, Ys, Yt, labels = D
        Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)
        pred = model(Xs, Xt)
        if args.mode == "sim":
            Y = torch.max(Ys, Yt)
        else:
            Y = Yt
        loss = BCE_loss(pred, Y, with_logits=True)
        if args.bw:
            gauss = kornia.filters.GaussianBlur2d((5, 5), (3, 3))
            Y_edge = (kornia.sobel(gauss(Y)) > 0.01).float()
            loss_edge = torch.sum(-Y_edge * F.logsigmoid(pred)) / (torch.sum(Y_edge)+1e-8)
            loss += args.gamma2 * loss_edge
        _str = (
            f"{iteration:5d}: f: {tval(loss):.4f} "
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_val = loss.data.cpu().numpy()

    print(_str)

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val
