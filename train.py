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

        loss_edge_s = torch.sum(-Ys_edge * F.logsigmoid(preds)) / torch.sum(Ys_edge)
        loss_edge_t = torch.sum(-Yt_edge * F.logsigmoid(predt)) / torch.sum(Yt_edge)
        loss_edge = loss_edge_s + loss_edge_t
        loss += args.gamma2 * loss_edge

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()

    _str = (
        f"{iteration:5d}: f(probe+donor+det): {tval(loss_p):.4f} + "
        + f"{tval(loss_q):.4f} + {tval(loss_det):.4f} = {tval(loss):.4f}"
    )

    if args.bw:
        _str += f" + {tval(loss_edge):.4f}"

    print(_str)

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val


def train_det(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()
    if args.eval_bn:
        module.set_bn_to_eval()

    X, Y, labels = D
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    labels = labels.float().to(device)
    X = X.to(device)
    Y = Y.to(device)

    pred_det, pred_seg = model(X)
    loss_det = F.binary_cross_entropy_with_logits(pred_det.squeeze(), labels.squeeze())
    loss_seg = F.binary_cross_entropy_with_logits(pred_seg, Y)

    loss = loss_seg + args.gamma * loss_det

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()

    print(
        f"{iteration}: loss: {loss_seg.data.cpu().numpy():.4f} + {loss_det.data.cpu().numpy():.4f}"
    )

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val


def train_temporal(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()
    if args.eval_bn:
        module.set_bn_to_eval()
    

    Xs, Xt, Ys, Yt, labels = D
    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)
    labels = labels.to(device)

    preds, predt, pred_det = model(Xs, Xt)

    mask_label = (labels > 0.5).clone()
    loss_t = BCE_loss(predt[mask_label], Yt[mask_label], with_logits=True)
    loss_s = BCE_loss(preds[mask_label], Ys[mask_label], with_logits=True)

    # detection whether video clips are copy move
    pred_det = torch.sigmoid(pred_det)
    pos_mean = torch.sum(labels * pred_det) / (torch.sum(labels)+1e-8)
    neg_mean = torch.sum((1-labels) * pred_det) / (torch.sum(1-labels)+1e-8)
    # loss_det = F.binary_cross_entropy_with_logits(pred_det, labels)
    loss_det = torch.max(neg_mean - pos_mean + args.beta, torch.tensor(0.).to(device))

    loss = loss_s + loss_t + args.gamma * loss_det

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()

    _str = (
        f"{iteration:5d}: f(probe+donor+det): {tval(loss_s):.4f} + "
        + f"{tval(loss_t):.4f} + {tval(loss_det):.4f}"
    )

    print(_str)
    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val


def train_dmac(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()
    if args.eval_bn:
        module.set_bn_to_eval()

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
