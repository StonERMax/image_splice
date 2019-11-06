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


def to_np(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()


def rev_inv(im, to_numpy=True):
    mean = torch.tensor([0.485, 0.456, 0.406], device=im.device).view(3, 1, 1)
    if im.max() > 10:
        std = torch.tensor([1.0 / 255, 1.0 / 255, 1.0 / 255], device=im.device).view(3, 1, 1)
    else:
        std = torch.tensor([0.229, 0.224, 0.225], device=im.device).view(3, 1, 1)
    im = im * std + mean
    if to_numpy:
        return to_np(im)
    return im


def torch_to_im(x):
    if isinstance(x, torch.Tensor):
        if x.shape[0] == 3:
            x = rev_inv(x)
            x = np.transpose(x, (1, 2, 0))
        else:
            x = to_np(x)
            x = np.transpose(x, (1, 2, 0)).squeeze()
    return x


@torch.no_grad()
def test(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    metric = utils.Metric()
    # metric_im = utils.Metric_image()
    loss_list = []

    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data):
        Xs, Xt, Ys, Yt, labels = ret
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(np.array(labels, dtype=np.float32)).to(device)
        labels = labels.float().to(device)
        Xs, Xt, Ys, Yt = (Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device))

        preds, predt, pred_det = model(Xs, Xt)

        loss_p = BCE_loss(predt, Yt, with_logits=True)
        loss_q = BCE_loss(preds, Ys, with_logits=True)
        # loss_det = F.binary_cross_entropy_with_logits(
        #     pred_det.squeeze(), labels.squeeze()
        # )
        loss = loss_p + loss_q

        loss_list.append(loss.data.cpu().numpy())
        print(f"{i}:")

        def fnp(x):
            return x.data.cpu().numpy()

        predt = torch.sigmoid(predt)
        preds = torch.sigmoid(preds)

        metric.update([fnp(Ys), fnp(Yt)], [fnp(preds), fnp(predt)])

        # if logger:
        #     logger.add_scalar("test_loss/total", loss, iteration)
        if plot:
            plot_dir = Path("tmp_plot") / args.dataset
            plot_dir.mkdir(exist_ok=True, parents=True)

            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xt[ii]), torch_to_im(Xs[ii])
                gt1, gt2 = torch_to_im(Yt[ii]), torch_to_im(Ys[ii])
                pred1, pred2 = torch_to_im(predt[ii]), torch_to_im(preds[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1, cmap="jet")
                axes[2, 1].imshow(pred2, cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")

        if num is not None and i >= num:
            break

    metric.final()

    test_loss = np.mean(loss_list)
    print(f"\ntest loss : {test_loss:.4f}\n")
    return test_loss


@torch.no_grad()
def test_temporal(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    metric = utils.Metric()
    im_pred = []

    list_loss = []

    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data.load_temporal(evaluate=True)):
        Xs, Xt, Ys, Yt, labels = ret

        labels = labels.to(device)
        Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

        preds, predt, pred_det = model(Xs, Xt)
        print(f"{i}:")

        loss_t = BCE_loss(predt[labels > 0.5], Yt[labels > 0.5], with_logits=True)
        loss_s = BCE_loss(preds[labels > 0.5], Ys[labels > 0.5], with_logits=True)
        # detection whether video clips are copy move
        # loss_det = F.binary_cross_entropy_with_logits(pred_det, labels)
        pred_det = torch.sigmoid(pred_det)
        pos_mean = torch.sum(labels * pred_det) / (torch.sum(labels) + 1e-8)
        neg_mean = torch.sum((1 - labels) * pred_det) / (torch.sum(1 - labels) + 1e-8)
        loss_det = torch.max(neg_mean - pos_mean + args.beta, torch.tensor(0.0).to(device))
        loss = loss_s + loss_t + args.gamma * loss_det
        list_loss.append(loss.data.cpu().numpy())

        predt = torch.sigmoid(predt)
        preds = torch.sigmoid(preds)
        # pred_det = torch.sigmoid(pred_det)

        labels = to_np(labels)
        metric.update(
            [to_np(Ys)[labels == 1], to_np(Yt)[labels == 1]],
            [to_np(preds)[labels == 1], to_np(predt)[labels == 1]],
        )

        out_ = to_np(pred_det).argmax() == labels.argmax()
        im_pred.append(out_)
        print("CORRECTLY DETECTED!!" if out_ else "WRONG DETECTION!!")

        if num is not None and i >= num:
            break

    metric.final()
    print(f"\nDetection accuracy: {np.sum(im_pred)/len(im_pred)*100: .2f}%\n")

    total_loss = np.mean(list_loss)
    print(f" Test Loss: {total_loss:.4f}")

    return total_loss


@torch.no_grad()
def test_det_vid(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    metric_im = utils.Metric_image()

    def fnp(x):
        return x.data.cpu().numpy()

    for i, ret in enumerate(data.load_mani_vid()):
        X, Y, labels, name = ret
        Y = Y.data.numpy()
        X = X.to(device)
        pred_det, _ = model(X)

        print(f"{i}", end=": ")
        pred_det = fnp(torch.sigmoid(pred_det))

        print(name, end=" : ")
        metric_im.update(labels.flatten(), pred_det.flatten(), thres=args.thres, log=True)

        if num is not None and i >= num:
            break

    out = metric_im.final()

    return out


@torch.no_grad()
def test_det(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    metric_im = utils.Metric_image()
    metric = utils.MMetric()

    list_loss = []
    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data):
        if len(ret) > 3:
            _, X, _, Y, labels = ret
        else:
            X, Y, labels = ret
        Y = Y.data.numpy()
        X = X.to(device)
        pred_det, pred_seg = model(X)

        print(f"{i}:")

        loss = F.binary_cross_entropy_with_logits(
            pred_det.squeeze(), torch.from_numpy(labels).to(device).squeeze()
        )

        list_loss.append(loss.data.cpu().numpy())

        pred_det = to_np(torch.sigmoid(pred_det))
        pred_seg = to_np(torch.sigmoid(pred_seg))

        metric_im.update(labels.flatten(), pred_det.flatten(), thres=args.thres)
        metric.update(Y, pred_seg)

        if num is not None and i >= num:
            break

    metric_im.final()
    metric.final()
    print(f"test loss {np.mean(list_loss): .4f}")

    return np.mean(list_loss)


@torch.no_grad()
def test_dmac(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    metric = utils.Metric()
    # metric_im = utils.Metric_image()
    loss_list = []

    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data.load()):
        Xs, Xt, Ys, Yt, labels = ret
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(np.array(labels, dtype=np.float32)).to(device)
        labels = labels.float().to(device)
        Xs, Xt, Ys, Yt = (Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device))
        preds, predt, _ = model(Xs, Xt)

        def fnp(x):
            return x.data.cpu().numpy()

        if args.model == "dmac":
            predt = torch.softmax(predt, dim=1)[:, [1]]
            preds = torch.softmax(preds, dim=1)[:, [1]]
        else:
            predt = torch.sigmoid(predt)
            preds = torch.sigmoid(preds)

        metric.update([fnp(Ys), fnp(Yt)], [fnp(preds), fnp(predt)])
        if plot:
            plot_dir = Path("tmp_plot_dmac") / args.dataset
            plot_dir.mkdir(exist_ok=True, parents=True)

            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xt[ii]), torch_to_im(Xs[ii])
                gt1, gt2 = torch_to_im(Yt[ii]), torch_to_im(Ys[ii])
                pred1, pred2 = torch_to_im(predt[ii]), torch_to_im(preds[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1, cmap="jet")
                axes[2, 1].imshow(pred2, cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")

        if num is not None and i >= num:
            break

    out = metric.final()

    test_loss = np.mean(loss_list)
    print(f"\ntest loss : {test_loss:.4f}\n")

    return out, test_loss
