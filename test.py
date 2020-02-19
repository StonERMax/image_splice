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
import shutil

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
        std = torch.tensor([1.0 / 255, 1.0 / 255, 1.0 / 255], device=im.device).view(
            3, 1, 1
        )
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

    metric = utils.Metric(thres=args.thres)
    # metric_im = utils.Metric_image()
    loss_list = []

    if plot:
        plot_dir = Path("tmp_plot") / args.dataset / args.model
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)

    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data):
        Xs, Xt, Ys, Yt, labels = ret
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(np.array(labels, dtype=np.float32)).to(device)
        labels = labels.float().to(device)
        Xs, Xt, Ys, Yt = (Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device))

        preds, predt = model(Xs, Xt)

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
            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xt[ii]), torch_to_im(Xs[ii])
                gt1, gt2 = torch_to_im(Yt[ii]), torch_to_im(Ys[ii])
                pred1, pred2 = to_np(predt[ii]), to_np(preds[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1.squeeze(), cmap="jet")
                axes[2, 1].imshow(pred2.squeeze(), cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")

        if num is not None and i >= num:
            break

    metric.final()

    test_loss = np.mean(loss_list)
    print(f"\ntest loss : {test_loss:.4f}\n")
    return test_loss


@torch.no_grad()
def test_cmfd(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()

    if args.mode == "both":
        names = ["source", "forge", "all"]
    else:
        names = ["mask"]
    metric = utils.Metric(names=names, thres=args.thres)
    # metric_im = utils.Metric_image()
    loss_list = []

    if plot:
        plot_dir = Path("tmp_plot_cmfd") / args.dataset / args.model
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)

    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data):
        if len(ret) == 5:
            Xs, Xt, Ys, Yt, labels = ret
        elif len(ret) == 2:
            X, Y = ret
            if Y.shape[1] > 1:
                Xs, Xt, Ys, Yt = X, X, Y[:, [1]], Y[:, [0]]
            else:
                Xs, Xt, Ys, Yt = X, X, Y, Y

        Xs, Xt, Ys, Yt = (Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device))
        if args.mode == "both":
            preds, predt = model(Xs, Xt)
            predt = torch.sigmoid(predt)
            preds = torch.sigmoid(preds)
        else:
            preds = model(Xs, Xt)
            preds = torch.sigmoid(preds)
            predt = preds

        # loss_p = BCE_loss(predt, Yt, with_logits=True)
        # loss_q = BCE_loss(preds, Ys, with_logits=True)
        # loss_det = F.binary_cross_entropy_with_logits(
        #     pred_det.squeeze(), labels.squeeze()
        # )
        # loss = loss_p + loss_q

        # loss_list.append(loss.data.cpu().numpy())
        print(f"{i}:")

        if args.mode == "both":
            metric.update(
                [to_np(Ys), to_np(Yt), to_np(torch.max(Ys, Yt))],
                [to_np(preds), to_np(predt), to_np(torch.max(preds, predt))],
            )
        elif args.mode == "mani":
            metric.update([to_np(Yt)], [to_np(preds)])
        else:
            metric.update([to_np(torch.max(Ys, Yt))], [to_np(preds)])

        # if logger:
        #     logger.add_scalar("test_loss/total", loss, iteration)
        if plot:
            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xt[ii]), torch_to_im(Xs[ii])
                gt1, gt2 = torch_to_im(Yt[ii]), torch_to_im(Ys[ii])
                pred1, pred2 = to_np(predt[ii]), to_np(preds[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1.squeeze(), cmap="jet")
                axes[2, 1].imshow(pred2.squeeze(), cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")

        if num is not None and i >= num:
            break

    metric.final()

    test_loss = np.mean(loss_list)
    print(f"\ntest loss : {test_loss:.4f}\n")
    return test_loss




@torch.no_grad()
def test_casia(data, model, args, iteration, device, logger=None, num=None, plot=False):

    model.eval()
    # metric_im = utils.Metric_image()
    metric = utils.MMetric(name="forge", thres=args.thres)
    if iteration is not None:
        print(f"{iteration}")
    if plot:
        plot_dir = Path("tmp_plot") / args.dataset / args.model
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)

    for i, ret in enumerate(data):
        Xs, Xt, Y, labels = ret
        preds, predt = model(Xs.to(device), Xt.to(device))
        print(f"{i}:")

        if args.model == "dmac":
            predt = torch.softmax(predt, dim=1)[:, [1]]
            preds = torch.softmax(preds, dim=1)[:, [1]]
        else:
            predt = torch.sigmoid(predt)
            preds = torch.sigmoid(preds)

        metric.update(to_np(Y), to_np(predt))

        if plot:
            preds = preds.squeeze()
            predt = predt.squeeze()

            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xs[ii]), torch_to_im(Xt[ii])
                gt2 = to_np(Y[ii].squeeze())
                pred1, pred2 = to_np(preds[ii]), to_np(predt[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                # axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1.squeeze(), cmap="jet")
                axes[2, 1].imshow(pred2.squeeze(), cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")

        if num is not None and i >= num:
            break
    # metric_im.final()
    metric.final()


@torch.no_grad()
def test_casia_cmfd(
    data, model, args, iteration, device, logger=None, num=None, plot=False
):

    model.eval()
    # metric_im = utils.Metric_image()
    metric = utils.Metric(names=["all"], thres=args.thres)
    if iteration is not None:
        print(f"{iteration}")
    if plot:
        plot_dir = Path("tmp_plot_cmfd") / args.dataset / args.model
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)

    for i, ret in enumerate(data):
        X, Y = ret
        Xs, Xt = X, X
        if Y.shape[1] >= 2:
            Ys, Yt = Y[:, [1]], Y[:, [0]]
        else:
            Ys, Yt = Y, Y

        if args.mode == "both":
            preds, predt = model(Xs.to(device), Xt.to(device))
            predt = torch.sigmoid(predt)
            preds = torch.sigmoid(preds)
        else:
            preds = model(Xs.to(device), Xt.to(device))
            preds = torch.sigmoid(preds)
            predt = preds

        print(f"{i}:")
        ytn, ysn = to_np(Yt), to_np(Ys)
        predtn, predsn = to_np(predt), to_np(preds)
        metric.update([np.maximum(ytn, ysn)], [np.maximum(predsn, predtn)])

        if plot:
            preds = preds.squeeze()
            predt = predt.squeeze()

            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xs[ii]), torch_to_im(Xt[ii])
                gt1, gt2 = to_np(Ys[ii].squeeze()), to_np(Yt[ii].squeeze())
                pred1, pred2 = to_np(preds[ii]), to_np(predt[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1.squeeze(), cmap="jet")
                axes[2, 1].imshow(pred2.squeeze(), cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")

        if num is not None and i >= num:
            break
    # metric_im.final()
    metric.final()


@torch.no_grad()
def test_casia_det(
    data, model, args, iteration, device, logger=None, num=None, plot=False
):
    model.eval()
    metric_im = utils.Metric_image(thres=0.77, with_auc=True)
    if iteration is not None:
        print(f"{iteration}")

    if plot:
        plot_dir = Path("tmp_plot") / args.dataset / (args.model + "_det")
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)

    for i, ret in enumerate(data):
        Xs, Xt, labels = ret
        preds_, predt_, _ = model(Xs.to(device), Xt.to(device))
        print(f"{i}:")

        if args.model == "dmac":
            predt_ = torch.softmax(predt_, dim=1)[:, [1]]
            preds_ = torch.softmax(preds_, dim=1)[:, [1]]
        else:
            predt_ = torch.sigmoid(predt_)
            preds_ = torch.sigmoid(preds_)

        preds, predt = to_np(preds_), to_np(predt_)

        # detect
        gt_labels = labels.data.numpy()
        pred_labels = []
        for j in range(preds.shape[0]):
            # _sum = (preds[j] > args.thres).sum() + (predt[j] > args.thres).sum()
            # if (preds[j] > args.thres).sum() > 20 and (predt[j] > args.thres).sum() > 20:
            # if _sum > 100:
            #     pred_labels.append(1.0)
            # else:
            #     pred_labels.append(0.0)
            sa = np.mean(preds[j][preds[j] > args.thres])
            sb = np.mean(predt[j][predt[j] > args.thres])
            sab = (sa + sb) / 2
            if np.isnan(sab):
                sab = 0
            pred_labels.append(sab)

        pred_labels = np.array(pred_labels)
        metric_im.update(gt_labels, pred_labels, log=True)

        if plot:
            preds = preds.squeeze()
            predt = predt.squeeze()

            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xs[ii]), torch_to_im(Xt[ii])
                pred1, pred2 = to_np(preds[ii]), to_np(predt[ii])

                fig, axes = plt.subplots(nrows=2, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(pred1.squeeze(), cmap="jet")
                axes[1, 1].imshow(pred2.squeeze(), cmap="jet")
                fig.savefig(
                    str(
                        plot_dir
                        / f"{i}_{ii}_{'pos' if gt_labels[ii]==1 else 'neg'}_{'pos' if pred_labels[ii]==1 else 'neg'}.jpg"
                    )
                )
                plt.close("all")

        if num is not None and i >= num:
            break
    metric_im.final()
