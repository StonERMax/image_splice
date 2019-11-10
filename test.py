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
def test_temporal(data, model, args, iteration, device, logger=None, num=None, plot=False):
    model.eval()
    metric = utils.Metric()
    im_pred = []
    list_loss = []
    if iteration is not None:
        print(f"{iteration}")

    for i, ret in enumerate(data.load_temporal_pos(evaluate=True, batch_size=1)):
        Xs, Xt, Ys, Yt, labels = ret

        labels = labels.to(device)
        Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

        preds, predt = model(Xs, Xt)
        print(f"{i}:")

        loss_t = BCE_loss(predt[labels > 0.5], Yt[labels > 0.5], with_logits=True)
        loss_s = BCE_loss(preds[labels > 0.5], Ys[labels > 0.5], with_logits=True)
        loss = loss_s + loss_t
        list_loss.append(loss.data.cpu().numpy())

        predt = torch.sigmoid(predt)
        preds = torch.sigmoid(preds)
        # pred_det = torch.sigmoid(pred_det)

        labels = to_np(labels)
        metric.update(
            [to_np(Ys)[labels == 1], to_np(Yt)[labels == 1]],
            [to_np(preds)[labels == 1], to_np(predt)[labels == 1]],
            batch_mode=False
        )

        # out_ = to_np(pred_det).argmax() == labels.argmax()
        # im_pred.append(out_)
        # print("CORRECTLY DETECTED!!" if out_ else "WRONG DETECTION!!")

        if num is not None and i >= num:
            break

    metric.final()
    # print(f"\nDetection accuracy: {np.sum(im_pred)/len(im_pred)*100: .2f}%\n")

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
            pred_det.squeeze(), labels.to(device).squeeze().float()
        )

        list_loss.append(loss.data.cpu().numpy())

        pred_det = to_np(torch.sigmoid(pred_det))
        pred_seg = to_np(torch.sigmoid(pred_seg))

        metric_im.update(to_np(labels).flatten(), pred_det.flatten(), thres=args.thres)
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
        preds, predt, _ = model(Xs, Xt)
       
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
        
        if args.model == "dmac":
            predt = torch.softmax(predt, dim=1)[:, [1]]
            preds = torch.softmax(preds, dim=1)[:, [1]]
        else:
            predt = torch.sigmoid(predt)
            preds = torch.sigmoid(preds)
        
        loss_list.append(to_np(loss))

        metric.update([to_np(Ys), to_np(Yt)], [to_np(preds), to_np(predt)])

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
        preds, predt, _ = model(Xs.to(device), Xt.to(device))
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
def test_casia_det(data, model, args, iteration, device, logger=None, num=None, plot=False):
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


@torch.no_grad()
def test_dmac_casia(data, model, args, iteration, device, logger=None, num=None, plot=False):

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

    out = metric.final()

    test_loss = np.mean(loss_list)
    print(f"\ntest loss : {test_loss:.4f}\n")

    return out, test_loss
