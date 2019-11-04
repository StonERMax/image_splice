from sklearn.metrics import precision_recall_fscore_support
from scipy.ndimage import morphology
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import transforms
import numpy as np
import cv2
import skimage
from skimage import io
from skimage import transform

from scipy.ndimage import morphology
from scipy.ndimage import measurements

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import collections


def get_MCC(hist):
    tn, fp, fn, tp = hist
    denominator = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if denominator == 0:
        return 0
    mcc = (tp*tn - fp*fn)/denominator
    return mcc

def fscore(T_score):
    Fp, Fn, Tp = T_score[1:]
    f_score = 2 * Tp / (2 * Tp + Fp + Fn)
    return f_score


def calc_iou(T_score):
    tn, fp, fn, tp = T_score
    return tp / (fn + fp + tp + 1e-8)


def precision(T_score):
    Fp, _, Tp = T_score[1:]
    recall = Tp / (Tp + Fp + 1e-8)
    return recall


def recall(T_score):
    _, Fn, Tp = T_score[1:]
    recall = Tp / (Tp + Fn + 1e-8)
    return recall


def conf_mat(labels, preds):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels).ravel()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds).ravel()

    tp = np.sum(preds[labels == 1] == 1)
    tn = np.sum(preds[labels == 0] == 0)
    fp = np.sum(preds[labels == 0] == 1)
    fn = np.sum(preds[labels == 1] == 0)
    return np.array([tn, fp, fn, tp])


class SimTransform:
    def __init__(self, size=(224, 224)):
        if isinstance(size, int) or isinstance(size, float):
            size = (size, size)
        # scale
        self.scale = np.random.choice(np.linspace(0.9, 1.1, 30))
        # rotation
        self.rot = np.random.choice(np.linspace(-np.pi / 10, np.pi / 10, 50))
        # translate
        self.translate = (
            np.random.choice(np.linspace(-0.05 * size[1], 0.1 * size[1], 50)),
            np.random.choice(np.linspace(-0.05 * size[0], 0.1 * size[0], 50)),
        )

        # flip lr
        self.flip = np.random.rand() > 0.5

        self.tfm = transform.SimilarityTransform(
            scale=self.scale, translation=self.translate
        )

    def __call__(self, im=None, mask=None):
        if im is not None:
            im = transform.warp(im, self.tfm)
            # im = transform.rotate(im, self.rot)
            if self.flip:
                im = np.flip(im, 1).copy()
        if mask is not None:
            mask = transform.warp(mask, self.tfm)
            # mask = transform.rotate(mask, self.rot)
            if self.flip:
                mask = np.flip(mask, 1).copy()
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
        return im, mask


class CustomTransform_vgg:
    def __init__(self, size=224):
        if isinstance(size, int) or isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([1./255, 1./255, 1./255], dtype=np.float32)
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        if img is not None:
            img = skimage.img_as_float32(img)
            if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[0] != self.size[0] or mask.shape[1] != self.size[1]:
                mask = cv2.resize(
                    mask, self.size, interpolation=cv2.INTER_NEAREST
                )
        return img, mask

    def inverse(self, x, mask=False):
        if x.is_cuda:
            x = x.squeeze().data.cpu().numpy()
        else:
            x = x.squeeze().data.numpy()
        x = x.transpose((1, 2, 0))
        if not mask:
            x = x * self.std + self.mean
        return x

    def __call__(self, img=None, mask=None, other_tfm=None):
        img, mask = self.resize(img, mask)
        if other_tfm is not None:
            img, mask = other_tfm(img, mask)
        if img is not None:
            img = (img - self.mean) / self.std
            img = self.to_tensor(img)
        if mask is not None:
            mask = self.to_tensor(mask)

        return img, mask


class CustomTransform:
    def __init__(self, size=224):
        if isinstance(size, int) or isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        if img is not None:
            img = skimage.img_as_float32(img)
            if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[0] != self.size[0] or mask.shape[1] != self.size[1]:
                mask = cv2.resize(
                    mask, self.size, interpolation=cv2.INTER_NEAREST
                )
        return img, mask

    def inverse(self, x, mask=False):
        if x.is_cuda:
            x = x.squeeze().data.cpu().numpy()
        else:
            x = x.squeeze().data.numpy()
        x = x.transpose((1, 2, 0))
        if not mask:
            x = x * self.std + self.mean
        return x

    def __call__(self, img=None, mask=None, other_tfm=None):
        img, mask = self.resize(img, mask)
        if other_tfm is not None:
            img, mask = other_tfm(img, mask)
        if img is not None:
            img = (img - self.mean) / self.std
            img = self.to_tensor(img)

        if mask is not None:
            mask = self.to_tensor(mask)

        return img, mask


def custom_transform_images(images=None, masks=None, size=320, tsfm=None, other_tfm=None):
    if isinstance(size, int) or isinstance(size, float):
        size = (size, size)
    else:
        size = size
    if tsfm is None:
        tsfm = CustomTransform(size=size)
    X, Y = None, None
    if images is not None:
        X = torch.zeros((images.shape[0], 3, *size), dtype=torch.float32)
        for i in range(images.shape[0]):
            X[i], _ = tsfm(img=images[i], other_tfm=other_tfm)
    if masks is not None:
        Y = torch.zeros((masks.shape[0], 1, *size), dtype=torch.float32)
        for i in range(masks.shape[0]):
            _, Y[i, 0] = tsfm(img=None, mask=masks[i], other_tfm=other_tfm)

    return X, Y

def src_forge(src, forge, c1=2, c2=0):
    im = np.zeros((*src.shape[:2], 3))
    im[..., c1] = src
    im[..., c2] = forge
    return im


def add_overlay(im, m1, m2=None, alpha=0.5, c1=[0, 1, 0], c2=[1, 0, 0]):
    r, c = im.shape[:2]

    M1 = np.zeros((r, c, 3), dtype=np.float32)
    M2 = np.zeros((r, c, 3), dtype=np.float32)

    if m2 is not None:
        M1[m1 > 0.5] = c1
        M2[m2 > 0.5] = c2
        M = cv2.addWeighted(M1, 1, M2, 1, 0, None)
    else:
        M1[m1 > 0.5] = c1
        M = M1
    Im = cv2.addWeighted(im, 1, M, alpha, 0, None)
    Im = np.clip(Im, 0, 1.)
    return Im


class MultiPagePdf:
    def __init__(self, total_im, out_name, nrows=4, ncols=4, figsize=(8, 6)):
        """init

        Keyword Arguments:
            total_im {int} -- #images
            nrows {int} -- #rows per page (default: {4})
            ncols {int} -- #columns per page (default: {4})
            figsize {tuple} -- fig size (default: {(8, 6)})
        """
        self.total_im = total_im
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = tuple(figsize)
        self.out_name = out_name

        # create figure and axes
        total_pages = int(np.ceil(total_im / (nrows * ncols)))

        self.figs = []
        self.axes = []

        for _ in range(total_pages):
            f, a = plt.subplots(nrows, ncols)

            f.set_size_inches(figsize)
            self.figs.append(f)
            self.axes.extend(a.flatten())

        self.cnt_ax = 0

    def plot_one(self, x, *args, **kwargs):
        ax = self.axes[self.cnt_ax]
        ax.imshow(x, *args, **kwargs)  # prediction
        # ax.imshow(x[0])  # ground truth

        ax.set_xticks([])
        ax.set_yticks([])

        self.cnt_ax += 1
        return ax

    def final(self):
        with PdfPages(self.out_name) as pdf:
            for fig in self.figs:
                fig.tight_layout()
                pdf.savefig(fig)
        plt.close("all")


class MMetric:
    def __init__(self, name="", thres=0.5):
        self.T = np.zeros(4)
        self.fscore = []
        self.prec = []
        self.rec = []
        self.iou = []

        self.MCC = []

        self.name = name
        self.thres = thres

    def update(self, gt, pred, batch_mode=True, log=True):

        if batch_mode:
            num = gt.shape[0]
            arr = []
            for i in range(num):
                fs = self._update(gt[i], pred[i], log=False)
                if fs != -1:
                    arr.append(fs)
            if len(arr) > 0:
                print(f"\t\t{self.name} f score : {np.mean(arr):.4f}")
        else:
            self._update(gt, pred, log=log)

    def _update(self, gt, pred, log=True):
        gt = gt.ravel() > self.thres
        pred = pred.ravel() > self.thres

        tt = conf_mat(gt, pred)
        self.T += tt

        if np.all(gt == 0):
            return -1

        prec = precision(tt)
        rec = recall(tt)
        fs = fscore(tt)
        mcc = get_MCC(tt)

        self.prec.append(prec)
        self.rec.append(rec)
        self.fscore.append(fs)
        self.iou.append(calc_iou(tt))
        self.MCC.append(mcc)

        if log:
            print(
                f"{self.name} precision : {prec:.4f}, recall : {rec:.4f}, "
                + f"f1 : {fs:.4f}"
            )
        return fs

    def final(self):
        # protocal A
        print(f"\n{self.name} ")
        print("-" * 50)
        # print("\nProtocol A:")
        # print(
        #     f"precision : {precision(self.T):.4f}, "
        #     + f"recall : {recall(self.T):.4f}, "
        #     + f"f1 : {fscore(self.T):.4f}, "
        #     + f"iou : {calc_iou(self.T):.4f}"
        # )

        # protocol B
        # print("\nProtocol B:")Pmai
        print(
            f"precision : {np.mean(self.prec):.4f}, "
            + f"recall : {np.mean(self.rec):.4f}, "
            + f"f1 : {np.mean(self.fscore):.4f}, "
            + f"iou : {np.mean(self.iou):.4f} "
            + f"mcc : {np.mean(self.MCC):.4f}"
        )

        return np.mean(self.fscore)


class Metric:
    def __init__(self, names=["source", "forge"], thres=0.5):
        self.names = names
        self.dims = len(names)

        self.list_metrics = []
        for i in range(self.dims):
            self.list_metrics.append(MMetric(name=names[i], thres=0.5))

    def update(self, gt, pred, batch_mode=True):
        for i in range(self.dims):
            self.list_metrics[i].update(gt[i], pred[i], batch_mode=batch_mode)

    def final(self):
        sc = []
        for i in range(self.dims):
            sc.append(self.list_metrics[i].final())
        return np.mean(sc)


class Metric_image(object):
    def __init__(self):
        self.gt = []
        self.pred = []

    def update(self, _gt, _pred, thres=0.5, log=False):
        _gt = _gt > thres
        _pred = _pred > thres

        if isinstance(_gt, collections.abc.Iterable):
            self.gt.extend(list(_gt))
            self.pred.extend(list(_pred))
        else:
            self.gt.append(_gt)
            self.pred.append(_pred)
        if log:
            pr, re, f, _ = precision_recall_fscore_support(
                _gt, _pred, average="binary"
            )
            print(f"precision: {pr:.4f}, recall: {re:.4f}, f-score: {f :.4f} ")

    def final(self):
        pr, re, f, _ = precision_recall_fscore_support(
            self.gt, self.pred, average="binary"
        )

        print("Image level score")
        print(f"precision: {pr:.4f}, recall: {re:.4f}, f-score: {f :.4f} ")


class Preprocessor():
    def __init__(self, args):
        self.args = args

    # morphological preprocessing
    def morph(self, mask):
        # threshold
        mask = mask > self.args.thres

        # do closing
        mask_closed = morphology.binary_closing(mask, structure=np.ones((5, 5)))

        # do opening
        mask_opened = morphology.binary_closing(
            mask_closed, structure=np.ones((5, 5)))

        # hole filling
        mask_filled = morphology.binary_fill_holes(
            mask_opened, structure=np.ones((8, 8)))

        # get maximum connected component
        labels, _ = measurements.label(mask_filled)
        if len(np.unique(labels)) == 1:
            return np.zeros_like(mask)

        mask_maxlab = labels == np.argmax(np.bincount(labels.flat)[1:])+1

        return mask_maxlab.astype(np.float)

    def calc_hist(self, im, mask=None):
        # convert to hsv color space
        im = skimage.img_as_ubyte(im)
        mask = skimage.img_as_ubyte(mask)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

        h_bins = 50
        s_bins = 50
        histSize = [h_bins, s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges  # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]

        hist_base = cv2.calcHist([im_hsv], channels, mask,
                                 histSize, ranges, accumulate=False)
        cv2.normalize(hist_base, hist_base, alpha=0,
                     beta=1, norm_type=cv2.NORM_MINMAX)
        return hist_base

    def comp_hist(self, x1, x2, mask1=None, mask2=None, compare_method=0):
        hist1 = self.calc_hist(x1, mask1)
        hist2 = self.calc_hist(x2, mask2)

        value = cv2.compareHist(hist1, hist2, compare_method)
        return value
