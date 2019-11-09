"""main file for saving source and forge mask, with pre-processing by 
detection
"""

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import skimage

# custom module
import config
import dataset_vid
from utils import CustomTransform, MultiPagePdf
import shutil
import models
import models_vid
import utils
from test import torch_to_im, to_np


def iou_time(t1, t2):
    iou = len(set(t1).intersection(set(t2))) / (
        max(len(set(t1).union(set(t2))), 1e-8)
    )
    return iou


def load_state_base(model, state):
    new_state = {}
    for k in state:
        k = str(k)
        if k.startswith("base"):
            new_k = k.replace("base.", "")
            new_state[new_k] = state[k]
    model.load_state_dict(new_state)
    return model


if __name__ == "__main__":
    # device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_grip()
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    transform = utils.CustomTransform(size=args.size)

    # model
    model = models.DOAModel()
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model.eval()

    def to_np(x):
        return x.data.cpu().numpy()

    data = dataset_vid.Dataset_grip(args)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    if args.plot:
        plot_dir = Path("tmp_plot") / args.dataset
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(exist_ok=True, parents=True)

    for i, ret in tqdm(enumerate(data_loader)):
        Xs, Xt, Ys, Yt = ret
        Xs, Xt = (Xs.to(device), Xt.to(device))
        with torch.no_grad():
            preds, predt, pred_det = model(Xs, Xt)
        predt = torch.sigmoid(predt.squeeze())
        preds = torch.sigmoid(preds.squeeze())

        if args.plot:
            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xt[ii]), torch_to_im(Xs[ii])
                gt1, gt2 = torch_to_im(Yt[ii]), torch_to_im(Ys[ii])
                pred1, pred2 = to_np(predt[ii]), to_np(preds[ii])

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1, cmap="jet")
                axes[2, 1].imshow(pred2, cmap="jet")

                fig.savefig(str(plot_dir / f"{i}_{ii}.jpg"))
                plt.close("all")
        
        # TODO: how many iterations?
        if i > 30:
            break
