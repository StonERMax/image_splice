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
from dataset_vid import Dataset_vid
from utils import CustomTransform, MultiPagePdf

import models
import models_vid
import utils


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

    args = config.config_video_full()
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

    frame_root = Path(args.root) / "grip_video_data" / "frames" / "VIDEO_FORG_rigid_02"

    frame1 = frame_root / f"{33:05d}.png"
    frame2 = frame_root / f"{116:05d}.png"

    im1 = skimage.img_as_float32(skimage.io.imread(frame1))
    im2 = skimage.img_as_float32(skimage.io.imread(frame2))

    im1, _ = transform(im1)
    im2, _ = transform(im2)

    im1 = im1.unsqueeze(0)
    im2 = im2.unsqueeze(0)

    with torch.no_grad():
        preds, predt, pred_det = model(im1.to(device), im2.to(device))

    predt = to_np(torch.sigmoid(predt.squeeze()))
    preds = to_np(torch.sigmoid(preds.squeeze()))

    print("")
