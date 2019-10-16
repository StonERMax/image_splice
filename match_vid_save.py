"""main file for training bubblenet type comparison patch matching
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
import utils


if __name__ == "__main__":
    # device

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_video()
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    transform = utils.CustomTransform(size=args.size)

    # model
    if args.model in ("dmvn", "dmac"):
        model = models.get_dmac(args.model)
    else:
        model = models.DOAModel(out_channel=args.out_channel)
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    dataset = Dataset_vid(args=args, is_training=False)

    root = Path("tmp_video_match") / args.dataset / args.model

    def to_np(x):
        return x.data.cpu().numpy()

    mask_processor = utils.Preprocessor(args)

    # for batch normalization
    with torch.no_grad():
        for i, ret in enumerate(dataset.load()):
            Xs, Xt, Ys, Yt, labels = ret
            Xs, Xt = (Xs.to(device), Xt.to(device))
            _ = model(Xs, Xt)
            if i > 5:
                break

    model.eval()

    for ret in tqdm(dataset.load_videos_all(is_training=False,
                                            shuffle=True, to_tensor=True)):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        N = X.shape[0]
        print(name, " : ", N)

        forge_time = np.arange(forge_time[0], forge_time[1]+1)
        gt_time = np.arange(gt_time[0], gt_time[1]+1)

        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        D_pred = torch.zeros((N, N, 2, *args.size))
        for i in range(N):
            Xr = X.to(device)
            Xt = X[[i]].repeat((Xr.shape[0], 1, 1, 1)).to(device)

            if i in forge_time:
                i_ind = np.where(forge_time == i)[0][0]
                gt_ind = gt_time[i_ind]
            else:
                gt_ind = None

            with torch.no_grad():
                out1, out2, _ = model(Xr, Xt)

            if args.model in ("dmac"):
                out1 = torch.softmax(out1, dim=1)[:, 1]
                out2 = torch.softmax(out2, dim=1)[:, 1]
            else:
                out1, out2 = torch.sigmoid(out1), torch.sigmoid(out2)

            D_pred[i, :, 0] = out1.squeeze().data.cpu()
            D_pred[i, :, 1] = out2.squeeze().data.cpu()

        torch.save(
            {
                "X": X,
                "Y_forge": Y_forge,
                "Y_orig": Y_orig,
                "forge_time": forge_time,
                "gt_time": gt_time,
                "D_pred": D_pred,
                "name": name
            },
            str(path / "data_pred.pt")
        )
