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

    # model for manipulation detection
    model_forge = models.Base_DetSegModel()
    if args.ckptM is not None:
        checkpoint = torch.load(args.ckptM)
        load_state_base(model_forge, checkpoint['model_state'])
    model_forge.to(device)

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
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    dataset = Dataset_vid(args=args, is_training=False)

    root = Path("tmp_video_match_mani") / args.dataset / args.model

    def to_np(x):
        return x.data.cpu().numpy()

    # mask_processor = utils.Preprocessor(args)

    # for batch normalization
    with torch.no_grad():
        for i, ret in enumerate(dataset.load()):
            Xs, Xt, Ys, Yt, labels = ret
            Xs, Xt = (Xs.to(device), Xt.to(device))
            _ = model(Xs, Xt)
            _ = model_forge(Xt)
            if i > 5:
                break

    model.eval()
    model_forge.eval()

    counter = 0
    for ret in tqdm(dataset.load_videos_all(is_training=False,
                                            shuffle=False, to_tensor=True)):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        N = X.shape[0]
        print(name, " : ", N)

        forge_time = np.arange(forge_time[0], forge_time[1]+1)
        gt_time = np.arange(gt_time[0], gt_time[1]+1)

        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        """Forge time prediction
        """
        X_tensor = X.to(device)
        with torch.no_grad():
            pred_det, _ = model_forge(X_tensor)

        pred_det = torch.sigmoid(pred_det).data.cpu().numpy()
        D_pred = np.zeros((N, N, 2, *args.size))
        _ind = np.where(pred_det > args.thres)[0]
        pred_forge_time = np.arange(_ind.min(), _ind.max() + 1)
        N_forge = len(pred_forge_time)

        Xt = X[pred_forge_time].to(device)

        for i in range(0, N - N_forge + 1):
            Xr = X_tensor[i : i + N_forge]
            
            with torch.no_grad():
                out1, out2, _ = model(Xr, Xt)

            if args.model in ("dmac"):
                out1 = torch.softmax(out1, dim=1)[:, 1]
                out2 = torch.softmax(out2, dim=1)[:, 1]
            else:
                out1, out2 = torch.sigmoid(out1), torch.sigmoid(out2)

            # D_pred[pred_forge_time, i : i + N_forge, 0] = out1.squeeze().data.cpu()
            # D_pred[pred_forge_time, i : i + N_forge, 1] = out2.squeeze().data.cpu()
            out1_np = out1.squeeze().data.cpu()
            out2_np = out2.squeeze().data.cpu()
            for cnt, (ki, kj) in enumerate(zip(pred_forge_time, range(i, i+N_forge))):
                    D_pred[ki, kj, 0] = out1_np[cnt]
                    D_pred[ki, kj, 1] = out2_np[cnt]

        torch.save(
            {
                "forge_time": pred_forge_time,
                "D_pred": D_pred,
                "name": name
            },
            str(path / "data_pred.pt")
        )

        iou_forge_time = iou_time(pred_forge_time, forge_time)
        print(f"time iou forge : {iou_forge_time: .4f}")

        counter += 1

        # TODO:  Keep an eye here
        if counter > 20:
            break
