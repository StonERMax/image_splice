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
from dataset_vid import Dataset_world
from utils import CustomTransform, MultiPagePdf

import models
import models_vid
import utils
from test import to_np, torch_to_im

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

    args = config.config_world()
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    transform = utils.CustomTransform(size=args.size)

    # model
    model = models.DOAModel(out_channel=args.out_channel)
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    dataset = Dataset_world(args=args)

    root = Path("tmp_temporal_video_match_mani") / args.dataset / args.model

    # mask_processor = utils.Preprocessor(args)

    model.eval()

    metric = utils.Metric(names=["source", "forge"])
    counter = 0
    for ret in tqdm(
        dataset.load_videos_all(
            is_training=False, shuffle=False, to_tensor=True
        )
    ):
        Xs, Xt, Ys, Yt = ret

        path = root / f"{counter}"
        path.mkdir(parents=True, exist_ok=True)

        N = Xs.shape[0]

        """Forge time prediction
        """
        pred_forge_time = np.arange(0, Xt.shape[0])
        N_forge = len(pred_forge_time)

        Xt = Xt[pred_forge_time].to(device)

        list_out_det_score = []

        D_pred = []
        for i in range(0, N - N_forge + 1):
            Xr = Xs[i : i + N_forge]

            with torch.no_grad():
                out1, out2, out_det = model(Xr.to(device), Xt.to(device))
            out1 = torch.sigmoid(out1)
            out2 = torch.sigmoid(out2)

            out1_np = out1.squeeze().data.cpu().numpy()
            out2_np = out2.squeeze().data.cpu().numpy()
            # for cnt, (ki, kj) in enumerate(zip(pred_forge_time, range(i, i+N_forge))):
            #         D_pred[ki, kj, 0] = out1_np[cnt]
            #         D_pred[ki, kj, 1] = out2_np[cnt]
            # D_pred.append((out1_np, out2_np))

        # amax = np.argmax(list_out_det_score)
        # pred_source_time = np.arange(amax, amax + N_forge)
        pred_source_time = np.arange(0, Xs.shape[0])

        # iou_forge_time = iou_time(pred_forge_time, forge_time)
        # iou_source_time = iou_time(pred_source_time, gt_time)

        # print(f"time iou forge : {iou_forge_time: .4f}")
        # print(f"time iou source : {iou_source_time: .4f}")

        # pred_s, pred_t = D_pred[amax]
        pred_s, pred_t = out1_np, out2_np

        Y_forge = Yt.squeeze().data.numpy()
        Y_orig = Ys.squeeze().data.numpy()

        Y_pred_forge = np.zeros_like(Y_forge)
        Y_pred_source = np.zeros_like(Y_orig)

        Y_pred_forge[pred_forge_time] = pred_t
        Y_pred_source[pred_source_time] = pred_s

        metric.update((Y_orig, Y_forge), (Y_pred_source, Y_pred_forge), batch_mode=False)

        if args.plot:
            for ii in range(Xt.shape[0]):
                im1, im2 = torch_to_im(Xt[ii]), torch_to_im(Xs[ii])
                gt1, gt2 = Y_forge[ii].squeeze(), Y_orig[ii].squeeze()
                pred1, pred2 = pred_t[ii].squeeze(), pred_s[ii].squeeze()

                fig, axes = plt.subplots(nrows=3, ncols=2)
                axes[0, 0].imshow(im1)
                axes[0, 1].imshow(im2)
                axes[1, 0].imshow(gt1, cmap="jet")
                axes[1, 1].imshow(gt2, cmap="jet")
                axes[2, 0].imshow(pred1, cmap="jet")
                axes[2, 1].imshow(pred2, cmap="jet")

                fig.savefig(str(root / f"{i}_{ii}.jpg"))
                plt.close("all")

        counter += 1

        # TODO:  Keep an eye here
        # if counter > 20:
        #     break
    metric.final()
