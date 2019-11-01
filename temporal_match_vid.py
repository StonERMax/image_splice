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

    # model for manipulation detection
    model_forge = models.Base_DetSegModel()
    if args.ckptM is not None:
        checkpoint = torch.load(args.ckptM)
        load_state_base(model_forge, checkpoint['model_state'])
    model_forge.to(device)

    # model
    model = models_vid.DOAModel(out_channel=args.out_channel)
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    dataset = Dataset_vid(args=args, is_training=False)

    root = Path("tmp_temporal_video_match_mani") / args.dataset / args.model

    def to_np(x):
        return x.data.cpu().numpy()

    # mask_processor = utils.Preprocessor(args)

    # for batch normalization
    with torch.no_grad():
        for i, ret in enumerate(dataset.load_temporal(t_t_max=5, batch_size=5)):
            Xs, Xt, Ys, Yt, labels = ret
            Xs, Xt = Xs.to(device), Xt.to(device)
            _ = model(Xs, Xt)
            if i > 5:
                break

    model.eval()
    model_forge.eval()

    metric = utils.Metric(names=["source", "forge"])

    counter = 0
    for ret in tqdm(dataset.load_videos_all(is_training=False,
                                            shuffle=False, to_tensor=True)):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        N = X.shape[0]
        print(name, " : ", N)

        if N < 5:
            print("-----")
            # continue    

        forge_time = np.arange(forge_time[0], forge_time[1]+1)
        gt_time = np.arange(gt_time[0], gt_time[1]+1)

        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        """Forge time prediction
        """
        X_tensor = X.to(device)
        if X.shape[0] > 20:
            ind_splits = np.array_split(np.arange(X.shape[0]), int(np.ceil(X.shape[0]/20)))
            pred_det = []
            for _ind in ind_splits:
                with torch.no_grad():
                    _pred_det, _ = model_forge(X_tensor[_ind])
                    _pred_det = torch.sigmoid(_pred_det).data.cpu().numpy()
                    pred_det.append(_pred_det)
            pred_det = np.concatenate(pred_det)
        else:
            with torch.no_grad():
                pred_det, _ = model_forge(X_tensor)
                pred_det = torch.sigmoid(pred_det).data.cpu().numpy()

        D_pred = np.zeros((N, N, 2, *args.size))
        _ind = np.where(pred_det > args.thres)[0]
        pred_forge_time = np.arange(_ind.min(), _ind.max() + 1)
        N_forge = len(pred_forge_time)

        Xt = X[pred_forge_time].to(device)

        list_out_det_score = []

        D_pred = []
        for i in range(0, N - N_forge + 1):
            Xr = X_tensor[i : i + N_forge]
            
            with torch.no_grad():
                out1, out2, out_det = model(Xr.unsqueeze(0), Xt.unsqueeze(0))
            out1 = torch.sigmoid(out1)
            out2 = torch.sigmoid(out2)
            out_det = torch.sigmoid(out_det)

            out1_np = out1.squeeze().data.cpu().numpy()
            out2_np = out2.squeeze().data.cpu().numpy()
            list_out_det_score.append(out_det.squeeze().data.cpu().numpy())
            # for cnt, (ki, kj) in enumerate(zip(pred_forge_time, range(i, i+N_forge))):
            #         D_pred[ki, kj, 0] = out1_np[cnt]
            #         D_pred[ki, kj, 1] = out2_np[cnt]
            D_pred.append((out1_np, out2_np))

        amax = np.argmax(list_out_det_score)
        pred_source_time = np.arange(amax, amax+N_forge)

        iou_forge_time = iou_time(pred_forge_time, forge_time)
        iou_source_time = iou_time(pred_source_time, gt_time)
        
        print(f"time iou forge : {iou_forge_time: .4f}")
        print(f"time iou source : {iou_source_time: .4f}")

        pred_s, pred_t = D_pred[amax]

        Y_forge = Y_forge.squeeze().data.numpy()
        Y_orig = Y_orig.squeeze().data.numpy()

        Y_pred_forge = np.zeros_like(Y_forge)
        Y_pred_source = np.zeros_like(Y_orig)

        Y_pred_forge[pred_forge_time] = pred_t
        Y_pred_source[pred_source_time] = pred_s

        metric.update(
            (Y_orig, Y_forge),
            (Y_pred_source, Y_pred_forge)
        )
        counter += 1

        # TODO:  Keep an eye here
        if counter > 30:
            break
    metric.final()
