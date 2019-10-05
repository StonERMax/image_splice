import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

# custom module
import config
import models

from torchvision.utils import save_image

import dataset_vid
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

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model
    model = models.DOAModel(out_channel=args.out_channel)
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    data = dataset_vid.Dataset_vid(args)

    def fnp(x):
        return x.data.cpu().numpy()

    metric = utils.Metric()
    for i, ret in enumerate(
        data.load_data_template_match_pair(is_training=False)
    ):
        Xs, Xt, Ys, Yt, name = ret
        Xs, Xt, Ys, Yt = (
            Xs.to(device),
            Xt.to(device),
            Ys.to(device),
            Yt.to(device),
        )

        preds, predt, pred_det = model(Xs, Xt)
        predt = torch.sigmoid(predt)
        preds = torch.sigmoid(preds)

        metric.update([fnp(Ys), fnp(Yt)], [fnp(preds), fnp(predt)])
        break

    out = metric.final()
