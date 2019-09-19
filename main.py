import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import skimage

# custom module
import config
import models
from train import train

from test import test
import dataset
import utils


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config()

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

    # load dataset
    data_train = dataset.Dataset_COCO_CISDL(args, mode=None, is_training=True)
    data_test = dataset.Dataset_COCO_CISDL(args, mode=None, is_training=False)

    # model
    model = models.DOAModel(out_channel=args.out_channel)
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model_params = [
        {"params": model.get_1x_lr_params(), "lr": args.lr},
        {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
    ]

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True, threshold=0.1, min_lr=1e-7
    )

    list_loss = []

    if args.test:
        test(data_test, model, args, iteration=None, device=device, logger=None, num=5)
    else:
        for ep in tqdm(range(init_ep, args.max_epoch)):
            # train
            for ret in data_train.load():
                loss = train(ret, model, optimizer, args, iteration, device, logger=logger)
                list_loss.append(loss)
                iteration += 1

                if iteration % 300 == 0:
                    scheduler.step(np.mean(list_loss))
                    list_loss = []

                    # test(data_test, model, args, iteration=None, device=device, logger=None, num=5)
                    state = model.module.state_dict() if isinstance(model, nn.DataParallel) \
                        else model.state_dict()

                    torch.save(
                        {
                            "epoch": ep,
                            "model_state": state
                        },
                        "./ckpt/" + model_name + ".pkl",
                    )

    logger.close()
