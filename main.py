"""main file for training bubblenet type comparison patch matching
"""

import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
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

    args = config.config_USC()
    args.size = tuple(int(i) for i in args.size.split("x"))

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + \
        args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    logger = SummaryWriter("./logs/" + model_name)

    # load dataset
    dataset_usc = dataset.USCISI_CMD_Dataset(lmdb_dir=args.lmdb_dir, args=args,
                                             sample_file=args.train_key)
    dataset_coco = dataset.COCODataset(args=args, is_training=True,
                                       sample_len=len(dataset_usc))

    dataset_train = torch.utils.data.ConcatDataset((dataset_usc, dataset_coco))
    # dataset_train = dataset_usc

    dataset_test_usc = dataset.USCISI_CMD_Dataset(lmdb_dir=args.lmdb_dir, args=args,
                                                  sample_file=args.test_key)
    # dataset_test_coco = dataset.COCODataset(args=args, is_training=False,
    #                                         sample_len=len(dataset_test_usc))
    # dataset_test = torch.utils.data.ConcatDataset(
    #     (dataset_test_usc, dataset_test_coco))

    dataset_test = dataset_test_usc

    train_data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4
    )
    test_data_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True,
        num_workers=2
    )

    # model
    model = models.BusterModel(out_channel=args.out_channel)
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
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True, threshold=0.1,
        min_lr=1e-7)

    list_loss = []

    if args.test:

        # for i, (X, _) in tqdm(enumerate(test_data_loader)):
        #     with torch.no_grad():
        #         _ = model(X.to(device))
        #     if i > 5:
                # break

        test(test_data_loader, model, args, iteration, device,
             logger=None, num=5)
    else:
        for ep in tqdm(range(init_ep, args.max_epoch)):
            # train
            for ret in train_data_loader:
                X, Y = ret
                loss = train(X, Y, model, optimizer, args,
                             iteration, device, logger=logger)
                list_loss.append(loss)
                iteration += 1

                if iteration % 300 == 0:
                    scheduler.step(np.mean(list_loss))
                    list_loss = []

                    test(test_data_loader, model, args, iteration, device,
                         logger=logger, num=10)
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
