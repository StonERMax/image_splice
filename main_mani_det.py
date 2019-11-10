"""
forge detection for dmac-coco dataset
"""

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
from train import train_det

from test import test_det
import dataset
import dataset_cmfd


def tune_layer_params(model, layer_ex=[]):
    parameters = []
    for name, param in model.named_parameters():
        flag = True
        for each_l in layer_ex:
            if name.startswith(each_l):
                flag = False
                break
        if flag:
            parameters.append(param)
    return parameters


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
    model_name = "detseg_" + args.model + "_" + args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model

    model = models.DetSegModel()
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    if args.tune:
        model_params = tune_layer_params(model, layer_ex=["base.conv_det"])
    else:
        model_params = model.parameters()

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, threshold=0.1, min_lr=1e-7
    )

    # load dataset
    data_test_cisdl = dataset.Dataset_COCO_CISDL(
        args, mode=args.mode, is_training=False, test_fore_only=False, no_back=True
    )
    # dataset_usc_test = dataset_cmfd.USCISI_CMD_Dataset(
    #     args=args, is_training=False, sample_len=len(data_test_cisdl) // 2
    # )
    # dataset_test = torch.utils.data.ConcatDataset((data_test_cisdl, dataset_usc_test))
    dataset_test = data_test_cisdl
    data_test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    if args.test:
        with torch.no_grad():
            for i, ret in enumerate(data_test_loader):
                _, X, *_ = ret
                X = X.to(device)
                _ = model(X)
                if i > 5:
                    break
        test_det(
            data_test_loader,
            model,
            args,
            iteration=None,
            device=device,
            logger=None,
            num=10,
            plot=args.plot,
        )
        logger.close()
        raise SystemExit

    data_cisdl = dataset.Dataset_COCO_CISDL(args, mode=None, is_training=True, no_back=True)
    # dataset_usc = dataset_cmfd.USCISI_CMD_Dataset(
    #     args=args, is_training=True, sample_len=len(data_cisdl)
    # )

    # dataset_train = torch.utils.data.ConcatDataset((data_cisdl, dataset_usc))
    dataset_train = data_cisdl

    data_train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    list_loss = []

    for ep in tqdm(range(init_ep, args.max_epoch)):
        # train
        for ret in data_train_loader:
            ret = [ret[1], ret[3], ret[4]]
            loss = train_det(ret, model, optimizer, args, iteration, device, logger=logger)
            list_loss.append(loss)
            iteration += 1

            if iteration % 100 == 0:
                scheduler.step(np.mean(list_loss))
                list_loss = []

                test_det(
                    data_test_loader,
                    model,
                    args,
                    iteration=None,
                    device=device,
                    logger=None,
                    num=5,
                )

                state = (
                    model.module.state_dict()
                    if isinstance(model, nn.DataParallel)
                    else model.state_dict()
                )

                torch.save(
                    {"epoch": ep, "model_state": state},
                    os.path.join("./ckpt", model_name + ".pkl"),
                )

                print(f"weight saved in {model_name}.pkl")

    logger.close()
