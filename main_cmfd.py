import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

# custom module
import config_cmfd
import models_cmfd
import train

import test
import dataset_cmfd


def set_grad_false(model, layer_ex=[]):
    for name, param in model.named_parameters():
        flag = True
        for each_l in layer_ex:
            if name.startswith(each_l):
                flag = False
                break
        if flag:
            param.requires_grad = False
        else:
            param.requires_grad = True


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config_cmfd.config()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + args.dataset + "_" + args.mode + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model
    if args.mode == "both":
        model = models_cmfd.DOAModel()
    elif args.mode == "mani":
        model = models_cmfd.DOAModel_man()
    else:
        model = models_cmfd.DOAModel_sim()

    model.to(device)
    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        if args.tune:
            set_grad_false(model, ["head_mask"])

    model_params = model.parameters()

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True, threshold=0.1, min_lr=1e-7
    )

    # load dataset test
    if args.dataset == "usc":    
        dataset_test = dataset_cmfd.USCISI_CMD_Dataset(
            args=args, is_training=False
        )
    elif args.dataset == "casia":
        dataset_test = dataset_cmfd.Dataset_CASIA(args)
    elif args.dataset == "como":
        pass
    elif args.dataset == "tifs":
        dataset_test = dataset_cmfd.Dataset_tifs(args)
    elif args.dataset == "grip":
        dataset_test = dataset_cmfd.Dataset_grip(args)
    elif args.dataset == "wwt":
        dataset_test = dataset_cmfd.Dataset_wwt(args)

    data_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    if args.test:
        with torch.no_grad():
            for i, ret in enumerate(data_test):
                Xs, Xt, Ys, Yt, labels = ret
                Xs, Xt = (Xs.to(device), Xt.to(device))
                _ = model(Xs, Xt)
                if i > 5:
                    break
        test.test_cmfd(
            data_test,
            model,
            args,
            iteration=None,
            device=device,
            logger=None,
            num=args.num,
            plot=args.plot,
        )
        logger.close()
        raise SystemExit

    # load dataset train
    if args.dataset == "usc":
        dataset_train = dataset_cmfd.USCISI_CMD_Dataset(
            args=args, is_training=True
        )
    elif args.dataset == "casia":
        dataset_train = dataset_cmfd.Dataset_CASIA(args)
    elif args.dataset == "como":
        pass
    elif args.dataset == "tifs":
        dataset_train = dataset_cmfd.Dataset_tifs(args)
    elif args.dataset == "grip":
        dataset_train = dataset_cmfd.Dataset_grip(args)
    elif args.dataset == "wwt":
        dataset_train = dataset_cmfd.Dataset_wwt(args)

    data_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    if not os.path.exists("ckpt_cmfd"):
        os.mkdir("ckpt_cmfd")

    list_loss = []

    for ep in tqdm(range(init_ep, args.max_epoch)):
        # train
        for ret in data_train:
            loss = train.train_cmfd(ret, model, optimizer, args, iteration, device, logger=logger)
            list_loss.append(loss)
            iteration += 1

            if iteration % 100 == 0:
                scheduler.step(np.mean(list_loss))
                list_loss = []

                test.test_cmfd(
                    data_test,
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
                    os.path.join("./ckpt_cmfd", model_name + ".pkl"),
                )

                print(f"weight saved in {model_name}.pkl")

    logger.close()
