import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

"""
train test on video CMFD dataset
"""
import config
import models_vid
from train import train_temporal

from test import test_temporal
import dataset_vid


def load_model(model, state):
    mod_state = {}
    name_req_grad = []

    for k in state:
        k = str(k)
        if not k.startswith("head_mask"):
            mod_state[k] = state[k]
            name_req_grad.append(k)
    model.load_state_dict(mod_state, strict=False)

    for name, param in model.named_parameters():
        if name in name_req_grad:
            param.requires_grad = False


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_video_temporal()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = "temporal_" + args.model + "_" + args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model
    model = models_vid.DOAModel(out_channel=args.out_channel)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        if args.tune:
            load_model(model, checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"], strict=False)

    # model_params = filter(lambda x: x.requires_grad, model.parameters())
    model_params = model.parameters()

    model.to(device)

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=10,
        verbose=True,
        threshold=0.1,
        min_lr=1e-7,
    )

    # load dataset

    data_test = dataset_vid.Dataset_vid(args, is_training=False)
    if args.test:
        # with torch.no_grad():
        #     for i, ret in enumerate(data_test.load_temporal(t_t_max=5, batch_size=5)):
        #         Xs, Xt, Ys, Yt, labels = ret
        #         Xs, Xt = Xs.to(device), Xt.to(device)
        #         _ = model(Xs, Xt)
        #         if i > 5:
        #             break
        test_temporal(
            data_test,
            model,
            args,
            iteration=None,
            device=device,
            logger=None,
            num=5,
            plot=True,
        )
        logger.close()
        raise SystemExit

    data_train = dataset_vid.Dataset_vid(args, is_training=True)

    list_loss = []

    for ep in tqdm(range(init_ep, args.max_epoch)):
        # train
        for ret in data_train.load_temporal():
            loss = train_temporal(
                ret, model, optimizer, args, iteration, device, logger=logger
            )
            list_loss.append(loss)
            iteration += 1

            if iteration % 100 == 0:
                scheduler.step(np.mean(list_loss))
                list_loss = []

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


                test_temporal(
                    data_test,
                    model,
                    args,
                    iteration=None,
                    device=device,
                    logger=None,
                    num=5,
                )

    logger.close()