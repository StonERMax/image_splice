"""
matching two patches
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
from train import train_patchmatch_vid

from test import test_patchmatch_vid
import dataset_vid


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
    model_name = "patch_" + args.model + "_" + args.dataset + args.suffix

    if args.model in ("dmac", "dmvn"):
        from test import test_dmac as test
        from train import train_dmac as train

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model

    model = models.PatchMatch()
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model_params = model.parameters()

    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.1,
        patience=5,
        verbose=True,
        threshold=0.1,
        min_lr=1e-7,
    )

    # load dataset
    data_test = dataset_vid.Dataset_vid(args, is_training=False)

    if args.test:
        # with torch.no_grad():
        #     for i, ret in enumerate(data_test.load_template(t_t_max=5, batch_size=5)):
        #         X1, X2, *_ = ret
        #         _ = model(X1.to(device), X2.to(device))
        #         if i > 5:
        #             break
        test_patchmatch_vid(
            data_test,
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

    data_train = dataset_vid.Dataset_vid(args, is_training=True)

    for ep in tqdm(range(init_ep, args.max_epoch)):
        # train
        for ret in data_train.load_template():
            train_patchmatch_vid(
                ret, model, optimizer, args, iteration, device, logger=logger
            )
            iteration += 1

            if iteration % 20 == 0:
                loss = test_patchmatch_vid(
                    data_test,
                    model,
                    args,
                    iteration=None,
                    device=device,
                    logger=None,
                    num=5,
                )
                scheduler.step(loss)

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
