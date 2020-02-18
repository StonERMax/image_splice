import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

# custom module
import config
import models_custom as models
from train import train

from test import test
import dataset


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

    if args.model in ("dmac", "dmvn"):
        from test import test_dmac as test
        from train import train_dmac as train

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model
    model = models.CustomModel(out_channel=args.out_channel)
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
        patience=10,
        verbose=True,
        threshold=0.1,
        min_lr=1e-7,
    )

    # load dataset
    data_test = dataset.Dataset_COCO_CISDL(
        args, mode=args.mode, is_training=False
    )

    if args.test:
        # if not args.eval_bn:
        #     with torch.no_grad():
        #         for i, ret in enumerate(data_test.load()):
        #             Xs, Xt, Ys, Yt, labels = ret
        #             Xs, Xt = (Xs.to(device), Xt.to(device))
        #             _ = model(Xs, Xt)
        #             if i > 5:
        #                 break
        test(
            data_test,
            model,
            args,
            iteration=None,
            device=device,
            logger=None,
            num=50,
            plot=args.plot,
        )
        logger.close()
        raise SystemExit

        # TODO: There is a discrepency between test loss and train loss,
        # even when same dataset is used. Check out why!

    data_train = dataset.Dataset_COCO_CISDL(args, mode=None, is_training=True)

    list_loss = []

    for ep in tqdm(range(init_ep, args.max_epoch)):
        # train
        for ret in data_train.load():
            loss = train(
                ret, model, optimizer, args, iteration, device, logger=logger
            )
            list_loss.append(loss)
            iteration += 1

            if iteration % 300 == 0:
                scheduler.step(np.mean(list_loss))
                list_loss = []

                test(
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
                    os.path.join("./ckpt_scratch", model_name + ".pkl"),
                )

                print(f"weight saved in {model_name}.pkl")

    logger.close()