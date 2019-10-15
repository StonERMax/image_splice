import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn.functional as F

# custom module
import config
import models

from test import test_dmac as test
import dataset_vid

from loss import BCE_loss


def tval(x):
    if not isinstance(x, torch.Tensor):
        return x
    if x.is_cuda:
        return x.data.cpu().numpy()
    else:
        return x.data.numpy()


def train(D, model, optimizer, args, iteration, device, logger=None):
    module = model.module if isinstance(model, nn.DataParallel) else model
    module.train()

    Xs, Xt, Ys, Yt, labels = D
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(np.array(labels, dtype=np.float32))
    labels = labels.float().to(device)
    Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device), Yt.to(device)

    predt, preds, pred_det = model(Xt, Xs)

    criterion = nn.NLLLoss().cuda(device)

    log_op = F.log_softmax(predt, dim=1)
    log_oq = F.log_softmax(preds, dim=1)

    Yq = Ys.squeeze(1).long()
    Yp = Yt.squeeze(1).long()

    loss_p = criterion(log_op, Yp)
    loss_q = criterion(log_oq, Yq)

    loss = loss_p + loss_q

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(
        f"{iteration:5d}: f(probe+donor+det): {tval(loss_p):.4f} + "
        + f"{tval(loss_q):.4f}"
    )

    if logger is not None:
        logger.add_scalar("train_loss/total", loss, iteration)

    return loss_val


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_video()
    args.model = "dmac"

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
    model = models.get_dmac()
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model_params = model.parameters()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

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
        with torch.no_grad():
            for i, ret in enumerate(data_test.load()):
                Xs, Xt, Ys, Yt, labels = ret
                Xs, Xt = (Xs.to(device), Xt.to(device))
                _ = model(Xs, Xt)
                if i > 5:
                    break
        test(
            data_test,
            model,
            args,
            iteration=None,
            device=device,
            logger=None,
            num=5,
            plot=args.plot,
        )
        logger.close()
        raise SystemExit

    data_train = dataset_vid.Dataset_vid(args, is_training=True)

    list_loss = []

    for ep in tqdm(range(init_ep, args.max_epoch)):
        # train
        for ret in data_train.load():
            loss = train(
                ret, model, optimizer, args, iteration, device, logger=logger
            )
            list_loss.append(loss)
            iteration += 1

            if iteration % 100 == 0:
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
                    os.path.join("./ckpt", model_name + ".pkl"),
                )

                print(f"weight saved in {model_name}.pkl")

    logger.close()
