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

from test import test_casia_det
import dataset


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.config_casia()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # model name
    model_name = args.model + "_" + args.dataset + args.suffix

    print(f"Model Name: {model_name}")

    data_test = dataset.Dataset_casia_det(args)

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    if args.model in ("dmac", "dmvn"):
        model = models.get_dmac(args.model, pretrain=True)
    else:
        model = models.DOAModel(out_channel=args.out_channel)

    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    data_loader = torch.utils.data.DataLoader(
        data_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    with torch.no_grad():
        for i, ret in enumerate(data_loader):
            Xs, Xt, labels = ret
            Xs, Xt = (Xs.to(device), Xt.to(device))
            _ = model(Xs, Xt)
            if i > 5:
                break
    test_casia_det(
        data_loader,
        model,
        args,
        iteration=None,
        device=device,
        logger=None,
        num=args.num,
        plot=args.plot,
    )
    logger.close()