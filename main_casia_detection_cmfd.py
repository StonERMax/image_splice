import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from matplotlib import pyplot as plt

# custom module
import config
import models_cmfd

from test import test_casia_cmfd
import dataset_cmfd


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

    data_test = dataset_cmfd.Dataset_casia(args)

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

    data_loader = torch.utils.data.DataLoader(
        data_test, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    with torch.no_grad():
        for i, ret in enumerate(data_loader):
            X, Y = ret
            _ = model(X.to(device), X.to(device))
            if i > 5:
                break
    test_casia_cmfd(
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
