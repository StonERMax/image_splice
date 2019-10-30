"""

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

from test import test_det_vid
import dataset_vid


def load_state_base(model, state):
    new_state = {}
    for k in state:
        k = str(k)
        if k.startswith("base"):
            new_k = k.replace("base.", "")
            new_state[new_k] = state[k]
    model.load_state_dict(new_state)
    return model


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
    model_name = "detseg_" + args.model + "_" + args.dataset + args.suffix

    if args.model in ("dmac", "dmvn"):
        from test import test_dmac as test
        from train import train_dmac as train

    print(f"Model Name: {model_name}")

    # logger
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    logger = SummaryWriter("./logs/" + model_name)

    # model

    model = models.Base_DetSegModel()
    
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        load_state_base(model, checkpoint['model_state'])

    # model = model.base
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    # load dataset
    data_test = dataset_vid.Dataset_vid(args, is_training=False)

    # with torch.no_grad():
    #     for i, ret in enumerate(data_test.load_mani()):
    #         X, *_ = ret
    #         X = X.to(device)
    #         _ = model(X)
    #         if i > 30:
    #             break
    test_det_vid(
        data_test,
        model,
        args,
        iteration=None,
        device=device,
        logger=None,
        num=None,
        plot=args.plot,
    )
    logger.close()
    raise SystemExit

