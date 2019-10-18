import sys

sys.path.append("..")

import dataset_vid
import numpy as np
import skimage
import config
from pathlib import Path
import matplotlib.pyplot as plt

args = config.config_video()
np.random.seed(args.seed)

data = dataset_vid.Dataset_vid(args, is_training=False)

DIR = Path("./out/intro")

for i, ret in enumerate(data.load_videos_all(to_tensor=False)):
    X, Y_forge, forge_time, Y_orig, gt_time, name = ret

    path = DIR / name
    path.mkdir(exist_ok=True, parent=True)

    for i in range(X.shape[0]):
        im  = X[i]
        y = np.zeros(im.shape)
        y[..., 1] = Y_orig[i]
        y[..., 0] = Y_forge[i]

        skimage.io.imsave(
            path / f"{i}.jpg", im
        )
        skimage.io.imsave(
            path / f"{i}_gt.jpg", y
        )
    break
