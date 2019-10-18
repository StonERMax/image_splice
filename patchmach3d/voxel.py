import os
from pathlib import Path
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import argparse
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import cv2
from PIL import Image
from scipy.io import loadmat
from skimage.transform import resize as imresize
# from scipy.misc import imresize


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)



if __name__ == "__main__":


    root = Path("./tmp2_utube")

    scale = 1 / 10

    color1 = (128, 0, 0)
    color2 = (0, 128, 0)

    for each_pkl in tqdm(root.iterdir()):
        if each_pkl.suffix != '.mat':
            continue
        print(each_pkl)
        X = loadmat(str(each_pkl))['map']
        X = X.transpose((2, 0, 1)).astype(np.float)
        T = X.shape[0]
        ch, r, c = X.shape
        nr, nc = 30, 30
        X_down = np.zeros((ch, nr, nc), dtype=X.dtype)

        # for i in range(T):
        X1 = imresize(X, (T, nr, nc))


        # X1 = X_down
        # ind = np.random.choice(T, size=min(T, 60), replace=False)
        ind = np.arange(T)
        ind.sort()

        X1 = X1[ind]

        X1 = X1.transpose(2, 0, 1)

        Colors1 = np.zeros(X1.shape + (4,), dtype=np.float)


        Colors1[X1 > 0] = (0, 0, 1, 0.5)

        print(X1.shape)

        scale_x, scale_y, scale_z = (0.6, 1.7, 0.6)

        fig = plt.figure(figsize=(8,6))
        ax = fig.gca(projection="3d")

        ax.get_proj = lambda: np.dot(
            Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1])
        )

        ax.voxels(X1, facecolors=Colors1, label="source")
        # ax.legend()

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_xlabel("x")
        ax.view_init(elev=24, azim=-52)
        _dir = 'fig'       
        if not os.path.exists(_dir):
            os.mkdir(_dir)

        plt.savefig("{}/{}.png".format(_dir, each_pkl.stem))
        plt.close("all")
        #