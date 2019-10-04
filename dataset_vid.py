from __future__ import print_function
from pathlib import Path
import os
import skimage
import numpy as np
import torch
import utils
import utils_data
from pathlib import Path
import os
import sys
import pickle
import cv2
import skimage
from skimage import io
import numpy as np
import torch
from torchvision import transforms
from collections import defaultdict


def add_sorp(im, type="pepper"):
    im = skimage.img_as_float32(im)
    if type == "pepper":
        mask = np.ones(im.shape[:2], dtype=np.float32)
        mask = skimage.util.random_noise(mask, mode=type)
        if mask.shape != im.shape:
            mask = mask.reshape(im.shape)
        im = im * mask
    elif type == "salt":
        mask = np.zeros(im.shape[:2], dtype=np.float32)
        mask = skimage.util.random_noise(mask, mode=type)
        im = im.copy()
        if len(im.shape) > 2:
            im[mask > 0] = (1, 1, 1)
        else:
            im[mask > 0] = 1

    return im


def get_boundary(im):
    kernel = np.ones((5, 5), dtype=np.float32)
    im_bnd = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernel)
    return im_bnd


class Dataset_image:
    """class for dataset of image manipulation
    """

    def __init__(self, args=None, transform=None, videoset=None):
        # args contain necessary argument
        self.args = args
        if videoset is None:
            self.videoset = args.dataset
        else:
            self.videoset = videoset
        self.data_root = Path(args.root) / (self.videoset + "_tempered")
        self.data_root = self.data_root.expanduser()
        assert self.data_root.exists()

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform

        self.mask_root = self.data_root / "gt_mask"
        self.gt_root = self.data_root / "gt"
        self.im_mani_root = self.data_root / "vid"
        self._parse_all_images_with_gt()
        self._parse_images_with_copy_src()

    def split_train_test(self):
        ind = np.arange(len(self.data))
        np.random.shuffle(ind)
        ind_unto = int(len(self.data) * self.args.split)
        self.train_index = ind[:ind_unto]
        self.test_index = ind[ind_unto:]

    def _parse_all_images_with_gt(self):
        self.__im_files_with_gt = []
        self.data = []
        for i, name in enumerate(self.im_mani_root.iterdir()):
            info = {"name": name.name, "files": []}
            for _file in name.iterdir():
                if _file.suffix == ".png":
                    im_file = str(_file)
                    mask_file = os.path.join(
                        str(self.mask_root), name.name, (_file.stem + ".png")
                    )
                    if not os.path.exists(mask_file):
                        mask_file = os.path.join(
                            str(self.mask_root), name.name, (_file.stem + ".jpg")
                        )
                    try:
                        assert os.path.exists(mask_file)
                    except AssertionError:
                        raise FileNotFoundError(f"{mask_file} not found")
                    info["files"].append((im_file, mask_file))
                    self.__im_files_with_gt.append((i, im_file, mask_file))
            self.data.append(info)
        self.split_train_test()

    def randomize_mask(self, im):
        rand = np.random.randint(1, 3)
        kernel = np.ones((5, 5))
        if rand == 1:  # erosion
            im = cv2.erode(im, kernel)
        elif rand == 2:  # dilation
            im = cv2.dilate(im, kernel)
        return im

    def _parse_images_with_copy_src(self):
        Dict = defaultdict(lambda: [None, None, None])  # (i, forged, src)
        for i_d, D in enumerate(self.data):
            name = D["name"]
            gt_file = os.path.join(str(self.gt_root), Path(name).name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = sorted(list(data.keys()), key=lambda x: int(x.stem))
            offset = data[filenames[0]]["offset"]

            for i, cur_file in enumerate(filenames):
                cur_data = data[cur_file]
                mask_orig = cur_data["mask_orig"]
                mask_new = cur_data["mask_new"]

                fname = os.path.join(self.im_mani_root, *cur_file.parts[-2:])

                Dict[fname][0] = i_d
                fmask = os.path.join(self.mask_root, *cur_file.parts[-2:])
                Dict[fname][1] = fmask

                if mask_new is not None:
                    orig_file = filenames[i - offset]
                    forig = os.path.join(self.im_mani_root, *orig_file.parts[-2:])
                    Dict[forig][2] = fmask

        self.__im_file_with_src_copy = []
        for fp in Dict:
            iv, fmask, forig = Dict[fp]
            self.__im_file_with_src_copy.append((iv, fp, fmask, forig))

        self.__im_file_with_src_copy = []
        for fp in Dict:
            iv, fmask, forig = Dict[fp]
            self.__im_file_with_src_copy.append((iv, fp, fmask, forig))

    def load_videos_all(self, is_training=False, shuffle=True, to_tensor=True):
        if is_training:
            idx = self.train_index
        else:
            idx = self.test_index

        if shuffle:
            np.random.shuffle(idx)

        for ind in idx:
            D = self.data[ind]
            name = D["name"]
            # files = D['files']
            gt_file = os.path.join(str(self.gt_root), Path(name).name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = sorted(list(data.keys()), key=lambda x: int(x.stem))
            offset = data[filenames[0]]["offset"]

            _len = len(filenames)
            X = np.zeros((_len, *self.args.size, 3), dtype=np.float32)
            Y_forge = np.zeros((_len, *self.args.size), dtype=np.float32)
            Y_orig = np.zeros((_len, *self.args.size), dtype=np.float32)

            flag = False
            forge_time = None
            gt_time = None

            if is_training:
                # other_tfm = utils.SimTransform(
                #     size=self.args.size)
                other_tfm = None
            else:
                other_tfm = None

            for i, cur_file in enumerate(filenames):
                cur_data = data[cur_file]
                mask_orig = cur_data["mask_orig"]
                mask_new = cur_data["mask_new"]
                offset = cur_data["offset"]

                if mask_orig is not None and not flag:
                    forge_time = [i, -1]
                    gt_time = [i - offset, -1]
                    flag = True

                if mask_orig is None and flag:
                    gt_time[1] = i
                    forge_time[1] = i - offset

                fname = os.path.join(self.im_mani_root, *cur_file.parts[-2:])
                im = skimage.img_as_float32(io.imread(fname))

                X[i] = cv2.resize(im, self.args.size)
                if mask_new is None:
                    mask_new = np.zeros(
                        self.args.size, dtype=np.float32
                    )
                    mask_orig = np.zeros(
                        self.args.size, dtype=np.float32
                    )
                Y_forge[i] = (
                    cv2.resize(
                        mask_new.astype(np.float32), self.args.size
                    )
                    > 0.5
                )
                Y_orig[i - offset] = (
                    cv2.resize(
                        mask_orig.astype(np.float32), self.args.size
                    )
                    > 0.5
                )

            if forge_time is not None and forge_time[1] == -1:
                forge_time[1] = i
                gt_time[1] = i - offset
            if to_tensor:
                X, Y_forge = utils.custom_transform_images(
                    X, Y_forge, size=self.args.size, other_tfm=other_tfm
                )
                _, Y_orig = utils.custom_transform_images(
                    None, Y_orig, size=self.args.size, other_tfm=other_tfm
                )

            yield X, Y_forge, forge_time, Y_orig, gt_time, name