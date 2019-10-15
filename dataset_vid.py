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
import pandas as pd


def get_boundary(im):
    kernel = np.ones((5, 5), dtype=np.float32)
    im_bnd = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernel)
    return im_bnd


class Dataset_vid(torch.utils.data.Dataset):
    """class for dataset of image manipulation
    """

    def __init__(
        self, args=None, transform=None, videoset=None, is_training=True
    ):
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
            self.transform = utils.CustomTransform(size=args.size)
        else:
            self.transform = transform

        self.is_training = is_training

        self.mask_root = self.data_root / "gt_mask"
        self.gt_root = self.data_root / "gt"
        self.im_mani_root = self.data_root / "vid"
        self._parse_all_images_with_gt()
        # self._parse_images_with_copy_src()
        self._parse_source_target()

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
                            str(self.mask_root),
                            name.name,
                            (_file.stem + ".jpg"),
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

    def _parse_source_target(self):
        _list = []
        for ind in range(len(self.data)):
            D = self.data[ind]
            name = D["name"]
            gt_file = os.path.join(str(self.gt_root), Path(name).name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = sorted(list(data.keys()), key=lambda x: int(x.stem))
            offset = data[filenames[0]]["offset"]

            for i, cur_file in enumerate(filenames):
                if i < offset:
                    continue
                fname_target = os.path.join(
                    self.im_mani_root, *cur_file.parts[-2:]
                )
                if self.videoset == "davis":
                    _fsrc = f"{int(cur_file.stem)-offset:d}.png"
                else:
                    _fsrc = f"{int(cur_file.stem)-offset:05d}.png"
                fname_src = os.path.join(
                    self.im_mani_root,
                    cur_file.parts[-2],
                    _fsrc,
                )

                assert os.path.exists(fname_target)
                assert os.path.exists(fname_src)

                fname_mask = os.path.join(self.mask_root, *cur_file.parts[-2:])

                _list.append((ind, fname_src, fname_target, fname_mask))

        self.im_files_with_src_target = pd.DataFrame(
            data=_list, columns=["id", "src", "target", "mask"]
        )

    def __len__(self):
        if self.is_training:
            idx = self.train_index
        else:
            idx = self.test_index

        # get number of df for this id
        df = self.im_files_with_src_target
        return df[df["id"].isin(idx)].shape[0]

    def get_im(self, fp, to_tensor=True, is_mask=False):
        im = skimage.img_as_float32(skimage.io.imread(str(fp)))
        if is_mask:
            interp = 0
        else:
            interp = 1
        im = cv2.resize(im, self.args.size, interpolation=interp)

        if to_tensor:
            if is_mask:
                _, im = self.transform(mask=im)
            else:
                im, _ = self.transform(im)
        return im

    def __getitem__(self, index, to_tensor=True):
        if self.is_training:
            idx = self.train_index
        else:
            idx = self.test_index

        df = self.im_files_with_src_target
        df = df[df["id"].isin(idx)]
        row = df.iloc[index]
        fsrc = row["src"]
        ftar = row["target"]
        fmask = row["mask"]

        im_s = self.get_im(fsrc, to_tensor=to_tensor)
        im_t = self.get_im(ftar, to_tensor=to_tensor)
        im_mask = self.get_im(fmask, is_mask=True, to_tensor=to_tensor)
        if to_tensor:
            mask_s = im_mask[[0]]
            mask_t = im_mask[[-1]]
        else:
            mask_s = im_mask[..., 0]
            mask_t = im_mask[..., -1]

        if mask_s.sum() * mask_t.sum() > 0:
            label = 1.0
        else:
            label = 0.0
        return im_s, im_t, mask_s, mask_t, label

    def load(self, shuffle=True, batch_size=None):
        if batch_size is None:
            batch_size = self.args.batch_size
        loader = torch.utils.data.DataLoader(
            self, batch_size=batch_size, num_workers=4, shuffle=True
        )
        return loader

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

                X[i] = cv2.resize(
                    im, self.args.size, interpolation=cv2.INTER_LINEAR
                )
                if mask_new is None:
                    mask_new = np.zeros(self.args.size, dtype=np.float32)
                    mask_orig = np.zeros(self.args.size, dtype=np.float32)
                Y_forge[i] = (
                    cv2.resize(
                        mask_new.astype(np.float32),
                        self.args.size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0.05
                )
                Y_orig[i - offset] = (
                    cv2.resize(
                        mask_orig.astype(np.float32),
                        self.args.size,
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0.05
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

    def _load(self, ret, to_tensor=True, batch=None, is_training=True):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret
        if forge_time is None:
            return None

        forge_time = np.arange(forge_time[0], forge_time[1] + 1)
        gt_time = np.arange(gt_time[0], gt_time[1] + 1)

        if batch is None:
            batch_size = len(forge_time)
        else:
            batch_size = min(self.args.batch_size, len(forge_time))
            ind = np.arange(forge_time.size)
            np.random.shuffle(ind)
            forge_time = forge_time[ind]
            gt_time = gt_time[ind]

        Xref = np.zeros((batch_size, *self.args.size, 3), dtype=np.float32)
        Xtem = np.zeros((batch_size, *self.args.size, 3), dtype=np.float32)
        Yref = np.zeros((batch_size, *self.args.size), dtype=np.float32)
        Ytem = np.zeros((batch_size, *self.args.size), dtype=np.float32)

        for k in range(batch_size):
            ind_forge = forge_time[k]
            ind_orig = gt_time[k]

            im_orig = X[ind_orig]
            im_forge = X[ind_forge]

            mask_ref = np.zeros(im_orig.shape[:-1], dtype=np.float32)
            mask_tem = np.zeros(im_orig.shape[:-1], dtype=np.float32)

            # mask_ref[Y_forge[ind_orig] > 0.5] = 0.5
            mask_ref[Y_orig[ind_orig] > 0.5] = 1

            # mask_tem[Y_orig[ind_forge] > 0.5] = 0.5
            mask_tem[Y_forge[ind_forge] > 0.5] = 1

            im_f = skimage.transform.resize(
                im_forge, self.args.size, order=1, anti_aliasing=False
            )
            im_o = skimage.transform.resize(
                im_orig, self.args.size, order=1, anti_aliasing=False
            )
            mask_ref = skimage.transform.resize(
                mask_ref, self.args.size, order=0, anti_aliasing=False
            )
            mask_tem = skimage.transform.resize(
                mask_tem, self.args.size, order=0, anti_aliasing=False
            )

            Xref[k] = im_o  # * (1 - (mask_ref == 0.5)
            #   [..., None]).astype(im_o.dtype)
            Xtem[k] = im_f  # * ((mask_tem == 1)[..., None]).astype(im_f.dtype)
            Yref[
                k
            ] = mask_ref  # * (1 - (mask_ref == 0.5)).astype(mask_ref.dtype)
            Ytem[k] = mask_tem  # * (mask_tem == 1).astype(mask_tem.dtype)

        if to_tensor:
            tfm_o = utils.CustomTransform(self.args.size)
            tfm_f = utils.CustomTransform(self.args.size)
            if is_training:
                # other_tfm = utils.SimTransform(size=self.args.size)
                other_tfm = None
            else:
                other_tfm = None
            Xreft = torch.zeros(batch_size, 3, *self.args.size)
            Xtemt = torch.zeros(batch_size, 3, *self.args.size)
            Yreft = torch.zeros(batch_size, 1, *self.args.size)
            Ytemt = torch.zeros(batch_size, 1, *self.args.size)
            for k in range(batch_size):
                Xreft[k], Yreft[k] = tfm_o(
                    Xref[k], Yref[k], other_tfm=other_tfm
                )
                Xtemt[k], Ytemt[k] = tfm_f(
                    Xtem[k], Ytem[k], other_tfm=other_tfm
                )
            Xref, Xtem, Yref, Ytem = Xreft, Xtemt, Yreft, Ytemt

            Ytem[Ytem > 0.5] = 1
            Ytem[Ytem <= 0.5] = 0
            # Ytem[(Ytem > 0.2) & (Ytem <= 0.8)] = 0.5

            Yref[Yref > 0.5] = 1
            Yref[Yref <= 0.5] = 0
            # Yref[(Yref > 0.2) & (Yref <= 0.8)] = 0.5

        return Xref, Xtem, Yref, Ytem, name

    def load_data_template_match_pair(
        self, to_tensor=True, is_training=True, batch=None
    ):
        def mixer(in1, in2, lib):
            if lib == np:
                fn_cat = np.concatenate
            else:
                fn_cat = torch.cat
            Xref1, Xtem1, Yref1, Ytem1, name1 = in1
            Xref2, Xtem2, Yref2, Ytem2, name2 = in2
            Xref = fn_cat((Xref1, Xref2), 0)
            Xtem = fn_cat((Xtem1, Xtem2), 0)
            Yref = fn_cat((Yref1, Yref2), 0)
            Ytem = fn_cat((Ytem1, Ytem2), 0)
            name = name1 + "_" + name2
            total = Xref1.shape[0] + Xref2.shape[0]
            ind_rand = np.random.choice(
                total, size=Xref1.shape[0], replace=False
            )
            return (
                Xref[ind_rand],
                Xtem[ind_rand],
                Yref[ind_rand],
                Ytem[ind_rand],
                name,
            )

        loader = self.load_videos_all(is_training=is_training, to_tensor=False)
        while True:
            try:
                ret1 = next(loader)
                ret2 = next(loader)
            except StopIteration:
                return
            dat1 = self._load(
                ret1, to_tensor=to_tensor, batch=batch, is_training=is_training
            )
            dat2 = self._load(
                ret2, to_tensor=to_tensor, batch=batch, is_training=is_training
            )

            if dat1 is None:
                continue

            Xref1, Xtem1, Yref1, Ytem1, name1 = dat1

            yield dat1

            if dat2 is None:
                continue

            Xref2, Xtem2, Yref2, Ytem2, name2 = dat2
            yield dat2

            # if is_training and np.random.rand() > 0.8:
            #     dat = (
            #         Xref1,
            #         torch.zeros_like(Xtem1),
            #         torch.zeros_like(Yref1),
            #         torch.zeros_like(Ytem1),
            #         name1 + "_0",
            #     )
            #     yield mixer(dat1, dat, torch)
            # # mix both
            # if is_training and np.random.rand() > 0.5:
            #     if to_tensor:
            #         lib = torch
            #     else:
            #         lib = np

            #     ind2 = np.random.choice(
            #         np.arange(Xref2.shape[0]), size=Xref1.shape[0]
            #     )

            #     Xtem_d = copy.deepcopy(Xtem2[ind2])
            #     Yref_d = lib.zeros_like(Yref1)
            #     Ytem_d = lib.zeros_like(Yref1)
            #     name = name1 + "_" + name2

            #     dat3 = Xref1, Xtem_d, Yref_d, Ytem_d, name

            #     yield mixer(dat1, dat3, lib)
