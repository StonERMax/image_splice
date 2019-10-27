from __future__ import print_function
from pathlib import Path
import os
import skimage
import numpy as np
import torch
import utils
import utils_data


class Dataset_COCO_CISDL(torch.utils.data.Dataset):
    def __init__(self, args, mode=None, is_training=True, test_fore_only=True, no_back=False):
        """ mode should be ['easy', 'medi', 'diff']
            default: None (all leels)
        """
        self.args = args
        subpath_list = []

        args.root = os.path.join(
            args.data_root, "train2014" if is_training else "val2014"
        )

        labelfiles_path = Path(args.root) / "labelfiles"

        if is_training:
            for each in labelfiles_path.iterdir():
                if each.suffix == ".csv":
                    if no_back:
                        if "_back_" in each.name:
                            continue
                    if mode is None:
                        subpath_list.append(each.name)
                    else:
                        each_level = each.stem.split("_")[-1]
                        if each_level == mode:
                            subpath_list.append(each.name)
        else:  # testing
            for each in labelfiles_path.iterdir():
                if each.suffix == ".csv":
                    if test_fore_only:
                        if "_fore_" not in each.name:
                            continue
                    if no_back:
                        if "_back_" in each.name:
                            continue
                    if mode is None:
                        subpath_list.append(each.name)
                    else:
                        each_level = each.stem.split("_")[-1]
                        if each_level == mode:
                            subpath_list.append(each.name)

        self.pair_list = utils_data.load_pairs(
            args.root, subpath_list, args.root
        )

        if args.model in ("dmac", "dmvn"):
            self.transform = utils.CustomTransform_vgg(size=args.size)
        else:
            self.transform = utils.CustomTransform(size=args.size)

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        piece = self.pair_list[idx]
        # flip_p = np.random.uniform(0, 1)
        img_temp = skimage.io.imread(piece[0])
        im1 = skimage.img_as_float32(img_temp)
        # im1 = utils_data.flip(img_temp, flip_p)

        img_temp = skimage.io.imread(piece[1])
        im2 = skimage.img_as_float32(img_temp)
        # im2 = utils_data.flip(img_temp, flip_p)

        label_tmp = int(piece[2])
        if label_tmp == 1:
            gt_temp = skimage.io.imread(piece[3])
            if len(gt_temp.shape) > 2:
                gt_temp = gt_temp[..., 0]
            gt1 = skimage.img_as_float32(gt_temp)
            # gt1 = utils_data.flip(gt_temp, flip_p)

            gt_temp = skimage.io.imread(piece[4])
            if len(gt_temp.shape) > 2:
                gt_temp = gt_temp[..., 0]
            gt2 = skimage.img_as_float32(gt_temp)
            # gt2 = utils_data.flip(gt_temp, flip_p)
        else:
            gt1 = np.zeros(im1.shape[:2], dtype=np.float32)
            gt2 = np.zeros(im1.shape[:2], dtype=np.float32)

        imt1, gtt1 = self.transform(im1, gt1)
        imt2, gtt2 = self.transform(im2, gt2)

        return imt1, imt2, gtt1, gtt2, label_tmp

    def load(self, batch_size=None, shuffle=True):
        bs = self.args.batch_size if batch_size is None else batch_size

        chunk = [
            np.arange(pos, pos + bs) for pos in range(0, len(self) - bs, bs)
        ]
        seq = np.arange(len(self))
        if shuffle:
            np.random.shuffle(seq)

        for _inds in chunk:
            inds = seq[_inds]
            im1s = []
            im2s = []
            gt1s = []
            gt2s = []
            labels = []

            for i in inds:
                im1, im2, gt1, gt2, lab = self[i]
                im1s.append(im1)
                im2s.append(im2)
                gt1s.append(gt1)
                gt2s.append(gt2)
                labels.append(lab)
            im1s = torch.stack(im1s, 0)
            im2s = torch.stack(im2s, 0)
            gt1s = torch.stack(gt1s, 0)
            gt2s = torch.stack(gt2s, 0)
            yield im1s, im2s, gt1s, gt2s, labels

    def load_mani(self, batch_size=None, shuffle=True):
        bs = self.args.batch_size if batch_size is None else batch_size
        chunk = [
            np.arange(pos, pos + bs) for pos in range(0, len(self) - bs, bs)
        ]
        seq = np.arange(len(self))
        if shuffle:
            np.random.shuffle(seq)

        for _inds in chunk:
            inds = seq[_inds]
            im = []
            labels = []
            segm = []

            for i in inds:
                _, im2, _, gt2, lab = self[i]
                im.append(im2)
                segm.append(gt2)
                labels.append(lab)
            im = torch.stack(im, 0)
            segm = torch.stack(segm, 0)
            yield im, segm, np.array(labels)