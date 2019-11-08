from __future__ import print_function
from pathlib import Path
import os
import skimage
import numpy as np
import torch
import utils
import utils_data
from parse import parse
import pandas as pd
from tqdm import tqdm
import cv2


class Dataset_COCO_CISDL(torch.utils.data.Dataset):
    def __init__(
        self, args, mode=None, is_training=True, test_fore_only=True, no_back=True
    ):
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
                        if "_neg_" in each.name:
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

        self.pair_list = utils_data.load_pairs(args.root, subpath_list, args.root)

        if args.model in ("dmac", "dmvn"):
            self.transform = utils.CustomTransform_vgg(size=args.size)
        else:
            self.transform = utils.CustomTransform(size=args.size)
        print(f"data size : {len(self)}")

        self.is_training = is_training

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        piece = self.pair_list[idx]
        # flip_p = np.random.uniform(0, 1)
        img_temp = skimage.io.imread(piece[0])
        im1 = skimage.img_as_float32(img_temp)

        if self.is_training:
            flip_p = np.random.rand() > 0.8
        else:
            flip_p = 0

        img_temp = skimage.io.imread(piece[1])
        im2 = skimage.img_as_float32(img_temp)
        im2 = utils_data.flip(img_temp, flip_p)

        label_tmp = int(piece[2])
        if label_tmp == 1:
            gt_temp = skimage.io.imread(piece[3])
            if len(gt_temp.shape) > 2:
                gt_temp = gt_temp[..., 0]
            gt1 = skimage.img_as_float32(gt_temp)

            gt_temp = skimage.io.imread(piece[4])
            if len(gt_temp.shape) > 2:
                gt_temp = gt_temp[..., 0]
            gt2 = skimage.img_as_float32(gt_temp)
            gt2 = utils_data.flip(gt_temp, flip_p)
        else:
            gt1 = np.zeros(im1.shape[:2], dtype=np.float32)
            gt2 = np.zeros(im1.shape[:2], dtype=np.float32)

        imt1, gtt1 = self.transform(im1, gt1)
        imt2, gtt2 = self.transform(im2, gt2)

        return imt1, imt2, gtt1, gtt2, label_tmp

    def load(self, batch_size=None, shuffle=True):
        bs = self.args.batch_size if batch_size is None else batch_size

        chunk = [np.arange(pos, pos + bs) for pos in range(0, len(self) - bs, bs)]
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
        chunk = [np.arange(pos, pos + bs) for pos in range(0, len(self) - bs, bs)]
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
            yield im, segm, np.array(labels, dtype=np.float32)


class Dataset_casia(torch.utils.data.Dataset):
    def __init__(self, args=None, both=None):
        self.args = args
        self.transform = None
        self.transform = utils.CustomTransform(size=args.size)

        self.root = Path(os.environ["HOME"]) / "dataset" / "CMFD" / "CASIA"

        self.imroot = self.root / "CASIA2.0"
        self.gtroot = self.root / "GT"

        imnames = []
        src_names = []
        target_names = []
        gt_names = []

        au_files = sorted((self.imroot / "Au").glob("Au_*"))
        self.au_base_name = {x.stem: x.suffix for x in au_files}

        for efile in tqdm(sorted((self.imroot / "Tp").glob("Tp_D_*"))):
            if efile.suffix in (".bmp", ".tif", ".jpg", ".png"):
                src, forg = self.get_src_dest(efile.name)
                if src is None or forg is None:
                    continue
                gtfile = f"{efile.stem}_gt.png"
                im_file = self.imroot / "Tp" / efile
                gt_file = self.gtroot / gtfile
                src_file = self.imroot / "Au" / src
                target_file = self.imroot / "Au" / forg

                if (
                    im_file.exists()
                    and gt_file.exists()
                    # and src_file.exists()
                    and target_file.exists()
                ):
                    imnames.append(im_file)
                    gt_names.append(gt_file)
                    src_names.append(src_file)
                    target_names.append(target_file)
                else:
                    pass
        # pos_len = len(imnames)
        # if both is not None:
        #     for i, efile in enumerate((self.imroot / "Au").glob("Tp_S_*")):
        #         if efile.suffix in (".bmp", ".tif", ".jpg", ".png"):
        #             imnames.append(str(self.imroot / "Au" / efile))
        #             gt_names.append(None)
        #         if i >= pos_len:
        #             break
        self.df = pd.DataFrame(
            data={
                "imfile": imnames,
                "gt": gt_names,
                "src": src_names,
                "target": target_names,
            },
            dtype=str,
        )
        print(f"number of images {self.df.shape[0]}")

    def get_src_dest(self, fn):
        src, forg = fn.split("_")[-3:-1]
        #
        _type, num = parse("{:l}{}", src)
        src_base_name = "Au" + "_" + _type + "_" + num

        src_file = None
        if src_base_name in self.au_base_name:
            src_file = src_base_name + self.au_base_name[src_base_name]

        #
        _type, num = parse("{:l}{}", forg)
        forg_base_name = "Au" + "_" + _type + "_" + num

        forg_file = None
        if forg_base_name in self.au_base_name:
            forg_file = forg_base_name + self.au_base_name[forg_base_name]

        return src_file, forg_file

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index, im_only=False):
        row = self.df.loc[index]

        imfile = row["imfile"]
        gtfile = row["gt"]
        src_file = row["target"]

        imt = skimage.img_as_float32(skimage.io.imread(str(imfile))[:, :, :3])
        ims = skimage.img_as_float32(skimage.io.imread(str(src_file))[:, :, :3])
        label = 0

        if gtfile is not None:
            y = skimage.img_as_float32(skimage.io.imread(str(gtfile), as_gray=True))
            label = 1.
        else:
            y = np.zeros(ims.shape[:2], dtype=ims.dtype)
            label = 0
        ims, y = self.transform(ims, y)
        imt, _ = self.transform(imt)
        return ims, imt, y, label


class Dataset_casia_v1(torch.utils.data.Dataset):
    def __init__(self, args=None, both=None):
        self.args = args
        self.transform = None
        if args.model in ("dmac", "dmvn"):
            self.transform = utils.CustomTransform_vgg(size=args.size)
        else:
            self.transform = utils.CustomTransform(size=args.size)

        self.root = Path(os.environ["HOME"]) / "dataset" / "CMFD" / "CASIA_v1"

        self.imroot = self.root
        self.gtroot = self.root / "GT" / "Sp"

        imnames = []
        src_names = []
        target_names = []
        gt_names = []

        au_files = sorted((self.imroot / "Au").glob("Au_*"))
        self.au_base_name = {x.stem: x.suffix for x in au_files}

        for efile in tqdm(sorted((self.imroot/"Tp"/"Sp").glob("Sp_D_*"))):
            if efile.suffix in (".bmp", ".tif", ".jpg", ".png", ".JPG"):
                src, forg = self.get_src_dest(efile.name)
                if src is None or forg is None:
                    continue
                gtfile = f"{efile.stem}_gt.png"
                im_file = self.imroot / "Tp" / efile
                gt_file = self.gtroot / gtfile
                src_file = self.imroot / "Au" / src
                target_file = self.imroot / "Au" / forg

                if (
                    im_file.exists()
                    and gt_file.exists()
                    # and src_file.exists()
                    and target_file.exists()
                ):
                    imnames.append(im_file)
                    gt_names.append(gt_file)
                    src_names.append(src_file)
                    target_names.append(target_file)
                else:
                    pass
        # pos_len = len(imnames)
        # if both is not None:
        #     for i, efile in enumerate((self.imroot / "Au").glob("Tp_S_*")):
        #         if efile.suffix in (".bmp", ".tif", ".jpg", ".png"):
        #             imnames.append(str(self.imroot / "Au" / efile))
        #             gt_names.append(None)
        #         if i >= pos_len:
        #             break
        self.df = pd.DataFrame(
            data={
                "imfile": imnames,
                "gt": gt_names,
                "src": src_names,
                "target": target_names,
            },
            dtype=str,
        )
        print(f"number of images {self.df.shape[0]}")

    def get_src_dest(self, fn):
        src, forg = fn.split("_")[-3:-1]
        #
        _type, num = parse("{:l}{}", src)
        src_base_name = "Au" + "_" + _type + "_" + num

        src_file = None
        if src_base_name in self.au_base_name:
            src_file = src_base_name + self.au_base_name[src_base_name]

        #
        _type, num = parse("{:l}{}", forg)
        forg_base_name = "Au" + "_" + _type + "_" + num

        forg_file = None
        if forg_base_name in self.au_base_name:
            forg_file = forg_base_name + self.au_base_name[forg_base_name]

        return src_file, forg_file

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index, im_only=False):
        row = self.df.loc[index]

        imfile = row["imfile"]
        gtfile = row["gt"]
        src_file = row["target"]

        imt = skimage.img_as_float32(skimage.io.imread(str(imfile))[:, :, :3])
        ims = skimage.img_as_float32(skimage.io.imread(str(src_file))[:, :, :3])
        label = 0

        if gtfile is not None:
            y = skimage.img_as_float32(skimage.io.imread(str(gtfile), as_gray=True))
            label = 1.
        else:
            y = np.zeros(ims.shape[:2], dtype=ims.dtype)
            label = 0
        ims, y = self.transform(ims, y)
        imt, _ = self.transform(imt)
        return ims, imt, y, label


class Dataset_casia_det(torch.utils.data.Dataset):
    def __init__(self, args=None, both=None):
        self.args = args
        self.transform = None

        if args.model in ("dmac", "dmvn"):
            self.transform = utils.CustomTransform_vgg(size=args.size)
        else:
            self.transform = utils.CustomTransform(size=args.size)

        self.root = Path(os.environ["HOME"]) / "dataset" / "CMFD" / "CASIA"

        self.au_imroot = self.root / "CASIA2.0" / "Au"
        self.tp_imroot = self.root / "CASIA2.0" / "Tp"

        au_files = sorted((self.au_imroot).glob("Au_*"))
        self.au_base_name = {x.stem: x.suffix for x in au_files}

        tp_files = sorted((self.tp_imroot).glob("Tp_D_*"))
        self.tp_base_name = {x.stem: x.suffix for x in tp_files}

        csv_file = self.root / "data_paired_CASIA_ids.csv"

        self.df = pd.read_csv(str(csv_file), sep=",")

        self.filter_data()

    def filter_data(self):
        au_index = (
            (self.df['Label'] == 0) &
            (self.df['ProbeID'].map(lambda x: x in self.au_base_name)) &
            (self.df['DonorID'].map(lambda x: x in self.au_base_name))
        )

        tp_index = (
            (self.df['Label'] == 1) &
            (self.df['ProbeID'].map(lambda x: x in self.tp_base_name)) &
            (self.df['DonorID'].map(lambda x: x in self.au_base_name))
        )

        index = au_index | tp_index
        self.df = self.df[index]
        print(f"Data size: {self.df.shape[0]} ")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index, im_only=False):
        row = self.df.iloc[index]
        label = row["Label"]

        if label == 0:
            imfile = self.au_imroot / (row["ProbeID"] + self.au_base_name[row["ProbeID"]])
            srcfile = self.au_imroot / (row["DonorID"] + self.au_base_name[row["DonorID"]])
        else:
            imfile = self.tp_imroot / (row["ProbeID"] + self.tp_base_name[row["ProbeID"]])
            srcfile = self.au_imroot / (row["DonorID"] + self.au_base_name[row["DonorID"]])

        imt = skimage.img_as_float32(skimage.io.imread(str(imfile))[:, :, :3])
        ims = skimage.img_as_float32(skimage.io.imread(str(srcfile))[:, :, :3])

        ims, _ = self.transform(ims)
        imt, _ = self.transform(imt)
        return ims, imt, label
