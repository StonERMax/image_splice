
import glob
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
import lmdb
import json
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
import utils
import h5py
from parse import parse
from sklearn.metrics import precision_recall_fscore_support
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import defaultdict
import random

HOME = os.environ['HOME']


class USCISI_CMD_Dataset(torch.utils.data.Dataset):

    def __init__(self, args=None, is_training=True, to_tensor=True, sample_len=None):

        if is_training:
            sample_file = "train.keys"
        else:
            sample_file = "test.keys"

        HOME = os.environ['HOME']
        lmdb_dir = HOME+"/dataset/CMFD/USCISI-CMFD"
        assert os.path.isdir(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        sample_file = os.path.join(lmdb_dir, sample_file)
        assert os.path.isfile(sample_file)

        self.sample_len = sample_len

        self.sample_keys = self._load_sample_keys(sample_file)

        print(
            "INFO: successfully load USC-ISI CMD LMDB with {} keys".format(self.nb_samples))

        if args.model in ("dmac", "dmvn"):
            self.transform = utils.CustomTransform_vgg(size=args.size)
        else:
            self.transform = utils.CustomTransform(size=args.size)
        self.to_tensor = to_tensor

    @property
    def nb_samples(self):
        if self.sample_len is None:
            return len(self.sample_keys)
        else:
            return min(self.sample_len, len(self.sample_keys))

    def _load_sample_keys(self, sample_file):
        '''Load sample keys from a given sample file
        INPUT:
            sample_file = str, path to sample key file
        OUTPUT:
            keys = list of str, each element is a valid key in LMDB
        '''
        with open(sample_file, 'r') as IN:
            keys = [line.strip() for line in IN.readlines()]
        return keys

    def _get_image_from_lut(self, lut):
        '''Decode image array from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            image = np.ndarray, dtype='uint8'
        '''
        image_jpeg_buffer = lut['image_jpeg_buffer']
        image = cv2.imdecode(np.array(image_jpeg_buffer).astype(
            'uint8').reshape([-1, 1]), 1)
        return image

    def _get_mask_from_lut(self, lut):
        '''Decode copy-move mask from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            cmd_mask = np.ndarray, dtype='float32'
                       shape of HxWx1, if differentiate_target=False
                       shape of HxWx3, if differentiate target=True
        NOTE:
            cmd_mask is encoded in the one-hot style, if differentiate target=True.
            color channel, R, G, and B stand for TARGET, SOURCE, and BACKGROUND classes
        '''
        def reconstruct(cnts, h, w, val=1):
            rst = np.zeros([h, w], dtype='uint8')
            cv2.fillPoly(rst, cnts, val)
            return rst
        h, w = lut['image_height'], lut['image_width']
        src_cnts = [np.array(cnts).reshape([-1, 1, 2])
                    for cnts in lut['source_contour']]
        src_mask = reconstruct(src_cnts, h, w, val=1)
        tgt_cnts = [np.array(cnts).reshape([-1, 1, 2])
                    for cnts in lut['target_contour']]
        tgt_mask = reconstruct(tgt_cnts, h, w, val=1)
        # if (self.differentiate_target):
        # 3-class target
        background = np.ones([h, w]).astype(
            'uint8') - np.maximum(src_mask, tgt_mask)
        cmd_mask = np.dstack(
            [tgt_mask, src_mask, background]).astype(np.float32)
        # else:
        #     # 2-class target
        #     cmd_mask = np.maximum(src_mask, tgt_mask).astype(np.float32)
        return cmd_mask

    def _get_transmat_from_lut(self, lut):
        '''Decode transform matrix between SOURCE and TARGET
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            trans_mat = np.ndarray, dtype='float32', size of 3x3
        '''
        trans_mat = lut['transform_matrix']
        return np.array(trans_mat).reshape([3, 3])

    def _decode_lut_str(self, lut_str):
        '''Decode a raw LMDB lut
        INPUT:
            lut_str = str, raw string retrieved from LMDB
        OUTPUT: 
            image = np.ndarray, dtype='uint8', cmd image
            cmd_mask = np.ndarray, dtype='float32', cmd mask
            trans_mat = np.ndarray, dtype='float32', cmd transform matrix
        '''
        # 1. get raw lut
        lut = json.loads(lut_str)
        # 2. reconstruct image
        image = self._get_image_from_lut(lut)
        # 3. reconstruct copy-move masks
        cmd_mask = self._get_mask_from_lut(lut)
        # 4. get transform matrix if necessary
        trans_mat = self._get_transmat_from_lut(lut)
        return (image, cmd_mask, trans_mat)

    def get_one_sample(self, key=None):
        '''Get a (random) sample from given key
        INPUT:
            key = str, a sample key or None, if None then use random key
        OUTPUT:
            sample = tuple of (image, cmd_mask, trans_mat)
        '''
        return self.get_samples([key])[0]

    def _preprocess(self, sample):
        image, cmd_mask, trans_mat = sample
        image = skimage.img_as_float32(image)

        if self.transform is not None:
            image, cmd_mask = self.transform(image, cmd_mask)
        trans_mat = torch.tensor(trans_mat, dtype=torch.float32)
        return image, cmd_mask

    def get_samples(self, key_list):
        '''Get samples according to a given key list
        INPUT:
            key_list = list, each element is a LMDB key or idx
        OUTPUT:
            sample_list = list, each element is a tuple of (image, cmd_mask, trans_mat)
        '''
        env = lmdb.open(self.lmdb_dir)
        sample_list = []
        with env.begin(write=False) as txn:
            for key in key_list:
                if not isinstance(key, str) and isinstance(key, int):
                    idx = key % self.nb_samples
                    key = self.sample_keys[idx]
                elif isinstance(key, str):
                    pass
                else:
                    key = np.random.choice(self.sample_keys, 1)[0]
                    print("INFO: use random key", key)
                lut_str = txn.get(key.encode())
                sample = self._decode_lut_str(lut_str)
                if self.to_tensor:
                    sample = self._preprocess(sample)
                sample_list.append(sample)
        return sample_list

    def __len__(self):
        return self.nb_samples

    def __call__(self, key_list):
        return self.get_samples(key_list)

    def __getitem__(self, key_idx):
        img, cmd_mask = self.get_one_sample(key=key_idx)
        im_s, im_t, mask_s, mask_t = img, img.clone(), cmd_mask[1], cmd_mask[0]
        if torch.any(mask_t > 0.5):
            label = 1.0
        else:
            label = 0.0
        return im_s, im_t, mask_s.unsqueeze(0), mask_t.unsqueeze(0), label


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, args=None, is_training=True, sample_len=4000):
        self.dataDir = Path('~/dataset/coco').expanduser()
        self.year = '2014'
        self.train_ann_file = self.dataDir / 'annotations' / \
            'instances_train{}.json'.format(self.year)
        self.test_ann_file = self.dataDir / 'annotations' / \
            'instances_val{}.json'.format(self.year)
        self.train_im_folder = self.dataDir / 'images' / f'train{self.year}'
        self.test_im_folder = self.dataDir / 'images' / f'val{self.year}'

        self.args = args
        self.transform = utils.CustomTransform(size=args.size)

        self.is_training = is_training

        if is_training:
            annFile = self.train_ann_file
            self.imDir = self.train_im_folder
        else:
            annFile = self.test_ann_file
            self.imDir = self.test_im_folder

        self.coco = COCO(annFile)
        imids = self.coco.getImgIds()
        self.imids = np.random.choice(imids, size=sample_len, replace=False)

    def __len__(self):
        return len(self.imids)

    def __getitem__(self, idx):
        index = self.imids[idx]
        im_info = self.coco.loadImgs([index])[0]
        img = skimage.img_as_float32(io.imread(
            str(self.imDir / im_info['file_name'])
        ))
        img = skimage.color.gray2rgb(img)
        img, _ = self.transform(img)
        mask = torch.zeros((self.args.out_channel,
                            *self.args.size), dtype=img.dtype)
        if self.args.out_channel == 3:
            mask[2, ...] = 1.
        return img, mask


class Dataset_CASIA(torch.utils.data.Dataset):
    def __init__(self, args=None):
        self.args = args
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)
        HOME = os.environ['HOME']
        data_path = HOME + "/dataset/CMFD/CASIA/CASIA-CMFD-Pos.hd5"
        data = h5py.File(data_path, 'r')
        self.X = data['X']
        self.Y = data['Y']
        # self.get_split()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x = self.X[index]
        x = x + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 3])
        x = x[..., ::-1]
        x = x / 255.
        x = x.clip(min=0, max=1)

        # get Y
        y = self.Y[index]
        y = y.astype(np.float32)

        if self.args.out_channel == 1:
            y = np.maximum(y[..., 0], y[..., 1])

        if len(np.unique(y)) > 2:
            print(f"{index} unique more")
            return self.__getitem__(0)

        x_t, y_t = self.transform(x, y, other_tfm=None)
        return x_t, y_t


class Dataset_tifs(torch.utils.data.Dataset):
    def __init__(self, args=None, is_training=None):
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)
        data = h5py.File(HOME+"/dataset/CMFD/tifs/tifs.hd5", 'r')
        self.X = data['X']
        self.Y = data['Y']
        self.data = data
        self.is_training = is_training

    def __len__(self):
        # if self.is_training is None:
        return len(self.X)

    def __getitem__(self, index, is_training=True, im_only=False):

        x = self.X[index]
        x = skimage.img_as_float32(x)
        # get Y
        y = self.Y[index]
        y = y.astype(np.float32)
        if im_only:
            return x, y
        x_t, y_t = self.transform(x, y)
        return x_t, y_t
        # output similarity mask


class Dataset_grip(torch.utils.data.Dataset):
    def __init__(self, args=None, is_training=None):
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)
        data = h5py.File(HOME+"/dataset/CMFD/grip/grip.hd5", 'r')
        self.X = data['X']
        self.Y = data['Y']
        self.data = data
        self.is_training = is_training

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind, is_training=True):
        index = ind
        x = self.X[index]
        x = skimage.img_as_float32(x)

        # get Y
        y = self.Y[index]
        y = y.astype(np.float32)

        # if is_training:
        #     other_tfm = utils.SimTransform(size=self.args.size)
        # else:
        #     other_tfm = None
        x_t, y_t = self.transform(x, y, other_tfm=None)
        return x_t, y_t
        # output similarity mask


class Dataset_wwt(torch.utils.data.Dataset):
    def __init__(self, args):
        root = Path(HOME + "/dataset/CMFD/WildWebDataset/WildWeb")
        self.transform = utils.CustomTransform(size=args.size)
        list_files = sorted(root.iterdir())
        self.dict = {}
        for each_file in list_files:
            _list = []
            _mask = []
            for path in Path(each_file).iterdir():
                if path.suffix == ".jpg":
                    _list.append(path)
            for each in Path(each_file / "Mask").iterdir():
                if each.suffix == ".png":
                    _mask.append(each)
            self.dict[each_file.name] = {"files": _list, "mask": _mask}
        self.keys = list(self.dict.keys())
        print(f"Number of unique images {len(self)}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        name = self.keys[index]
        files, list_mask = self.dict[name]['files'], self.dict[name]['mask']
        imfile = random.choice(files)

        im = skimage.img_as_float32(skimage.io.imread(str(imfile)))
        im = skimage.color.gray2rgb(im)
        x, _ = self.transform(im)
        im_masks = []
        for each_mask in list_mask:
            _msk = skimage.img_as_float32(skimage.io.imread(str(each_mask), as_gray=True))
            im_masks.append(self.transform(None, _msk)[1])
        if len(im_masks) == 1:
            y = im_masks[0]
        else:
            y = torch.max(torch.cat(im_masks, 0), 0, keepdim=True)[0]
        return x, y
        # output forge mask
