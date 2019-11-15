"""
coco dataset with background cmfd
"""
from __future__ import print_function
import glob
from pathlib import Path
from matplotlib import pyplot
import json
import os
import sys
import cv2
import skimage
from skimage import io
import numpy as np
import torch
from torchvision import transforms
import utils
from parse import parse
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import defaultdict
from utils import ImAug


def centroid_bb(x):
    return np.array([int((x[0]+x[2]/2)), int((x[1]+x[3]/2))])


class COCODatasetBack(torch.utils.data.Dataset):
    def __init__(self, args=None, is_training=False, sample_len=400):
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

        # augmenter
        self.im_aug = ImAug()

    def __len__(self):
        return len(self.imids)

    def transform_mask(self, img, mask, ann):
        h, w = img.shape[:2]
        mask_bb = ann['bbox']
        mask_h, mask_w = mask_bb[3], mask_bb[2]
        
        # scale wrt original mask
        s_h, s_w = (
            np.random.choice(np.linspace(0.6, 1.4, num=10)),
            np.random.choice(np.linspace(0.6, 1.4, num=10))
        )
        # new mask shape
        m_h, m_w = int(np.clip(s_h * mask_h, 0.2 * h, 0.4 * h)), int(np.clip(s_w * mask_w, 0.2*w, 0.4 * w))
        # new mask scale wrt original mask
        sx, sy = m_w / mask_w, m_h / mask_h

        # new mask centroid
        new_centroid_x = np.random.choice(
            np.arange(m_w//2, w//2-m_w//2)
        )
        new_centroid_y = np.random.choice(
            np.arange(m_h//2, h-m_h//2)
        )

        # translate new mask
        x, y = centroid_bb(mask_bb)
        translate = [new_centroid_x - x, new_centroid_y - y]

        matrix1 = np.array([
            [sx, 0, x*(1-sx)],
            [0, sy, y*(1-sy)],
            [0, 0, 1]
        ])

        matrix2 = np.array([
            [1, 0, translate[0]],
            [0, 1, translate[1]],
            [0, 0, 1]
        ])
        matrix = np.matmul(matrix2, matrix1)
        tfm = skimage.transform.AffineTransform(matrix)

        if self.is_training:
            mask = self.im_aug.apply_water(mask)

        new_mask = skimage.transform.warp(mask, tfm.inverse, order=0)

        # forge
        new_centroid_x_2 = np.random.choice(
            np.arange(w//2 + m_w//2, w-m_w//2)
        )
        new_centroid_y_2 = np.random.choice(
            np.arange(m_h//2, h-m_h//2)
        )
        scale_new = np.random.choice(np.linspace(0.8, 1.2, 10))
        sx2, sy2 = scale_new, scale_new
        translate2 = [new_centroid_x_2 - new_centroid_x, new_centroid_y_2 - new_centroid_y]

        matrix1 = np.array([
            [sx2, 0, new_centroid_x_2*(1-sx2)],
            [0, sy2, new_centroid_y_2*(1-sy2)],
            [0, 0, 1]
        ])

        matrix2 = np.array([
            [1, 0, translate2[0]],
            [0, 1, translate2[1]],
            [0, 0, 1]
        ])
        matrix_f = np.matmul(matrix2, matrix1)
        tfm_f = skimage.transform.AffineTransform(matrix_f)
        new_mask_f = skimage.transform.warp(new_mask, tfm_f.inverse, order=0)
        im_fn = skimage.transform.warp(img, tfm_f.inverse, order=1)

        if self.is_training:
            im_fn, new_mask_f = self.im_aug.apply_coco_back(im_fn, new_mask_f)
            im_fn = ImAug.apply_blur(im_fn)

        im_cmfd = img * (1 - new_mask_f[..., None]) + im_fn * new_mask_f[..., None]

        return im_cmfd, new_mask, new_mask_f

    def __getitem__(self, idx=None):
        coco = self.coco
        if idx is None:
            index = np.random.choice(self.imids)
        else:
            index = self.imids[idx]
        im_info = self.coco.loadImgs([index])[0]
        
        img = skimage.img_as_float32(io.imread(
            str(self.imDir / im_info['file_name'])
        ))
        img = skimage.color.gray2rgb(img)

        while True:
            im_info2 = self.coco.loadImgs([np.random.choice(self.imids)])[0]
            annids = coco.getAnnIds(imgIds=im_info2['id'])
            try:
                annid = np.random.choice(annids)
                break
            except:
                continue
        ann = np.random.choice(coco.loadAnns([annid]))
        mask = coco.annToMask(ann).astype(np.float32)

        img = cv2.resize(img, (im_info2['width'], im_info2['height']))

        img, mask_s, mask_f = self.transform_mask(img, mask, ann)
        img, mask_s = self.transform(img, mask_s)
        _,  mask_f = self.transform(None, mask_f)
        # return img, mask_s, mask_f

        if self.args.out_channel == 3:
            return img, img, mask_s.float(), mask_f, 1.0
        else:
            mask = torch.max(mask_s.float(), mask_f)
        return img, mask


if __name__ == '__main__':
    # np.random.seed(1)
    dat = COCODatasetBack()
    im, ms, mf = dat.__getitem__()

    io.imsave('1.png', im)
    io.imsave('2.png', ms)
    io.imsave('3.png', mf)

    print("saved")