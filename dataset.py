from __future__ import print_function
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


def get_dataset_path(dataset_name="como"):
    dataset_name = dataset_name.lower()
    ROOT = Path(os.environ["HOME"] / "dataset" / "CMFD")
    if dataset_name == "como":
        path = ROOT / "CoMoFoD-CMFD.hd5"
    elif dataset_name == "casia":
        path = ROOT / "CASIA-CMFD-Pos.hd5"
    else:
        raise ValueError("No dataset named {}".format(dataset_name))
    return str(path)


def write_arr(fp, arr):
    with open(fp, 'w') as file:
        _str = ','.join([str(i) for i in arr])
        file.write(_str)


class USCISI_CMD_Dataset(torch.utils.data.Dataset):

    def __init__(self, lmdb_dir, sample_file, args=None, to_tensor=True):
        assert os.path.isdir(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        sample_file = os.path.join(lmdb_dir, sample_file)
        assert os.path.isfile(sample_file)
        self.sample_keys = self._load_sample_keys(sample_file)

        if args.out_channel == 3:
            self.differentiate_target = True
        else:
            self.differentiate_target = False
        print(
            "INFO: successfully load USC-ISI CMD LMDB with {} keys".format(self.nb_samples))

        self.transform = utils.CustomTransform(size=args.size)
        self.to_tensor = to_tensor
        self.args = args

    @property
    def nb_samples(self):
        return len(self.sample_keys)

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
        if (self.differentiate_target):
            # 3-class target
            background = np.ones([h, w]).astype(
                'uint8') - np.maximum(src_mask, tgt_mask)
            cmd_mask = np.dstack(
                [tgt_mask, src_mask, background]).astype(np.float32)
        else:
            # 2-class target
            cmd_mask = np.maximum(src_mask, tgt_mask).astype(np.float32)
        if self.args.bw:
            bw = self.get_boundary(tgt_mask)
            cmd_mask = np.dstack((cmd_mask, bw))
        return cmd_mask

    def get_boundary(self, x):
        kernel = np.ones((10, 10))
        erosion = cv2.erode(x, kernel, iterations=1)
        dilation = cv2.dilate(x, kernel, iterations=1)
        bw = dilation - erosion
        bw = (bw>0.5).astype(x.dtype)
        return bw

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
        return self.get_one_sample(key=key_idx)


class Dataset_COMO(torch.utils.data.Dataset):
    def __init__(self, args):
        self.root = args.root
        self.data = h5py.File(args.data_path, 'r')
        self.args = args
        self.transform = utils.CustomTransform(size=args.size)

        self.save_root = Path('tmp_como')
        self.save_root.mkdir(exist_ok=True, parents=True)

        self.ynames = []
        for yn in self.data['YN']:
            self.ynames.append(yn.decode())

        self.xnames = []
        for xn in self.data['XN']:
            self.xnames.append(xn.decode())

    def close(self):
        self.data.close()

    def __len__(self):
        return self.data['X'].shape[0]

    def __getitem__(self, index, im_only=False, with_post=False):
        x = self.data['X'][index]
        name = self.xnames[index]
        x = x + np.array([103.939, 116.779, 123.68]).reshape([1, 1, 3])
        x = x / 255.
        x = x[..., ::-1]
        x = x.clip(min=0, max=1)

        # get Y
        idx, img_id, postproc = self.get_target_idx(name)
        y = self.data['Y'][idx]
        y = y.astype(np.float32)

        if self.args.out_channel == 1:
            y = np.maximum(y[..., 0], y[..., 1])
            y[:, 0] = 0
            y[0, :] = 0
            y[-1, :] = 0
            y[:, -1] = 0

        if im_only:
            if with_post:
                return x, y, postproc
            return x, y

        x_t, y_t = self.transform(x, y)
        return x_t, y_t

    def load_data(self, num=None, is_training=True):
        all_ind = self.train_ind if is_training else self.test_ind
        if num is None:
            indices = all_ind
        else:
            indices = np.random.choice(all_ind, size=num)

        X = []
        Y = []
        # data_names = []

        for i in indices:
            x, y, name = self[i]
            X.append(x)
            Y.append(y)
            # data_names.append(name)
        X = torch.stack(X, dim=0)
        Y = torch.stack(Y, dim=0)

        return X, Y

    def load_batch(self, shuffle=True, is_training=True):
        indices = np.arange(len(self))

        for ind in np.array_split(indices, len(indices)//self.args.batch_size):
            X = []
            Y = []
            for i in ind:
                x, y, _ = self[i]
                X.append(x)
                Y.append(y)
            X = torch.stack(X, dim=0)
            Y = torch.stack(Y, dim=0)
            yield X, Y

    def get_target_idx(self, xn):
        fmt = '{}_F_{}'
        try:
            img_id, postproc = parse(fmt, xn)
        except:
            img_id = xn.rsplit('_')[0]
            postproc = 'BASE'
        idx = self.ynames.index(img_id)
        return idx, img_id, postproc

    def write_to_file(self, post=None):
        data_root = Path(os.environ['HOME']) / \
            "dataset" / "CMFD" / "COMO" / "images"
        forged_root = data_root / "forged_base"
        gt_root = data_root / "gt_base"

        forged_root.mkdir(parents=True, exist_ok=True)
        gt_root.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(len(self))):
            im, y, gpost = self.__getitem__(i, im_only=True, with_post=True)

            if post is not None and gpost != post:
                continue

            skimage.io.imsave(
                str(forged_root / f"{i}.png"), im
            )
            skimage.io.imsave(
                str(gt_root / f"{i}.png"), y
            )

    def evaluate(self, Z, visualize=False, thresh=0.3,
                 rand_size=None):

        XN = self.xnames
        # 2. evaluate performance for each sample
        plut = {'mapping': {}}
        Y = self.data['Y']

        for xidx, (xn, z) in tqdm(enumerate(zip(XN, Z))):
            # 3. get corresponding target file
            idx, img_id, postproc = self.get_target_idx(xn)
            y = Y[idx].astype(np.float)

            # 4. evaluate performance
            if postproc not in plut:
                plut[postproc] = []
            ref = y[..., 2].ravel() == 0
            hyp = z.ravel() >= self.args.thres
            precision, recall, fscore, _ = precision_recall_fscore_support(ref, hyp,
                                                                           pos_label=1,
                                                                           average='binary')
            plut[postproc].append([precision, recall, fscore])
            if postproc == 'BASE':
                plut['mapping'][xidx] = [idx, fscore]
        # 5. show results
        print("INFO: BusterNet Performance on CoMoFoD-CMFD "
              "Dataset using the number of correct detections")
        print("-" * 100)
        for key, res in sorted(plut.items()):
            if key == 'mapping':
                continue
            # a sample is correct if its F1 score is above 0.5
            nb_correct = np.sum(np.row_stack(res)[:, -1] > thresh)
            print("{:>4s}: {:>3}".format(key, nb_correct))

        prf_lut = plut
        if visualize:
            print("INFO: show random results with F1 score > {}".format(thresh))
            candi = list(filter(None, [
                xidx if f1 > thresh else None for xidx, (yidx, f1) in prf_lut['mapping'].items()]))
            if rand_size is None:
                rand_size = int(Z.shape[0] * 0.5)
            indices = np.random.choice(candi, size=rand_size)
            X = self.data['X']
            for idx in indices:
                # 1. add back imageNet BGR means
                x = np.array(X[idx]) + \
                    np.array([103.939, 116.779, 123.68]).reshape([1, 1, 3])
                name = self.data['XN'][idx].decode()
                # 2. restore image dtype and BGR->RGB
                x = np.round(x).astype('uint8')[..., ::-1]
                # 3. set gt to float
                yidx, f1 = prf_lut['mapping'][idx]
                y = np.array(Y[yidx]).astype('float32')
                z = np.array(Z[idx])
                # 4. display
                fig = pyplot.figure(figsize=(12, 4))
                pyplot.subplot(131)
                pyplot.imshow(x)
                pyplot.title('test image')
                pyplot.subplot(132)
                pyplot.imshow(y)
                pyplot.title('ground truth')
                pyplot.subplot(133)
                pyplot.imshow(z)
                pyplot.title('BusterNet predicted')

                path = self.save_root / (str(name) + ".png")
                fig.savefig(path)

        return plut


class Dataset_CASIA(torch.utils.data.Dataset):
    def __init__(self, args=None):
        self.root = args.root
        self.args = args
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)

        self.save_root = Path('tmp_casia')
        self.save_root.mkdir(exist_ok=True, parents=True)

        data = h5py.File(args.data_path, 'r')
        self.X = data['X']
        self.Y = data['Y']

    def get_split(self):
        with open(os.path.join(self.root, f'train_{self.args.split}.txt')) as fp:
            indices_str = fp.read()
            self.train_ind = [int(i) for i in indices_str.strip().split(',')]

        with open(os.path.join(self.root, f'test_{self.args.split}.txt')) as fp:
            indices_str = fp.read()
            self.test_ind = [int(i) for i in indices_str.strip().split(',')]

    def create_split(self):
        indices = np.arange(self.X.shape[0])

        per = 0.6
        for i in range(5):
            np.random.shuffle(indices)
            tlen = int(per * len(indices))
            trn_ind, tst_ind = indices[:tlen], indices[tlen:]

            write_arr(
                os.path.join(self.root, f'train_{i}.txt'), trn_ind
            )
            write_arr(
                os.path.join(self.root, f'test_{i}.txt'), tst_ind
            )

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


class Dataset_CASIA_Neg(torch.utils.data.Dataset):
    def __init__(self, args=None):
        self.root = "/home/islama6a/dataset/CMFD/CASIA"
        self.args = args
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)

        self.save_root = Path('tmp_casia')
        self.save_root.mkdir(exist_ok=True, parents=True)

        data_path = ("/home/islama6a/dataset/CMFD/CASIA"
                     "/CASIA-CMFD-Neg.hd5")
        data = h5py.File(data_path, 'r')
        self.X = data['X']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index, is_training=True):
        x = self.X[index]
        x_t, _ = self.transform(x)
        y_t = torch.zeros((self.args.out_channel,
                           *self.args.size), dtype=x_t.dtype)
        return x_t, y_t


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
        mask = torch.zeros((self.args.out_channel + (1 if self.args.bw else 0),
                            *self.args.size), dtype=img.dtype)
        if self.args.out_channel == 3:
            mask[2, ...] = 1.
        return img, mask


class Dataset_tifs(torch.utils.data.Dataset):
    def __init__(self, args=None, is_training=None):
        self.root = args.root
        self.args = args
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)

        data = h5py.File(os.path.join(args.root, args.data_path), 'r')
        self.X = data['X']
        self.Y = data['Y']
        self.data = data
        self.is_training = is_training

        # self.create_split()
        # self.get_split()

    def get_split(self):
        with open(os.path.join(self.root, f'train_{self.args.split}.txt')) as fp:
            indices_str = fp.read()
            self.train_ind = [int(i) for i in indices_str.strip().split(',')]

        with open(os.path.join(self.root, f'test_{self.args.split}.txt')) as fp:
            indices_str = fp.read()
            self.test_ind = [int(i) for i in indices_str.strip().split(',')]

    def create_split(self):
        XN = self.data['XN']
        basenames = [i[0] for i in XN]
        basenames = np.unique(basenames)

        per = 0.8
        for i in range(3):
            np.random.shuffle(basenames)
            tlen = int(per * len(basenames))

            trn_ind = [i for i, d in enumerate(XN) if d[0] in basenames[:tlen]]
            tst_ind = [i for i, d in enumerate(XN) if d[0] in basenames[tlen:]]

            write_arr(
                os.path.join(self.root, f'train_{i}.txt'), trn_ind
            )
            write_arr(
                os.path.join(self.root, f'test_{i}.txt'), tst_ind
            )

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

        other_tfm = None
        # if is_training:
        #     other_tfm = utils.SimTransform(size=self.args.size)
        # else:
        #     other_tfm = None
        x_t, y_t = self.transform(x, y, other_tfm=other_tfm)
        return x_t, y_t

    def write_to_file(self):
        data_root = Path(os.environ['HOME']) / \
            "dataset" / "CMFD" / "tifs" / "images"
        forged_root = data_root / "forged"
        gt_root = data_root / "gt"

        forged_root.mkdir(parents=True, exist_ok=True)
        gt_root.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(len(self))):
            im, y = self.__getitem__(i, im_only=True)

            skimage.io.imsave(
                str(forged_root / f"{i}.png"), im
            )
            skimage.io.imsave(
                str(gt_root / f"{i}.png"), y
            )


class Dataset_grip(torch.utils.data.Dataset):
    def __init__(self, args=None, is_training=None):
        self.root = args.root
        self.args = args
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)

        data = h5py.File(os.path.join(args.root, args.data_path), 'r')
        self.X = data['X']
        self.Y = data['Y']
        self.data = data
        self.is_training = is_training

        self.create_split()
        self.get_split()

    def get_split(self):
        with open(os.path.join(self.root, f'train_{self.args.split}.txt')) as fp:
            indices_str = fp.read()
            self.train_ind = [int(i) for i in indices_str.strip().split(',')]

        with open(os.path.join(self.root, f'test_{self.args.split}.txt')) as fp:
            indices_str = fp.read()
            self.test_ind = [int(i) for i in indices_str.strip().split(',')]

    def create_split(self):
        XN = self.data['XN']
        basenames = [i[0] for i in XN]
        basenames = np.unique(basenames)

        per = 0.8
        for i in range(3):
            np.random.shuffle(basenames)
            tlen = int(per * len(basenames))

            trn_ind = [i for i, d in enumerate(XN) if d[0] in basenames[:tlen]]
            tst_ind = [i for i, d in enumerate(XN) if d[0] in basenames[tlen:]]

            write_arr(
                os.path.join(self.root, f'train_{i}.txt'), trn_ind
            )
            write_arr(
                os.path.join(self.root, f'test_{i}.txt'), tst_ind
            )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ind, is_training=True):

        index = ind
        x = self.X[index]
        x = skimage.img_as_float32(x)

        # get Y
        y = self.Y[index]
        y = y.astype(np.float32)

        if is_training:
            other_tfm = utils.SimTransform(size=self.args.size)
        else:
            other_tfm = None
        x_t, y_t = self.transform(x, y, other_tfm=other_tfm)
        return x_t, y_t


class Dataset_como_orig(torch.utils.data.Dataset):
    def __init__(self, args=None, both=None, mode=None):
        self.args = args

        self.transform = None
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)
        self.root = Path(os.environ['HOME']) / 'dataset/CMFD/CoMoFoD_small_v2'
        self.mode = mode
        num = np.arange(1, 201)
        imnames = []
        gt_names = []
        post_proc = []
        for i in num:
            glob_name = f'{i:03d}_F*.png'
            for efile in self.root.glob(glob_name):
                imid, post = self.id_post(efile.stem)
                gtfile = f'{imid}_B.png'

                if mode is not None and post != mode:
                    continue
                imnames.append(efile.name)
                gt_names.append(gtfile)
                post_proc.append(post)
            if both is not None:
                glob_name = f'{i:03d}_O*.png'
                for efile in self.root.glob(glob_name):
                    imid, post = self.id_post(efile.stem)
                    if mode is not None and post != mode:
                        continue
                    gtfile = f'{imid}_B.png'
                    imnames.append(efile.name)
                    gt_names.append(None)
                    post_proc.append(None)
        self.df = pd.DataFrame(
            data={'file': imnames, 'gt': gt_names, 'post': post_proc}, dtype=str)

    def id_post(self, xn):
        fmt = '{}_F_{}'
        try:
            img_id, postproc = parse(fmt, xn)
        except:
            img_id = xn.rsplit('_')[0]
            postproc = 'BASE'
        return img_id, postproc

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index, im_only=False):
        row = self.df.loc[index]

        imfile = row['file']
        gtfile = row['gt']
        proc = row['post']

        im = skimage.img_as_float32(skimage.io.imread(
            str(self.root / imfile))[:, :, :3])

        if gtfile is not None:
            y = skimage.img_as_float32(skimage.io.imread(
                str(self.root / gtfile), as_gray=True))
        else:
            y = np.zeros(im.shape[:2], dtype=im.dtype)

        if self.args is not None:
            im = cv2.resize(im, self.args.size, interpolation=1)
            y = cv2.resize(y, self.args.size, interpolation=0)

        if im_only:
            return im, y, proc
        im, y = self.transform(im, y)
        return im, y

    def evaluate(self, Z, visualize=False, thresh=0.5,
                 rand_size=None):

        # 2. evaluate performance for each sample
        plut = defaultdict(list)

        for idx in tqdm(range(Z.shape[0])):
            row = self.df.iloc[idx]

            gtfile = row['gt']
            postproc = row['post']

            if gtfile is not None:
                y = skimage.img_as_float32(skimage.io.imread(
                    str(self.root / gtfile), as_gray=True))
                y = cv2.resize(y, tuple(Z[idx].shape[-2:]), interpolation=0)
            else:
                y = np.zeros(Z[idx].shape[-2:], dtype=np.float)

            ref = y.ravel() > 0.5
            hyp = Z[idx].ravel() > self.args.thres
            precision, recall, fscore, _ = precision_recall_fscore_support(ref, hyp,
                                                                           pos_label=1,
                                                                           average='binary')
            plut[postproc].append([precision, recall, fscore])

        print("INFO: BusterNet Performance on CoMoFoD-CMFD "
              "Dataset using the number of correct detections")
        print("-" * 100)
        for key, res in sorted(plut.items()):
            if key == 'BASE':
                continue
            # a sample is correct if its F1 score is above 0.5
            rlen = len(np.row_stack(res)[:, -1])
            nb_correct = np.sum(np.row_stack(res)[:, -1] > thresh)
            mean_f = np.mean(np.row_stack(res)[:, -1]) 
            print("{:>4s}({}): {:>3}, {:.2f}".format(key, rlen, nb_correct, mean_f))


class Dataset_casia_orig(torch.utils.data.Dataset):
    def __init__(self, args=None, both=None):
        self.args = args
        self.transform = None
        if args is not None:
            self.transform = utils.CustomTransform(size=args.size)

        self.root = Path(os.environ['HOME']) / 'dataset' / 'CMFD' / 'CASIA'

        self.imroot = self.root / 'CASIA2.0'
        self.gtroot = self.root / 'GT'

        imnames = []
        gt_names = []

        for efile in (self.imroot/'Tp').glob('Tp_S_*'):
            if efile.suffix in ('.bmp', '.tif', '.jpg', '.png'):
                gtfile = f'{efile.stem}_gt.png'
                imnames.append(str(self.imroot / 'Tp' / efile))
                gt_names.append(str(self.gtroot / gtfile))
        pos_len = len(imnames)
        if both is not None:
            for i, efile in enumerate((self.imroot/'Au').glob('Tp_S_*')):
                if efile.suffix in ('.bmp', '.tif', '.jpg', '.png'):
                    imnames.append(str(self.imroot / 'Au' / efile))
                    gt_names.append(None)
                if i >= pos_len:
                    break
        self.df = pd.DataFrame(data={'file': imnames, 'gt': gt_names},
                               dtype=str)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index, im_only=False):
        row = self.df.loc[index]

        imfile = row['file']
        gtfile = row['gt']

        im = skimage.img_as_float32(skimage.io.imread(
            str(imfile))[:, :, :3])

        if gtfile is not None:
            y = skimage.img_as_float32(skimage.io.imread(
                str(gtfile), as_gray=True))
        else:
            y = np.zeros(im.shape[:2], dtype=im.dtype)

        if self.args is not None:
            im = cv2.resize(im, self.args.size, interpolation=cv2.INTER_CUBIC)
            y = cv2.resize(y, self.args.size, interpolation=0)

        if im_only:
            return im, y
        im, y = self.transform(im, y)
        return im, y
