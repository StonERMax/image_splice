import numpy as np
from scipy.io import loadmat

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from pathlib import Path
import skimage
import create_volume

# custom module
import config
from dataset_vid import Dataset_vid
import utils
import os
import cv2

if __name__ == "__main__":
    args = config.config_video()
    # seed
    np.random.seed(args.seed)
    args.model = "patchmatch3d"

    dataset = Dataset_vid(args=args, is_training=False)

    map_dir = f"./patchmach3d/map_{args.dataset}"

    metric = utils.Metric(names=["all"])

    # save all images
    for ret in tqdm(dataset.load_videos_all(is_training=False,
                                            shuffle=False, to_tensor=False)):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret
        map_file = os.path.join(map_dir, name.split(".")[0]+'.mat')

        Y_all = np.maximum(Y_forge, Y_orig)
        out_all = loadmat(map_file)['map']
        out_all = cv2.resize(out_all, (X.shape[1], X.shape[2]), interpolation=0)
        out_all = out_all.transpose(2, 0, 1)
        N = X.shape[0]

        metric.update(Y_all, out_all)

        folder_name = Path("tmp_out_final") / args.dataset / args.model / name
        folder_gt = folder_name / "gt"
        folder_pred = folder_name / "pred"
        folder_gt.mkdir(parents=True, exist_ok=True)
        folder_pred.mkdir(parents=True, exist_ok=True)

        for i_cnt in range(N):
            im = X[i_cnt]
            im_with_gt = utils.add_overlay(im, Y_all[i_cnt], c1=[1, 0, 0])
            im_with_pred = utils.add_overlay(
                im, out_all[i_cnt], c1=[1, 0, 0]
            )

            skimage.io.imsave(
                str(folder_gt / f"{i_cnt}.jpg"),
                skimage.img_as_ubyte(im_with_gt),
            )
            skimage.io.imsave(
                str(folder_pred / f"{i_cnt}.jpg"),
                skimage.img_as_ubyte(im_with_pred),
            )
        create_volume.create(Y_all, Y_all, path=folder_name / "gt_vol.png")

        create_volume.create(
            out_all, out_all, path=folder_name / "pred_vol.png"
        )
    
    print("final score")
    metric.final()

