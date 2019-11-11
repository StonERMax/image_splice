#!/usr/bin/fish

# usage: fish run_all.sh [dataset] [cuda-device-ids]

conda activate torch

set -x CUDA_VISIBLE_DEVICES 0

# python main_cmfd.py --mode mani --max-iter 1000  --batch-size 20 --ckpt ./ckpt_cmfd/cmfd_usc_both_direct.pkl
python main_cmfd.py --mode sim --max-iter 3000 --ckpt ./ckpt_cmfd/cmfd_usc_both_direct.pkl --tune
python main_cmfd.py --mode sim --max-iter 3000 --ckpt ./ckpt_cmfd/cmfd_usc_sim.pkl

# python main_cmfd.py --mode both --max-iter 2000 --ckpt --ckpt ./ckpt/base_coco_for_video.pkl
# python main_cmfd.py --mode both --max-iter 2000 --ckpt ./ckpt_cmfd/cmfd_usc_sim.pkl
