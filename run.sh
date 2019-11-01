conda activate torch

set -x CUDA_VISIBLE_DEVICES 0

env CUDA_VISIBLE_DEVICES=0 python -m pdb main_train_vid.py --ckpt ./ckpt/base_coco_exp.pkl --max-epoch 50 --suffix _exp --dataset davis
python  match_vid_save_with_mani.py --ckpt ./ckpt/base_tmp_youtube_exp.pkl --ckptM ./ckpt/detseg_base_tmp_youtube.pkl
python  match_vid_load_with_mani.py --ckpt ./ckpt/base_tmp_youtube_exp.pkl --ckptM ./ckpt/detseg_base_tmp_youtube.pkl