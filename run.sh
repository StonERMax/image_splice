conda activate torch

set -x CUDA_VISIBLE_DEVICES 0

python  match_vid_save_with_mani.py --ckpt ./ckpt/base_tmp_youtube.pkl --ckptM ./ckpt/detseg_base_tmp_youtube.pkl
python  match_vid_load_with_mani.py --ckpt ./ckpt/base_tmp_youtube.pkl --ckptM ./ckpt/detseg_base_tmp_youtube.pkl