conda activate torch

env CUDA_VISIBLE_DEVICES=0 python  main_train_vid_dmac.py --dataset davis \
 --max-epoch 20 --model dmac --lr 1e-3

env CUDA_VISIBLE_DEVICES=0 python  main_train_vid_dmac.py \
 --model dmvn  --batch-size 10 --max-epoch 30 --lr 1e-4 \
  --dataset tmp_youtube >> log_out/train_dmvn_yt.txt

env CUDA_VISIBLE_DEVICES=0 python  main_train_vid_dmac.py \
 --model dmac  --batch-size 10 --max-epoch 30 --lr 1e-4 \
  --dataset tmp_youtube >> log_out/train_dmac_yt.txt

echo training finished

env CUDA_VISIBLE_DEVICES=0 python  match_vid_save.py  --model dmvn \
    --dataset tmp_youtube --ckpt ./ckpt/dmvn_tmp_youtube.pkl
env CUDA_VISIBLE_DEVICES=0 python  match_vid_save.py  --model dmac \
    --dataset tmp_youtube --ckpt ./ckpt/dmac_tmp_youtube.pkl
# env CUDA_VISIBLE_DEVICES=0 python  match_vid_save.py  --model base \
#     --dataset tmp_youtube --ckpt ./ckpt/base_tmp_youtube.pkl

echo save finished


# python match_vid_load.py --dataset tmp_youtube --model base >> ./log_out/youtube/base.txt
python match_vid_load.py --dataset tmp_youtube --model dmvn >> ./log_out/youtube/dmvn.txt
python match_vid_load.py --dataset tmp_youtube --model dmac >> ./log_out/youtube/dmac.txt