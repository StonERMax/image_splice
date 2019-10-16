conda activate torch

env CUDA_VISIBLE_DEVICES=2 python  main_train_vid_dmac.py \
 --model dmvn  --batch-size 10 --max-epoch 20 --lr 1e-4 \
  --dataset davis


env CUDA_VISIBLE_DEVICES=2 python  main_train_vid.py \
 --model base --ckpt ./ckpt/refine_davis.pkl  --batch-size 10 --max-epoch 20 \
  --lr 1e-4 --dataset davis

env CUDA_VISIBLE_DEVICES=2 python  match_vid_save.py  --model dmvn \
    --dataset davis --ckpt ./ckpt/dmvn_davis.pkl
env CUDA_VISIBLE_DEVICES=2 python  match_vid_save.py  --model dmac \
    --dataset davis --ckpt ./ckpt/dmac_davis.pkl
env CUDA_VISIBLE_DEVICES=2 python  match_vid_save.py  --model base \
    --dataset davis --ckpt ./ckpt/base_davis.pkl


# python match_vid_load.py --dataset davis --model base >> ./log_out/davis/base.txt
python match_vid_load.py --dataset davis --model dmvn >> ./log_out/davis/dmvn.txt
python match_vid_load.py --dataset davis --model dmac >> ./log_out/davis/dmac.txt