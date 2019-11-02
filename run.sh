conda activate torch
set -x CUDA_VISIBLE_DEVICES 0

# git checkout ablation_1_aspp
# env CUDA_VISIBLE_DEVICES=0 python main.py  \
#  --model aspp1 --suffix _exp --max-epoch 5

git checkout ablation_1_aspp
env CUDA_VISIBLE_DEVICES=0 python main_train_temporal.py  --tune\
 --model aspp1 --suffix _exp --max-epoch 50 --ckpt ./ckpt/aspp1_coco_exp.pkl


git checkout ablation_1_aspp
env CUDA_VISIBLE_DEVICES=0 python main_train_temporal.py  --lr 1e-4 --batch-size 2\
 --model aspp1 --suffix _exp --max-epoch 50 --ckpt ./ckpt/temporal_aspp1_tmp_youtube_exp.pkl
