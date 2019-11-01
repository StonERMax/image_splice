conda activate torch
set -x CUDA_VISIBLE_DEVICES 1

git checkout ablation_a1
env CUDA_VISIBLE_DEVICES=1 python main_template_match.py  \
 --model wo_a1 --suffix _exp --max-epoch 5

git checkout ablation_a1
env CUDA_VISIBLE_DEVICES=1 python main_train_temporal.py  --tune\
 --model wo_a1 --suffix _exp --max-epoch 50 --ckpt ./ckpt/wo_a1_coco_exp.pkl


git checkout ablation_a1
env CUDA_VISIBLE_DEVICES=1 python main_train_temporal.py  --lr 1e-4 --batch-size 2\
 --model wo_a1 --suffix _exp --max-epoch 50 --ckpt ./ckpt/temporal_wo_a1_tmp_youtube_exp.pkl
