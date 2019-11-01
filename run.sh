conda activate torch
set -x CUDA_VISIBLE_DEVICES 0

git checkout ablation_1_aspp
env CUDA_VISIBLE_DEVICES=0 python -m pdb main_template_match.py  \
 --model aspp1 --suffix _exp --max-epoch 5

git checkout ablation_1_aspp
env CUDA_VISIBLE_DEVICES=0 python -m pdb main_train_temporal.py  --tune \
 --model aspp1 --suffix _exp --max-epoch 50 --ckpt ./ckpt/aspp1_coco_exp.pkl
