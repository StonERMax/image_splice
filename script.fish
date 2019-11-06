#!/usr/bin/fish

# usage: fish run_all.sh [dataset] [cuda-device-ids]

conda activate torch

set -x CUDA_VISIBLE_DEVICES 0

if test (count $argv) -lt 1
    set -x DATASET youtube
else
    set -x DATASET $argv[1]
    if test (count $argv) -lt 2
        set -x CUDA_VISIBLE_DEVICES 0
    else
        set -x CUDA_VISIBLE_DEVICES $argv[2]
    end
end

echo "dataset : " $DATASET
echo "cuda devices: " $CUDA_VISIBLE_DEVICES

# output model name: base_[dataset].pkl
python main_train_vid.py --dataset $DATASET --ckpt ./ckpt/base_coco_exp_comb.pkl \
    --max-epoch 50 | tee  ./log_out/run_exp_$DATASET.txt

# save on temporal_base_[dataset].pkl
# python main_train_temporal.py --dataset $DATASET --ckpt ./ckpt/base_$DATASET.pkl \
#     --tune --max-epoch 30 | tee -a ./log_out/run_$DATASET.txt

# python main_train_temporal.py --dataset $DATASET \
#     --ckpt ./ckpt/temporal_base_$DATASET.pkl   --max-epoch 30 | tee -a ./log_out/run_$DATASET.txt

# # save on detseg_base_[dataset].pkl
# python main_template_match.py --dataset $DATASET --max-epoch 50 | tee -a ./log_out/run_$DATASET.txt

# ######## Test #########
# python temporal_match_vid.py --dataset $DATASET --ckpt ./ckpt/temporal_base_$DATASET.pkl \
#     --ckptM ./ckpt/detseg_base_$DATASET.pkl | tee -a ./log_out/run_$DATASET.txt