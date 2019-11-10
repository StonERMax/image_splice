#!/usr/bin/fish

# usage: fish run_all.sh [dataset] [cuda-device-ids] [model]

# conda activate torch

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

if test (count $argv) -lt 3
    set -x MODEL dmac
else
    set -x MODEL $argv[3]
end

echo "dataset : " $DATASET
echo "cuda devices: " $CUDA_VISIBLE_DEVICES
echo "Model: " $MODEL

if test $MODEL = "base"
    set -x base_ckpt ./ckpt/base_coco_for_vid.pkl
else
    set -x base_ckpt ./ckpt/{$MODEL}_coco.pkl
end

echo "base model $base_ckpt"

# output model name: base_[dataset].pkl
python main_train_vid_dmac.py --dataset $DATASET --model $MODEL --ckpt $base_ckpt \
    --max-epoch 10 | tee  ./log_out/run_exp_$DATASET.txt

python match_vid_save.py --dataset $DATASET --model $MODEL --ckpt ./ckpt/{$MODEL}_$DATASET.pkl --num 50 | tee -a ./log_out/run_exp_$DATASET.txt
python match_vid_load.py --dataset $DATASET --model $MODEL --ckpt ./ckpt/{$MODEL}_$DATASET.pkl --num 50 | tee -a ./log_out/run_exp_$DATASET.txt