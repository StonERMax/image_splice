#!/usr/bin/fish

# usage: fish run_all.sh [dataset] [cuda-device-ids]

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

python main.py --max-epoch 2 --suffix _tmp | tee ./log_out/run_$DATASET.txt

# output model name: base_[dataset].pkl
python main_train_vid.py --dataset $DATASET --ckpt ./ckpt/base_coco_tmp.pkl \
    --max-epoch 30 --lr 1e-4 | tee  ./log_out/run_$DATASET.txt

# save on temporal_base_[dataset].pkl
python main_train_temporal.py --dataset $DATASET --ckpt ./ckpt/base_$DATASET.pkl \
    --tune --max-epoch 30 --lr 1e-4 | tee -a ./log_out/run_$DATASET.txt

python main_train_temporal.py --dataset $DATASET  \
    --ckpt ./ckpt/temporal_base_$DATASET.pkl  --batch-size 2 --max-epoch 30 --lr 1e-5 | tee -a ./log_out/run_$DATASET.txt

# save on detseg_base_[dataset].pkl
python main_template_match.py --dataset $DATASET --max-epoch 50 --ckpt ./ckpt/detseg_base_coco.pkl | tee -a ./log_out/run_$DATASET.txt
python main_template_match.py --dataset $DATASET --max-epoch 50 --tune --lr 1e-4 \
    --ckpt ./ckpt/detseg_base_$DATASET.pkl | tee -a ./log_out/run_$DATASET.txt

######## Test #########
python temporal_match_vid.py --dataset $DATASET --ckpt ./ckpt/temporal_base_$DATASET.pkl \
    --ckptM ./ckpt/detseg_base_$DATASET.pkl --eval-bn | tee -a ./log_out/run_$DATASET.txt