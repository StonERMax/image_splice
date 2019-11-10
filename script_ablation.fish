#!/usr/bin/fish

# usage: fish run_all.sh [cuda-device-ids]

conda activate torch

if test (count $argv) -lt 1
    set -x CUDA_VISIBLE_DEVICES 0
else
    set -x CUDA_VISIBLE_DEVICES $argv[1]    
end

set -x DATASET davis_same
set -x SUFFIX "_wo_gcn"
set -x MODEL base_abl

echo "dataset : " $DATASET
echo "cuda devices: " $CUDA_VISIBLE_DEVICES
echo "suffix: " $SUFFIX 

git checkout ablation{$SUFFIX}

python main_for_vid.py --suffix $SUFFIX --max-epoch 1 

git checkout ablation{$SUFFIX}

python main_train_vid.py --dataset $DATASET --suffix $SUFFIX --model $MODEL \
    --ckpt ./ckpt/{$MODEL}_coco{$SUFFIX}.pkl \
    --max-epoch 7 | tee -a  ./log_out/abl_{$SUFFIX}_{$MODEL}_{$DATASET}.txt

git checkout ablation{$SUFFIX}

set -x SAVE_PATH tmp_video_match_abblation{$SUFFIX}

python match_vid_save.py --dataset $DATASET --model $MODEL --save-path $SAVE_PATH\
    --ckpt ./ckpt/{$MODEL}_{$DATASET}{$SUFFIX}.pkl --num 20 \
    | tee -a  ./log_out/abl_{$SUFFIX}_{$MODEL}_{$DATASET}.txt
python -W ignore match_vid_load.py --dataset $DATASET --model $MODEL \
    --ckpt ./ckpt/{$MODEL}_$DATASET.pkl --num 20 --save-path $SAVE_PATH \
    | tee -a  ./log_out/abl_{$SUFFIX}_{$MODEL}_{$DATASET}.txt