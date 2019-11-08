conda activate torch

set -x CUDA_VISIBLE_DEVICES 0

python main.py --ckpt ./ckpt/base_coco_exp_new_comb.pkl --test --mode easy | tee log_out/cisl.txt
python main.py --ckpt ./ckpt/base_coco_exp_new_comb.pkl --test --mode medi | tee -a log_out/cisl.txt
python main.py --ckpt ./ckpt/base_coco_exp_new_comb.pkl --test --mode diff | tee -a log_out/cisl.txt
python main.py --ckpt ./ckpt/base_coco_exp_new_comb.pkl --test | tee -a log_out/cisl.txt