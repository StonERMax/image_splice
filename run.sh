
set -x CUDA_VISIBLE_DEVICES 0

# our model
# python main.py --test --ckpt ./ckpt/base_coco_exp_new_with_flip.pkl --num 50 --mode easy | tee log_out/base_coco.txt
# python main.py --test --ckpt ./ckpt/base_coco_exp_new_with_flip.pkl --num 50 --mode medi | tee -a log_out/base_coco.txt
# python main.py --test --ckpt ./ckpt/base_coco_exp_new_with_flip.pkl --num 50 --mode diff | tee -a log_out/base_coco.txt
python main.py --test --ckpt ./ckpt/base_coco_exp_new_with_flip.pkl --num 100 --plot | tee -a log_out/base_coco.txt

# dmac
python main.py --test  --num 50 --mode easy --model dmac | tee -a log_out/dmac_coco.txt
python main.py --test  --num 50 --mode medi --model dmac | tee -a log_out/dmac_coco.txt
python main.py --test  --num 50 --mode diff --model dmac | tee -a log_out/dmac_coco.txt
python main.py --test  --num 100 --plot --model dmac | tee -a log_out/dmac_coco.txt

# dmvn
# python main.py --test  --num 50 --mode easy --model dmvn | tee -a log_out/dmvn_coco.txt
# python main.py --test  --num 50 --mode medi --model dmvn | tee -a log_out/dmvn_coco.txt
# python main.py --test  --num 50 --mode diff --model dmvn | tee -a log_out/dmvn_coco.txt
# python main.py --test  --num 100 --plot --model dmvn | tee -a log_out/dmvn_coco.txt
