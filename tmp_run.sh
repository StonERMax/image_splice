
pkill -9 python
conda activate torch
env CUDA_VISIBLE_DEVICES=1 python main_train_vid.py --batch-size 15 \
  --model doanet_a2_wo_gcn --max-epoch 30 \
  --ckpt ./ckpt/doanet_a2_wo_gcn_coco.pkl  >> ./log_out/tmp_sh.txt
