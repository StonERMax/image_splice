conda activate torch

set -x CUDA_VISIBLE_DEVICES 2
python main_train_vid_dmac.py  --model dmac --max-epoch 30 --suffix _2 --dataset davis
python main_train_vid_dmac.py  --model dmvn --max-epoch 30 --suffix _2 --dataset davis

python match_vid_save.py  --ckpt ./ckpt/dmac_davis_2.pkl  --model dmac --dataset davis
python match_vid_save.py  --ckpt ./ckpt/dmvn_davis_2.pkl  --model dmvn --dataset davis
python match_vid_save.py  --ckpt ./ckpt/base_davis.pkl  --model base --dataset davis


python match_vid_load.py  --ckpt ./ckpt/dmac_davis_2.pkl  --model dmac --dataset davis >> ./log_out/davis/dmac.txt
python match_vid_load.py  --ckpt ./ckpt/dmvn_davis_2.pkl  --model dmvn --dataset davis >> ./log_out/davis/dmvn.txt
python match_vid_load.py  --ckpt ./ckpt/base_davis.pkl  --model base --dataset davis >> ./log_out/davis/base.txt
