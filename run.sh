conda activate torch

set -x CUDA_VISIBLE_DEVICES 0

python main_train_vid_dmac.py  --model dmac --max-epoch 50 --suffix _2 --lr 1e-4
python main_train_vid_dmac.py  --model dmvn --max-epoch 50 --suffix _2 --lr 1e-4

python match_vid_save.py  --ckpt ./ckpt/dmac_tmp_youtube_2.pkl  --model dmac
python match_vid_save.py  --ckpt ./ckpt/dmvn_tmp_youtube_2.pkl  --model dmvn
# python match_vid_save.py  --ckpt ./ckpt/base_tmp_youtube.pkl  --model base


python match_vid_load.py  --ckpt ./ckpt/dmac_tmp_youtube_2.pkl --thres 0.35  --model dmac >> ./log_out/youtube/dmac.txt
python match_vid_load.py  --ckpt ./ckpt/dmvn_tmp_youtube_2.pkl --thres 0.35 --model dmvn >> ./log_out/youtube/dmvn.txt
# python match_vid_load.py  --ckpt ./ckpt/base_tmp_youtube_2.pkl  --model base >> ./log_out/youtube/base.txt
