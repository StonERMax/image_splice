conda activate torch

python match_vid_save.py  --ckpt ./ckpt/dmac_tmp_youtube.pkl  --model dmac
python match_vid_save.py  --ckpt ./ckpt/dmvn_tmp_youtube.pkl  --model dmvn
python match_vid_save.py  --ckpt ./ckpt/base_tmp_youtube.pkl  --model base


python match_vid_load.py  --ckpt ./ckpt/dmac_tmp_youtube.pkl  --model dmac >> ./log_out/youtube/dmac.txt
python match_vid_load.py  --ckpt ./ckpt/dmvn_tmp_youtube.pkl  --model dmvn >> ./log_out/youtube/dmvn.txt
python match_vid_load.py  --ckpt ./ckpt/base_tmp_youtube.pkl  --model base >> ./log_out/youtube/base.txt
