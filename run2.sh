conda activate torch

# env CUDA_VISIBLE_DEVICES=1 python  main.py --model dmac --batch-size 15 --max-epoch 20 --test >> ./log_out/coco/dmac_all.txt
# env CUDA_VISIBLE_DEVICES=1 python  main.py --model dmac --batch-size 15 --max-epoch 20 --test --mode easy >> ./log_out/coco/dmac_easy.txt
# env CUDA_VISIBLE_DEVICES=1  python main.py --model dmac --batch-size 15 --max-epoch 20 --test --mode medi >> ./log_out/coco/dmac_medi.txt
# env CUDA_VISIBLE_DEVICES=1  python main.py --model dmac --batch-size 15 --max-epoch 20 --test --mode diff >> ./log_out/coco/dmac_diff.txt

# env CUDA_VISIBLE_DEVICES=1  python main.py --model dmvn --batch-size 15 --max-epoch 20 --test >> ./log_out/coco/dmvn_all.txt
# env CUDA_VISIBLE_DEVICES=1  python main.py --model dmvn --batch-size 15 --max-epoch 20 --test --mode easy >> ./log_out/coco/dmvn_easy.txt
# env CUDA_VISIBLE_DEVICES=1  python main.py --model dmvn --batch-size 15 --max-epoch 20 --test --mode medi >> ./log_out/coco/dmvn_medi.txt
# env CUDA_VISIBLE_DEVICES=1   python main.py --model dmvn --batch-size 15 --max-epoch 20 --test --mode diff >> ./log_out/coco/dmvn_diff.txt

env CUDA_VISIBLE_DEVICES=1 python  main.py --model base --batch-size 15 --max-epoch 20 --test --ckpt ./ckpt/gcn2_coco.pkl >> ./log_out/coco/ours_all.txt
env CUDA_VISIBLE_DEVICES=1 python  main.py --model base --batch-size 15 --max-epoch 20 --test --mode easy --ckpt ./ckpt/gcn2_coco.pkl >> ./log_out/coco/ours_easy.txt
env CUDA_VISIBLE_DEVICES=1  python main.py --model base --batch-size 15 --max-epoch 20 --test --mode medi --ckpt ./ckpt/gcn2_coco.pkl >> ./log_out/coco/ours_medi.txt
env CUDA_VISIBLE_DEVICES=1  python main.py --model base --batch-size 15 --max-epoch 20 --test --mode diff --ckpt ./ckpt/gcn2_coco.pkl >> ./log_out/coco/ours_diff.txt


