set -x CUDA_VISIBLE_DEVICES 2


#### CASIA 2 localization
# our model
# python main_casia.py --dataset casia --ckpt ./ckpt/base_coco_exp_new_with_flip.pkl --num 50 --plot | tee -a log_out/base_casia.txt
# # # dmac
# python main_casia.py --dataset casia --model dmac --num 50 --plot | tee -a log_out/dmac_casia.txt
# # # dmvn
# python main_casia.py --dataset casia --model dmvn --num 50 --plot | tee -a log_out/dmvn_casia.txt



#### CASIA 2 detection
# our model
python main_casia_detection.py --ckpt ./ckpt/base_coco_exp_new_with_flip.pkl --num 500 | tee -a log_out/base_casia.txt
# # dmac
# python main_casia_detection.py --model dmac --num 500 | tee -a log_out/dmac_casia.txt
# # dmvn
# python main_casia_detection.py --model dmvn --num 500 | tee -a log_out/dmvn_casia.txt
