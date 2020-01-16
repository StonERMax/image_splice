**10/24/19** : trained IFA_NET on `main.py` for 85,000 iteration, saved checkpoint on `base_coco.pkl`

**10/30/2019**: trained forge-detection on `main_mani_det_vid.py`, saved to *detseg_base_tmp_youtube.pkl*. Tested on the whole youtube-tempered dataset, the result is:
```python
Image level score                                                    
precision: 0.9564, recall: 0.8881, f-score: 0.9210
```
---
### Training Procedure:
#### Frame-based training
1. First train on coco with `main.py`, save checkpoint on `base_coco.pkl`
2. Then train on video with `main_train_vid.py` with `base_coco.pkl`, save it on `base_[vid].pkl`
3. At the same time, train `main_template_match.py` on video, and save in `detseg_[vid].pkl`

#### Video-based training
4. Run `main_train_temporal.py` with pretrained model from (2), save it on `temporal_[vid].py`

#### Testing
5. Run `temporal_match_vid.py` with pretrained model from (4) [--ckpt] and (3) [--ckptM]

##### Just run 'scipt.sh' to execute everythong
---