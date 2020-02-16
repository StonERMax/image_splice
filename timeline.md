**10/24/19** : trained IFA_NET on `main.py` for 85,000 iteration, saved checkpoint on `base_coco.ckpt`

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


# Feb 16

## main_custom
ckpt custom_coco.pkl
```
difficult:

source 
--------------------------------------------------
precision : 0.7453, recall : 0.6518, f1 : 0.6579, iou : 0.5662 mcc : 0.6667

forge 
--------------------------------------------------
precision : 0.8336, recall : 0.7693, f1 : 0.7805, iou : 0.7000 mcc : 0.7864


medi:

source
--------------------------------------------------
precision : 0.8913, recall : 0.8645, f1 : 0.8675, iou : 0.8049 mcc : 0.8551
   
forge                                  
--------------------------------------------------
precision : 0.9465, recall : 0.9225, f1 : 0.9285, iou : 0.8810 mcc : 0.9234



easy:

source                                                                     
--------------------------------------------------
precision : 0.9499, recall : 0.9455, f1 : 0.9464, iou : 0.9005 mcc : 0.9168
                                                  
forge                                                                      
--------------------------------------------------
precision : 0.9739, recall : 0.9650, f1 : 0.9689, iou : 0.9408 mcc : 0.9559

```
