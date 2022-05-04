# CrossRectify-SSOD
## 0. Introduction
Official code of "CrossRectify: Leveraging Disagreement for Semi-supervised Object Detection"

This repo includes training SSD300 on Pascal VOC, training Faster-RCNN-FPN on Pascal VOC, and training Faster-RCNN-FPN on MS-COCO.
The scripts about training SSD300 are based on [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch/), contained in ```SSD300```.
The scripts about training Faster-RCNN-FPN on Pascal VOC are based on [the official Detectron2 (v0.4) repo](https://github.com/facebookresearch/detectron2/tree/v0.4/), contained in ```detectron2```.
The scripts about training Faster-RCNN-FPN on MS-COCO are based on [the official MMDetection (v2.17.0) repo](https://github.com/open-mmlab/mmdetection/tree/v2.17.0/), contained in ```mmdetection```.

## 1. Environment
Python = 3.6.8
CUDA Version = 10.1
Pytorch Version = 1.6.0
detectron2 = 0.4 (training Faster-RCNN-FPN on Pascal VOC)
mmdetection = 2.17.0 (training Faster-RCNN-FPN on MS-COCO)

## 2. Prepare Dataset

### Download and extract the Pascal VOC dataset.
For training SSD300 on Pascal VOC, go into the ```SSD300``` subdirectory and specify the ```VOC_ROOT``` variable in ```data/voc0712.py``` and ```data/voc07_consistency.py``` as ```/path/to/dataset/VOCdevkit/```
For training Faster-RCNN-FPN on Pascal VOC, go into the ```detectron2``` subdirectory and set the environmental variable in this way: ```export DETECTRON2_DATASETS=/path/to/dataset/VOCdevkit/```

### Download and extract the MS-COCO dataset.
For training Faster-RCNN-FPN on MS-COCO, go into the ```mmdetection``` subdirectory and follow the instructions [here](https://github.com/microsoft/SoftTeacher/blob/main/README.md#data-preparation).

## 3. Instructions
### 3.1 Reproduce Table. 1
Go into the ```SSD300``` subdirectory, then run the following scripts to train detectors.

- fully-supervised training (VOC 07 labeled, without extra augmentation): ```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_ssd.py --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, without extra augmentation):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo39.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

- fully-supervised training (VOC 0712 labeled, without extra augmentation):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_ssd0712.py --save_interval 12000```

- fully-supervised training (VOC 07 labeled, with horizontal flip):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_csd_sup2.py --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, with horizontal flip):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_csd.py --save_interval 12000```

- fully-supervised training (VOC 0712 labeled, with horizontal flip):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_csd_sup_0712.py --save_interval 12000```

- fully-supervised training (VOC 07 labeled, with mix-up augmentation):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_isd_sup2.py --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, with mix-up augmentation):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_only_isd.py --save_interval 12000```

- fully-supervised training (VOC 0712 labeled, with mix-up augmentation):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_isd_sup_0712.py --save_interval 12000```

To eval the trained SSD300 on the Pascal VOC test set, run the following script:
- ```CUDA_VISIBLE_DEVICES=0 python3 eval.py --trained_model /path/to/trained/detector/ckpt.pth```

### 3.2 Reproduce Table. 2
Go into the ```SSD300``` subdirectory, then run the following scripts.

- fully-supervised training (VOC 07 labeled, without extra augmentation):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_ssd.py --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, confidence threshold = 0.5):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo39.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, confidence threshold = 0.8):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo39-0.8.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, confidence threshold increasing from 0.5 to 0.8):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo39-0.5-to-0.8.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.5, use TP and discard FP):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo36.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.5, use TP and random labeled FP, confidence threshold=0.5):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo102.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

- self-labeling (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.5, use GT labels for TP and FP, confidence threshold=0.5):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo32.py --resume weights/ssd300_12000.pth --ramp --save_interval 12000```

To eval the trained SSD300 on the Pascal VOC test set, run the following script:
- ```CUDA_VISIBLE_DEVICES=0 python3 eval.py --trained_model /path/to/trained/detector/ckpt.pth```


### 3.3 Reproduce Table.3
Go into the ```SSD300``` subdirectory, then run the following scripts.

- CrossRectify (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.5):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo137.py --resume weights/ssd300_12000.pth --resume2 weights/default/ssd300_12000.2.pth --save_interval 12000 --ramp --ema_rate 0.99 --ema_step 10```

- CrossRectify + mix-up augmentation (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.5):
```CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_pseudo151.py --resume weights/ssd300_12000.pth --resume2 weights/default/ssd300_12000.2.pth --save_interval 12000 --ramp --ema_rate 0.99 --ema_step 10```


To eval the trained SSD300 on the Pascal VOC test set, run the following script:
- ```CUDA_VISIBLE_DEVICES=0 python3 eval.py --trained_model /path/to/trained/detector/ckpt.pth```


Go into the ```detectron2``` subdirectory, then run the following script.

- CrossRectify (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.7):
```python3 train_net.py --resume --num-gpus 8 --config configs/voc/voc07_voc12.yaml MODEL.WEIGHTS output/model_0005999.pth SOLVER.CHECKPOINT_PERIOD 18000```


To eval the trained Faster-RCNN-FPN on the Pascal VOC test set, run the following script:
- ```python3 train_net.py --eval-only --num-gpus 8 --config configs/voc/voc07_voc12.yaml MODEL.WEIGHTS /path/to/trained/detector/ckpt.pth```


Go into the ```mmdetection``` subdirectory, then run the following script.

- CrossRectify (VOC 07 labeled + VOC 12 unlabeled, confidence threshold=0.9):
```bash tools/dist_train_partially.sh semi 1 10 8```

To eval the trained Faster-RCNN-FPN on the MS-COCO test set, run the following script:
- ```bash tools/dist_test.sh work_dirs/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k/10/1/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k.py work_dirs/soft_teacher_faster_rcnn_r50_caffe_fpn_coco_180k/10/1/iter_180000.pth 8 --eval bbox```


## Citation
If you find this work useful, please consider citing our paper. We provide a BibTeX entry of our paper below:
```
@misc{CrossRectify,
  title = {CrossRectify: Leveraging Disagreement for Semi-supervised Object Detection},
  author = {Ma, Chengcheng and Pan, Xingjia and Ye, Qixiang and Tang, Fan and Dong, Weiming and Xu, Changsheng},
  url = {https://arxiv.org/abs/2201.10734},
  year = {2022},
}
```
Feel free to contact [me](machengcheng2016@gmail.com) if you have any questions about our paper or codes.
