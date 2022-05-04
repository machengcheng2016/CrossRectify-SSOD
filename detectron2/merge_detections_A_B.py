import pickle as pkl
import os
from collections import defaultdict
from ensemble_boxes import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dA', type=str, required=True)
parser.add_argument('--dB', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--weights', metavar='N', type=float, nargs='+', help="[1,1]")
parser.add_argument('--iou_thr', type=float, default=0.60)
parser.add_argument('--skip_box_thr', type=float, default=0.0001)
args = parser.parse_args()

with open(args.dA, 'rb') as f0:
    detections0 = pkl.load(f0)
with open(args.dB, 'rb') as f1:
    detections1 = pkl.load(f1)
    
wbf_detections = defaultdict(list)
for image_id in detections0:
    boxes0, scores0, labels0 = [], [], []
    boxes1, scores1, labels1 = [], [], []
    
    for element in detections0[image_id]:
        cls, score, xmin_float, ymin_float, xmax_float, ymax_float, width0, height0 = element
        boxes0.append([xmin_float, ymin_float, xmax_float, ymax_float])
        scores0.append(score)
        labels0.append(cls)
    assert len(boxes0) == len(scores0) == len(labels0)

    for element in detections1[image_id]:
        cls, score, xmin_float, ymin_float, xmax_float, ymax_float, width1, height1 = element
        boxes1.append([xmin_float, ymin_float, xmax_float, ymax_float])
        scores1.append(score)
        labels1.append(cls)
    assert len(boxes1) == len(scores1) == len(labels1)
    
    if len(boxes0) > 0 and len(boxes1) > 0:
        boxes_list = [boxes0, boxes1]
        scores_list = [scores0, scores1]
        labels_list = [labels0, labels1]
        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=args.weights, iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr)
        assert boxes.shape[0] == scores.shape[0] == labels.shape[0]
        assert width0 == width1
        assert height0 == height1
        num = boxes.shape[0]
        '''
        print("boxes_list")
        for i in boxes_list:
            print(i)
            print('-'*10)
        print('-'*20)
        print("scores_list")
        for i in scores_list:
            print(i)
        print('-'*20)
        print(labels)
        '''
        for i in range(num):
            xmin_float, ymin_float, xmax_float, ymax_float = boxes[i].tolist()
            score = scores[i]
            cls = int(labels[i])
            wbf_detections[image_id].append([cls, score, xmin_float, ymin_float, xmax_float, ymax_float, width0, height0])
    elif len(boxes0) > 0:
        num = len(boxes0)
        for i in range(num):
            xmin_float, ymin_float, xmax_float, ymax_float = boxes0[i]
            score = scores0[i]
            cls = int(labels0[i])
            wbf_detections[image_id].append([cls, score, xmin_float, ymin_float, xmax_float, ymax_float, width0, height0])
    elif len(boxes1) > 0:
        num = len(boxes1)
        for i in range(num):
            xmin_float, ymin_float, xmax_float, ymax_float = boxes1[i]
            score = scores1[i]
            cls = int(labels1[i])
            wbf_detections[image_id].append([cls, score, xmin_float, ymin_float, xmax_float, ymax_float, width1, height1])

with open(os.path.join(args.out, "detections.pkl"), "wb") as f:
    pkl.dump(wbf_detections, f, pkl.HIGHEST_PROTOCOL)
