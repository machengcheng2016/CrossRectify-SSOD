import warnings
warnings.filterwarnings("ignore")

from data import *

from layers.modules import MultiBoxLoss
from utils.augmentations import *
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC300', choices=['VOC300', 'VOC512'],
                    type=str, help='VOC300 or VOC512')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, 
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume2', default=None, type=str, 
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--save_interval', default=2000, type=int,
                    help='Directory for saving checkpoint models')
parser.add_argument('--warmup', default=False, action='store_true',
                    help='Do Supervised Learning in first epoch')
parser.add_argument('--sup_aug_type', default='default', type=str,
                    help='default | autoaugment | gridmask')
parser.add_argument('--unsup_aug_type', default='default', type=str,
                    help='default | autoaugment | gridmask')
parser.add_argument('--ramp', action='store_true',
                    help='whether use ramp')
parser.add_argument('--ramp_weight', type=float, default=1.0,
                    help='whether use ramp')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--ema_rate', type=float, required=True)
parser.add_argument('--ema_step', type=int, required=True)

args = parser.parse_args()

print(args)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("random seed is set as {}".format(seed))

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def nms(boxes, scores, overlap=0.5, top_k=20):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def entropy_loss(logits):
    p = F.softmax(logits, dim=1)
    return -torch.mean(torch.sum(p * F.log_softmax(logits, dim=1), dim=1))

def update_teacher(ckpt_old, ckpt_new):
    for key in ckpt_old.keys():
        if key in ckpt_new.keys():
            ckpt_old[key].data = ckpt_old[key].data * args.ema_rate + ckpt_new[key].data * (1 - args.ema_rate)
        else:
            raise Exception("{} is not found in student model".format(key))
    return ckpt_old

def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
    elif args.dataset == 'VOC300':
        #if args.dataset_root == COCO_ROOT:
        #    parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc300
    elif args.dataset == 'VOC512':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc512

    if args.sup_aug_type == "default":
        img_transform_sup = SSDAugmentation(cfg['min_dim'], MEANS)
    elif (args.sup_aug_type).startswith('autoaugment'):
        from utils.Myautoaugment_utils import AutoAugmenter
        if args.sup_aug_type == 'autoaugment_v0':
            autoaugmenter = AutoAugmenter('v0')
        elif args.sup_aug_type == 'autoaugment_v1':
            autoaugmenter = AutoAugmenter('v1')
        elif args.sup_aug_type == 'autoaugment_v2':
            autoaugmenter = AutoAugmenter('v2')
        elif args.sup_aug_type == 'autoaugment_v3':
            autoaugmenter = AutoAugmenter('v3')
        elif args.sup_aug_type == 'autoaugment_v4':
            autoaugmenter = AutoAugmenter('v4')
        elif args.sup_aug_type == 'autoaugment_custom':
            autoaugmenter = AutoAugmenter('custom')
        else:
            raise ValueError("No such autoaugmenter version, please check.")
        img_transform_sup = Compose([ConvertFromInts(),
                                       ToAbsoluteCoords(),
                                       PhotometricDistort(),
                                       Expand(MEANS),
                                       RandomSampleCrop(),
                                       RandomMirror(),
                                       ToPercentCoords(),
                                       autoaugmenter,
                                       Resize(cfg['min_dim']),
                                       SubtractMeans(MEANS),
                                       ])
    elif args.sup_aug_type == "gridmask":
        from utils.MyGridMask import GridMask
        # default setting in https://github.com/Jia-Research-Lab/GridMask/blob/master/detection_grid/maskrcnn_benchmark/config/defaults.py
        GRID_ROTATE = 1
        GRID_OFFSET = 0
        GRID_RATIO = 0.5
        GRID_MODE = 1
        GRID_PROB = 0.5
        img_transform_sup = Compose([ConvertFromInts(),
                                      ToAbsoluteCoords(),
                                      PhotometricDistort(),
                                      Expand(MEANS),
                                      RandomSampleCrop(),
                                      RandomMirror(),
                                      ToPercentCoords(),
                                      GridMask(True, True, GRID_ROTATE, GRID_OFFSET, GRID_RATIO, GRID_MODE, GRID_PROB),
                                      Resize(cfg['min_dim']),
                                      SubtractMeans(MEANS),
                                      ])
    else:
        raise ValueError("args.sup_aug_type should be in [default | autoaugment | gridmask]")

    if args.unsup_aug_type == "default":
        img_transform_unsup = SSDAugmentation(cfg['min_dim'], MEANS)
    elif (args.unsup_aug_type).startswith('autoaugment'):
        from utils.Myautoaugment_utils import AutoAugmenter
        if args.unsup_aug_type == 'autoaugment_v0':
            autoaugmenter = AutoAugmenter('v0')
        elif args.unsup_aug_type == 'autoaugment_v1':
            autoaugmenter = AutoAugmenter('v1')
        elif args.unsup_aug_type == 'autoaugment_v2':
            autoaugmenter = AutoAugmenter('v2')
        elif args.unsup_aug_type == 'autoaugment_v3':
            autoaugmenter = AutoAugmenter('v3')
        elif args.unsup_aug_type == 'autoaugment_v4':
            autoaugmenter = AutoAugmenter('v4')
        elif args.unsup_aug_type == 'autoaugment_custom':
            autoaugmenter = AutoAugmenter('custom')
        else:
            raise ValueError("No such autoaugmenter version, please check.")
        img_transform_unsup = Compose([ConvertFromInts(),
                                       ToAbsoluteCoords(),
                                       PhotometricDistort(),
                                       Expand(MEANS),
                                       RandomSampleCrop(),
                                       RandomMirror(),
                                       ToPercentCoords(),
                                       autoaugmenter,
                                       Resize(cfg['min_dim']),
                                       SubtractMeans(MEANS),
                                       ])
    elif args.unsup_aug_type == "gridmask":
        from utils.MyGridMask import GridMask
        # default setting in https://github.com/Jia-Research-Lab/GridMask/blob/master/detection_grid/maskrcnn_benchmark/config/defaults.py
        GRID_ROTATE = 1
        GRID_OFFSET = 0
        GRID_RATIO = 0.5
        GRID_MODE = 1
        GRID_PROB = 0.5
        img_transform_unsup = Compose([ConvertFromInts(),
                                      ToAbsoluteCoords(),
                                      PhotometricDistort(),
                                      Expand(MEANS),
                                      RandomSampleCrop(),
                                      RandomMirror(),
                                      ToPercentCoords(),
                                      GridMask(True, True, GRID_ROTATE, GRID_OFFSET, GRID_RATIO, GRID_MODE, GRID_PROB),
                                      Resize(cfg['min_dim']),
                                      SubtractMeans(MEANS),
                                      ])
    else:
        raise ValueError("args.unsup_aug_type should be in [default | autoaugment | gridmask]")

    # while finish_flag:
    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net
    ssd_net_teacher = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net_teacher = ssd_net_teacher
    ssd_net2 = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net2 = ssd_net2
    ssd_net2_teacher = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net2_teacher = ssd_net2_teacher
    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        net_teacher = torch.nn.DataParallel(ssd_net_teacher)
        net2 = torch.nn.DataParallel(ssd_net2)
        net2_teacher = torch.nn.DataParallel(ssd_net2_teacher)

    if args.resume and args.resume2:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
        ssd_net_teacher.load_weights(args.resume)
        ssd_net2.load_weights(args.resume2)
        ssd_net2_teacher.load_weights(args.resume2)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)
        ssd_net_teacher.vgg.load_state_dict(vgg_weights)
        ssd_net2.vgg.load_state_dict(vgg_weights)
        ssd_net2_teacher.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()
        net_teacher = net_teacher.cuda()
        net2 = net2.cuda()
        net2_teacher = net2_teacher.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        
        ssd_net_teacher.extras.apply(weights_init)
        ssd_net_teacher.loc.apply(weights_init)
        ssd_net_teacher.conf.apply(weights_init)
        
        ssd_net2.extras.apply(weights_init)
        ssd_net2.loc.apply(weights_init)
        ssd_net2.conf.apply(weights_init)
        
        ssd_net2_teacher.extras.apply(weights_init)
        ssd_net2_teacher.loc.apply(weights_init)
        ssd_net2_teacher.conf.apply(weights_init)
        
    setup_seed(args.seed)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    
    net.train()
    net_teacher.train()
    net2.train()
    net2_teacher.train()
    
    print('Loading the dataset...')

    step_index = 0

    supervised_batch = args.batch_size
    
    supervised_dataset = VOCDetection_con(root=args.dataset_root, img_transform_sup=img_transform_sup, img_transform_unsup=img_transform_unsup, white_box=True)
    supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True, drop_last=True)
    batch_iterator = iter(supervised_data_loader)

    supervised_dataset2 = VOCDetection_con(root=args.dataset_root, img_transform_sup=img_transform_sup, img_transform_unsup=img_transform_unsup, white_box=True)
    supervised_data_loader2 = data.DataLoader(supervised_dataset2, supervised_batch, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True, drop_last=True)
    batch_iterator2 = iter(supervised_data_loader2)
    
    priors_single = torch.load("priors.pt")

    for iteration in range(cfg['max_iter'] // (args.batch_size // 32)):

        if iteration * (args.batch_size // 32) in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
            adjust_learning_rate(optimizer2, args.gamma, step_index)

        if iteration < int((args.resume).split('_')[-1].split('.')[0]):
            continue
            
        try:
            images, targets, semis = next(batch_iterator)
            images2, targets2, semis2 = next(batch_iterator2)
        except StopIteration:
            print("!"*50, iteration, "Line 297 happened!")
            supervised_dataset = VOCDetection_con(root=args.dataset_root, img_transform_sup=img_transform_sup, img_transform_unsup=img_transform_unsup, white_box=True)
            supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True, drop_last=True)
            batch_iterator = iter(supervised_data_loader)
            
            supervised_dataset2 = VOCDetection_con(root=args.dataset_root, img_transform_sup=img_transform_sup, img_transform_unsup=img_transform_unsup, white_box=True)
            supervised_data_loader2 = data.DataLoader(supervised_dataset2, supervised_batch, num_workers=args.num_workers, shuffle=True, collate_fn=detection_collate, pin_memory=True, drop_last=True)
            batch_iterator2 = iter(supervised_data_loader2)
            
            images,  targets,  semis = next(batch_iterator)
            images2, targets2, semis2 = next(batch_iterator2)
            
        if args.cuda:
            images = images.cuda()
            images2 = images2.cuda()
            with torch.no_grad():
                targets = [ann.cuda() for ann in targets]
                targets2 = [ann2.cuda() for ann2 in targets2]

        targets_unsup = targets.copy()
        targets_unsup2 = targets2.copy()
        
        # forward
        t0 = time.time()

        out = net(images)
        out2 = net2(images2)
        
        with torch.no_grad():
            out_teacher = net2_teacher(images)
            out2_teacher = net_teacher(images2)
            
        sup_image_binary_index = np.zeros([len(semis),1])
        for super_image in range(len(semis)):
            if int(semis[super_image]) == 1:
                sup_image_binary_index[super_image] = 1
            else:
                sup_image_binary_index[super_image] = 0
            if int(semis[len(semis)-1-super_image]) == 0:
                del targets[len(semis)-1-super_image]
            else:
                del targets_unsup[len(semis)-1-super_image]

        sup_image_index = np.where(sup_image_binary_index == 1)[0]
        unsup_image_index = np.where(sup_image_binary_index == 0)[0]
        
        sup_image_binary_index2 = np.zeros([len(semis2),1])
        for super_image2 in range(len(semis2)):
            if int(semis2[super_image2]) == 1:
                sup_image_binary_index2[super_image2] = 1
            else:
                sup_image_binary_index2[super_image2] = 0
            if int(semis2[len(semis2)-1-super_image2]) == 0:
                del targets2[len(semis2)-1-super_image2]
            else:
                del targets_unsup2[len(semis2)-1-super_image2]
        
        sup_image_index2 = np.where(sup_image_binary_index2 == 1)[0]
        unsup_image_index2 = np.where(sup_image_binary_index2 == 0)[0]

        loc, conf, priors = out
        loc2, conf2, priors = out2
        with torch.no_grad():
            loc_teacher,  conf_teacher,  _ = out_teacher
            loc2_teacher, conf2_teacher, _ = out2_teacher

        loss_l = torch.cuda.FloatTensor([0])
        loss_c = torch.cuda.FloatTensor([0])
        loss_l2 = torch.cuda.FloatTensor([0])
        loss_c2 = torch.cuda.FloatTensor([0])
        if not args.cuda:
            loss_l = loss_l.cpu()
            loss_c = loss_c.cpu()
            loss_l2 = loss_l2.cpu()
            loss_c2 = loss_c2.cpu()
            
        if len(sup_image_index) > 0:
            loc_data, conf_data = loc[sup_image_index,:,:], conf[sup_image_index,:,:]
            output = (loc_data, conf_data, priors)
            loss_l, loss_c = criterion(output, targets)
            
        if len(sup_image_index2) > 0:
            loc_data2, conf_data2 = loc2[sup_image_index2,:,:], conf2[sup_image_index2,:,:]
            output2 = (loc_data2, conf_data2, priors)
            loss_l2, loss_c2 = criterion(output2, targets2)

        loss_l_unsup_x = torch.cuda.FloatTensor([0])
        loss_l_unsup_y = torch.cuda.FloatTensor([0])
        loss_l_unsup_w = torch.cuda.FloatTensor([0])
        loss_l_unsup_h = torch.cuda.FloatTensor([0])
        loss_c_unsup = torch.cuda.FloatTensor([0])
        if not args.cuda:
            loss_l_unsup_x = loss_l_unsup_x.cpu()
            loss_l_unsup_y = loss_l_unsup_y.cpu()
            loss_l_unsup_w = loss_l_unsup_w.cpu()
            loss_l_unsup_h = loss_l_unsup_h.cpu()
            loss_c_unsup = loss_c_unsup.cpu()
        
        if len(unsup_image_index) > 0:
            N1, n1, m1 = 0, 0, 0
            fp1, fp_right1, tp_wrong1 = 0, 0, 0
            loc_data_unsup = loc[unsup_image_index,:,:]
            conf_data_unsup = conf[unsup_image_index,:,:]
            loc_data_unsup_teacher = loc_teacher[unsup_image_index,:,:]
            conf_data_unsup_teacher = conf_teacher[unsup_image_index,:,:]
            
            for i_unsup in range(len(unsup_image_index)):
                target_unsup = targets_unsup[i_unsup]
                conf_data_unsup_i = F.softmax(conf_data_unsup[i_unsup,:,:], dim=-1)
                conf_class = conf_data_unsup_i[:,1:].detach().clone()
                background_score = conf_data_unsup_i[:,0].detach().clone()
                each_val, each_index = torch.max(conf_class, dim=1)
                mask_val_fg = (each_val > 0.5).data
                
                if mask_val_fg.sum() > 0:
                    decoded_boxes = decode(loc_data_unsup[i_unsup, :, :].detach().clone(), priors_single, cfg['variance'])
                    loc_unsup = torch.clamp(decoded_boxes[mask_val_fg, :], 0.0, 1.0)
                    overlaps = jaccard(loc_unsup, target_unsup[:,:-1])
                    overlaps_max = overlaps.max(dim=1)[0]
                    mask_pseudo = overlaps_max >= 0.0
                    pseudo_label = target_unsup[:,-1][overlaps.max(dim=1)[1]].long() + 1
                    if mask_pseudo.sum() > 0:
                        # NMS by foreground confidence
                        conf_data_unsup_i_select = conf_data_unsup_i[mask_val_fg,:][mask_pseudo,:].detach().clone()
                        scores_fg_max_selected = conf_data_unsup_i_select[:,1:].max(dim=-1)[0]
                        loc_unsup_selected = loc_unsup[mask_pseudo,:]
                        ids_nms, count_nms = nms(loc_unsup_selected, scores_fg_max_selected, overlap=0.5)
                        
                        # NMS by IOU with GT
                        #loc_unsup_selected = loc_unsup[mask_pseudo,:]
                        #ids_nms, count_nms = nms(loc_unsup_selected, overlaps_max[mask_pseudo], overlap=0.5)
                        
                        indices_nms = ids_nms[:count_nms]
                        
                        loc_data_unsup_i_selected          = loc_data_unsup[i_unsup,:,:][mask_val_fg,:][mask_pseudo,:][indices_nms,:]
                        loc_data_unsup_i_selected_teacher  = loc_data_unsup_teacher[i_unsup,:,:][mask_val_fg,:][mask_pseudo,:][indices_nms,:].detach().clone()
                        conf_data_unsup_i_selected         = conf_data_unsup[i_unsup,:,:][mask_val_fg,:][mask_pseudo,:][indices_nms,:]
                        conf_data_unsup_i_selected_teacher = conf_data_unsup_teacher[i_unsup,:,:][mask_val_fg,:][mask_pseudo,:][indices_nms,:].detach().clone()
                        label_unsup_i_selected = pseudo_label[mask_pseudo][indices_nms]
                        loc_unsup_selected = loc_unsup_selected[indices_nms,:]
                        
                        overlaps = jaccard(loc_unsup_selected, target_unsup[:,:-1])
                        overlaps_max = overlaps.max(dim=1)[0]
                        mask_incorrect = (overlaps_max < 0.5).data
                        label_unsup_i_selected[mask_incorrect] = 0
                        
                        score_data_unsup_i_selected = F.softmax(conf_data_unsup_i_selected, dim=-1)
                        score_data_unsup_i_selected_teacher = F.softmax(conf_data_unsup_i_selected_teacher, dim=-1)
                        
                        mask_greater = (score_data_unsup_i_selected.max(dim=-1)[0] > score_data_unsup_i_selected_teacher.max(dim=-1)[0]).float()
                    
                        pseudo_label_unsup_i_selected = mask_greater * score_data_unsup_i_selected.argmax(dim=-1) + (1-mask_greater) * score_data_unsup_i_selected_teacher.argmax(dim=-1)
                        pseudo_label_unsup_i_selected = pseudo_label_unsup_i_selected.long()
                        
                        N1 += conf_data_unsup_i_selected.shape[0]
                        m1 += torch.sum(conf_data_unsup_i_selected.argmax(dim=-1)  == label_unsup_i_selected).item()
                        n1 += torch.sum(pseudo_label_unsup_i_selected == label_unsup_i_selected).item()
                        loss_l_unsup_x += F.mse_loss(loc_data_unsup_i_selected[:,0], loc_data_unsup_i_selected_teacher[:,0], size_average=False, reduce=False).sum(-1)
                        loss_l_unsup_y += F.mse_loss(loc_data_unsup_i_selected[:,1], loc_data_unsup_i_selected_teacher[:,1], size_average=False, reduce=False).sum(-1)
                        loss_l_unsup_w += F.mse_loss(loc_data_unsup_i_selected[:,2], loc_data_unsup_i_selected_teacher[:,2], size_average=False, reduce=False).sum(-1)
                        loss_l_unsup_h += F.mse_loss(loc_data_unsup_i_selected[:,3], loc_data_unsup_i_selected_teacher[:,3], size_average=False, reduce=False).sum(-1)
                        loss_c_unsup += F.cross_entropy(conf_data_unsup_i_selected,  pseudo_label_unsup_i_selected, size_average=False, reduce=False).sum(-1)
                        
                        fp_indices1 = (conf_data_unsup_i_selected.argmax(dim=-1) != label_unsup_i_selected).data
                        fp1 += torch.sum(fp_indices1).item()
                        fp_right1 += torch.sum(pseudo_label_unsup_i_selected[fp_indices1] == label_unsup_i_selected[fp_indices1]).item()
                        tp_indices1 = (conf_data_unsup_i_selected.argmax(dim=-1) == label_unsup_i_selected).data
                        tp_wrong1 += torch.sum(pseudo_label_unsup_i_selected[tp_indices1] != label_unsup_i_selected[tp_indices1]).item()

            if N1 > 0:
                loss_l_unsup_x = torch.div(loss_l_unsup_x, N1)
                loss_l_unsup_y = torch.div(loss_l_unsup_y, N1)
                loss_l_unsup_w = torch.div(loss_l_unsup_w, N1)
                loss_l_unsup_h = torch.div(loss_l_unsup_h, N1)
                loss_c_unsup  = torch.div(loss_c_unsup, N1)
                
            print('A', N1, n1, m1, fp1, fp_right1, tp_wrong1)
                
        loss_l_unsup2_x = torch.cuda.FloatTensor([0])
        loss_l_unsup2_y = torch.cuda.FloatTensor([0])
        loss_l_unsup2_w = torch.cuda.FloatTensor([0])
        loss_l_unsup2_h = torch.cuda.FloatTensor([0])
        loss_c_unsup2 = torch.cuda.FloatTensor([0])
        if not args.cuda:
            loss_l_unsup2_x = loss_l_unsup2_x.cpu()
            loss_l_unsup2_y = loss_l_unsup2_y.cpu()
            loss_l_unsup2_w = loss_l_unsup2_w.cpu()
            loss_l_unsup2_h = loss_l_unsup2_h.cpu()
            loss_c_unsup2 = loss_c_unsup2.cpu()
            
        if len(unsup_image_index2) > 0:
            N2, n2, m2 = 0, 0, 0
            fp2, fp_right2, tp_wrong2 = 0, 0, 0
            loc_data_unsup2 = loc2[unsup_image_index2,:,:]
            conf_data_unsup2 = conf2[unsup_image_index2,:,:]
            loc_data_unsup2_teacher = loc2_teacher[unsup_image_index2,:,:]
            conf_data_unsup2_teacher = conf2_teacher[unsup_image_index2,:,:]
            
            for i_unsup in range(len(unsup_image_index2)):
                target_unsup2 = targets_unsup2[i_unsup]
                conf_data_unsup_i2 = F.softmax(conf_data_unsup2[i_unsup,:,:], dim=-1)
                conf_class2 = conf_data_unsup_i2[:,1:].detach().clone()
                background_score2 = conf_data_unsup_i2[:,0].detach().clone()
                each_val2, each_index2 = torch.max(conf_class2, dim=1)
                mask_val_fg2 = (each_val2 > 0.5).data
                
                if mask_val_fg2.sum() > 0:
                    decoded_boxes2 = decode(loc_data_unsup2[i_unsup, :, :].detach().clone(), priors_single, cfg['variance'])
                    loc_unsup2 = torch.clamp(decoded_boxes2[mask_val_fg2, :], 0.0, 1.0)
                    overlaps2 = jaccard(loc_unsup2, target_unsup2[:,:-1])
                    overlaps_max2 = overlaps2.max(dim=1)[0]
                    mask_pseudo2 = overlaps_max2 >= 0.0
                    pseudo_label2 = target_unsup2[:,-1][overlaps2.max(dim=1)[1]].long() + 1
                    if mask_pseudo2.sum() > 0:
                        # NMS by foreground confidence
                        conf_data_unsup_i_select2 = conf_data_unsup_i2[mask_val_fg2,:][mask_pseudo2,:].detach().clone()
                        scores_fg_max_selected2 = conf_data_unsup_i_select2[:,1:].max(dim=-1)[0]
                        loc_unsup_selected2 = loc_unsup2[mask_pseudo2,:]
                        ids_nms2, count_nms2 = nms(loc_unsup_selected2, scores_fg_max_selected2, overlap=0.5)
                        
                        # NMS by IOU with GT
                        #loc_unsup_selected = loc_unsup[mask_pseudo,:]
                        #ids_nms, count_nms = nms(loc_unsup_selected, overlaps_max[mask_pseudo], overlap=0.5)
                        
                        indices_nms2 = ids_nms2[:count_nms2]
                        
                        loc_data_unsup_i_selected2 = loc_data_unsup2[i_unsup,:,:][mask_val_fg2,:][mask_pseudo2,:][indices_nms2,:]
                        loc_data_unsup_i_selected2_teacher = loc_data_unsup2_teacher[i_unsup,:,:][mask_val_fg2,:][mask_pseudo2,:][indices_nms2,:].detach().clone()
                        conf_data_unsup_i_selected2 = conf_data_unsup2[i_unsup,:,:][mask_val_fg2,:][mask_pseudo2,:][indices_nms2,:]
                        conf_data_unsup_i_selected2_teacher = conf_data_unsup2_teacher[i_unsup,:,:][mask_val_fg2,:][mask_pseudo2,:][indices_nms2,:].detach().clone()
                        label_unsup_i_selected2 = pseudo_label2[mask_pseudo2][indices_nms2]
                        loc_unsup_selected2 = loc_unsup_selected2[indices_nms2,:]
                        
                        overlaps2 = jaccard(loc_unsup_selected2, target_unsup2[:,:-1])
                        overlaps_max2 = overlaps2.max(dim=1)[0]
                        mask_incorrect2 = (overlaps_max2 < 0.5).data
                        label_unsup_i_selected2[mask_incorrect2] = 0
                        
                        score_data_unsup_i_selected2 = F.softmax(conf_data_unsup_i_selected2, dim=-1)
                        score_data_unsup_i_selected2_teacher = F.softmax(conf_data_unsup_i_selected2_teacher, dim=-1)
                        mask_greater2 = (score_data_unsup_i_selected2.max(dim=-1)[0] > score_data_unsup_i_selected2_teacher.max(dim=-1)[0]).float()
                        
                        pseudo_label_unsup_i_selected2 = mask_greater2 * score_data_unsup_i_selected2.argmax(dim=-1) + (1-mask_greater2) * score_data_unsup_i_selected2_teacher.argmax(dim=-1)
                        pseudo_label_unsup_i_selected2 = pseudo_label_unsup_i_selected2.long()
                        
                        N2 += conf_data_unsup_i_selected2.shape[0]
                        m2 += torch.sum(conf_data_unsup_i_selected2.argmax(dim=-1)  == label_unsup_i_selected2).item()
                        n2 += torch.sum(pseudo_label_unsup_i_selected2 == label_unsup_i_selected2).item()
                        loss_l_unsup2_x += F.mse_loss(loc_data_unsup_i_selected2[:,0], loc_data_unsup_i_selected2_teacher[:,0], size_average=False, reduce=False).sum(-1)
                        loss_l_unsup2_y += F.mse_loss(loc_data_unsup_i_selected2[:,1], loc_data_unsup_i_selected2_teacher[:,1], size_average=False, reduce=False).sum(-1)
                        loss_l_unsup2_w += F.mse_loss(loc_data_unsup_i_selected2[:,2], loc_data_unsup_i_selected2_teacher[:,2], size_average=False, reduce=False).sum(-1)
                        loss_l_unsup2_h += F.mse_loss(loc_data_unsup_i_selected2[:,3], loc_data_unsup_i_selected2_teacher[:,3], size_average=False, reduce=False).sum(-1)
                        loss_c_unsup2 += F.cross_entropy(conf_data_unsup_i_selected2, pseudo_label_unsup_i_selected2, size_average=False, reduce=False).sum(-1)
                        
                        fp_indices2 = (conf_data_unsup_i_selected2.argmax(dim=-1) != label_unsup_i_selected2).data
                        fp2 += torch.sum(fp_indices2).item()
                        fp_right2 += torch.sum(pseudo_label_unsup_i_selected2[fp_indices2] == label_unsup_i_selected2[fp_indices2]).item()
                        tp_indices2 = (conf_data_unsup_i_selected2.argmax(dim=-1) == label_unsup_i_selected2).data
                        tp_wrong2 += torch.sum(pseudo_label_unsup_i_selected2[tp_indices2] != label_unsup_i_selected2[tp_indices2]).item()

            if N2 > 0:
                loss_l_unsup2_x = torch.div(loss_l_unsup2_x, N2)
                loss_l_unsup2_y = torch.div(loss_l_unsup2_y, N2)
                loss_l_unsup2_w = torch.div(loss_l_unsup2_w, N2)
                loss_l_unsup2_h = torch.div(loss_l_unsup2_h, N2)
                loss_c_unsup2 = torch.div(loss_c_unsup2, N2)
                
            print('B', N2, n2, m2, fp2, fp_right2, tp_wrong2)
        
        if args.ramp:
            ramp_weight = rampweight(iteration)
        else:
            ramp_weight = 1.0
            
        if args.ramp_weight != 1.0:
            ramp_weight *= args.ramp_weight
            
        loss_l_unsup_x = torch.mul(loss_l_unsup_x, ramp_weight)
        loss_l_unsup_y = torch.mul(loss_l_unsup_y, ramp_weight)
        loss_l_unsup_w = torch.mul(loss_l_unsup_w, ramp_weight)
        loss_l_unsup_h = torch.mul(loss_l_unsup_h, ramp_weight)
        loss_c_unsup  = torch.mul(loss_c_unsup,  ramp_weight)
        
        loss_l_unsup2_x = torch.mul(loss_l_unsup2_x, ramp_weight)
        loss_l_unsup2_y = torch.mul(loss_l_unsup2_y, ramp_weight)
        loss_l_unsup2_w = torch.mul(loss_l_unsup2_w, ramp_weight)
        loss_l_unsup2_h = torch.mul(loss_l_unsup2_h, ramp_weight)
        loss_c_unsup2 = torch.mul(loss_c_unsup2, ramp_weight)
                    
        loss  = loss_l  + loss_c  + torch.div(loss_l_unsup_x  + loss_l_unsup_y  + loss_l_unsup_w  + loss_l_unsup_h,  4) + loss_c_unsup
        loss2 = loss_l2 + loss_c2 + torch.div(loss_l_unsup2_x + loss_l_unsup2_y + loss_l_unsup2_w + loss_l_unsup2_h, 4) + loss_c_unsup2
        
        if float(loss.item()) > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if float(loss2.item()) > 0:
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        # Updating teacher ckpt
        if iteration % args.ema_step == 0:
            net_teacher.load_state_dict(update_teacher(net_teacher.state_dict(), net.state_dict()))
            net2_teacher.load_state_dict(update_teacher(net2_teacher.state_dict(), net2.state_dict()))

        t1 = time.time()                

        if iteration % 200 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print("iter {} || loss: {:.4f} | loss_c: {:.4f}, loss_l: {:.4f}, loss_c_unsup: {:.4f}, lr: {:.4f}, super_len: {}, unsuper_len: {}".format(iteration, loss.item(), loss_c.item(), loss_l.item(), loss_c_unsup.item(), float(optimizer.param_groups[0]['lr']), len(sup_image_index), len(unsup_image_index)))
            print("loss2: {:.4f} | loss_c2: {:.4f}, loss_l2: {:.4f}, loss_c_unsup2: {:.4f}, lr2: {:.4f}, super_len: {}, unsuper_len: {}".format(loss2.item(), loss_c2.item(), loss_l2.item(), loss_c_unsup2.item(), float(optimizer2.param_groups[0]['lr']), len(sup_image_index), len(unsup_image_index)))

        if float(loss.item()) > 1000 or float(loss2.item()) > 1000:
            # raise ValueError("Whoa! loss.item() is larger than 100, something must be wrong!")
            break

        if iteration != 0 and (iteration+1) % args.save_interval == 0:
            print('Saving state, iter:', iteration)
            save_path = "only_conf"
            if args.ramp:
                save_path += "-ramp"
            if args.ramp_weight != 1.0:
                save_path += "-weight{}".format(args.ramp_weight)
                
            if not os.path.exists(os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path)):
                os.makedirs(os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path))
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo137_student_betterInitial_A_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))
            torch.save(ssd_net_teacher.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo137_teacher_betterInitial_A_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))
            torch.save(ssd_net2.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo137_student_betterInitial_B_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))
            torch.save(ssd_net2_teacher.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo137_teacher_betterInitial_B_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))

    print('-------------------------------\n')
    print(loss.item())
    print('-------------------------------')


def rampweight(iteration):
    ramp_up_end = 32000 // (args.batch_size // 32)
    ramp_down_start = 100000 // (args.batch_size // 32)

    if iteration < ramp_up_end:
        ramp_weight = math.exp(-5 * math.pow((1 - iteration / ramp_up_end), 2))
    elif iteration > ramp_down_start:
        ramp_weight = math.exp(-12.5 * math.pow((1 - (120000 // (args.batch_size // 32) - iteration) / (20000 // (args.batch_size // 32))), 2)) 
    else:
        ramp_weight = 1 

    if iteration == 0:
        ramp_weight = 0

    return ramp_weight
    
def thresholding(iteration):
    if iteration < 12000:
        threshold = 0.80
    elif iteration < 120000:
        threshold = 0.80 + (0.90-0.80) / (120000-12000) * (iteration-12000)
    return threshold

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()

