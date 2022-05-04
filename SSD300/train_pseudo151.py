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
parser.add_argument('--dataset', default='VOC300',
                    type=str, help='VOC300')
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
                    help='default')
parser.add_argument('--unsup_aug_type', default='default', type=str,
                    help='default')
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

def rotate(l, n):
    return l[-n:] + l[:-n]

def shuffle_tensor(tensor):
    N = tensor.size(0)
    tensor_tmp = tensor.detach().clone()
    indices = list(range(N))
    indices_rotate = rotate(indices, int(N/2))
    tensor_shuffle = tensor_tmp[indices_rotate]
    return tensor_shuffle
    
def train():
    assert args.dataset == 'VOC300'
    cfg = voc300
    
    if args.sup_aug_type == "default":
        img_transform_sup = SSDAugmentation(cfg['min_dim'], MEANS)
    else:
        raise NotImplementedError('only default augmentation is supported')

    if args.unsup_aug_type == "default":
        img_transform_unsup = SSDAugmentation(cfg['min_dim'], MEANS)
    else:
        raise NotImplementedError('only default augmentation is supported')


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
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()
    
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

        images_shuffle = shuffle_tensor(images)
        images_shuffle = flip(images_shuffle, 3)
        lam = np.random.beta(100.0, 100.0)
        images_mix = lam * images.clone() + (1 - lam) * images_shuffle.clone()
        
        images_shuffle2 = shuffle_tensor(images2)
        images_shuffle2 = flip(images_shuffle2, 3)
        lam2 = np.random.beta(100.0, 100.0)
        images_mix2 = lam2 * images2.clone() + (1 - lam2) * images_shuffle2.clone()

        out = net(images)
        out2 = net2(images2)
        out_mix = net(images_mix)
        out2_mix = net2(images_mix2)
        with torch.no_grad():
            out_teacher = net2_teacher(images)
            out2_teacher = net_teacher(images2)
        output_shuffle = net(images_shuffle)
        output2_shuffle = net2(images_shuffle2)
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
        loc_mix, conf_mix, _ = out_mix
        loc2_mix, conf2_mix, _ = out2_mix
        with torch.no_grad():
            loc_teacher,  conf_teacher,  _ = out_teacher
            loc2_teacher, conf2_teacher, _ = out2_teacher
        loc_shuffle, conf_shuffle, priors = output_shuffle
        loc2_shuffle, conf2_shuffle, priors = output2_shuffle

        loss_l = torch.cuda.FloatTensor([0])
        loss_c = torch.cuda.FloatTensor([0])
        loss_l2 = torch.cuda.FloatTensor([0])
        loss_c2 = torch.cuda.FloatTensor([0])
        loss_c_unsup_isd_type1_1 = torch.cuda.FloatTensor([0])
        loss_l_unsup_isd_type2_1 = torch.cuda.FloatTensor([0])
        loss_c_unsup_isd_type2_1 = torch.cuda.FloatTensor([0])
        loss_c_unsup_isd_type1_2 = torch.cuda.FloatTensor([0])
        loss_l_unsup_isd_type2_2 = torch.cuda.FloatTensor([0])
        loss_c_unsup_isd_type2_2 = torch.cuda.FloatTensor([0])
        if not args.cuda:
            loss_l = loss_l.cpu()
            loss_c = loss_c.cpu()
            loss_l2 = loss_l2.cpu()
            loss_c2 = loss_c2.cpu()
            loss_c_unsup_isd_type1_1 = loss_c_unsup_isd_type1_1.cpu()
            loss_l_unsup_isd_type2_1 = loss_l_unsup_isd_type2_1.cpu()
            loss_c_unsup_isd_type2_1 = loss_c_unsup_isd_type2_1.cpu()
            loss_c_unsup_isd_type1_2 = loss_c_unsup_isd_type1_2.cpu()
            loss_l_unsup_isd_type2_2 = loss_l_unsup_isd_type2_2.cpu()
            loss_c_unsup_isd_type2_2 = loss_c_unsup_isd_type2_2.cpu()                
            
        if len(sup_image_index) > 0:
            loc_data, conf_data = loc[sup_image_index,:,:], conf[sup_image_index,:,:]
            loc_mix_data, conf_mix_data = loc_mix[sup_image_index,:,:], conf_mix[sup_image_index,:,:]
            loc_shuffle_data, conf_shuffle_data = loc_shuffle[sup_image_index,:,:], conf_shuffle[sup_image_index,:,:]
            output = (loc_data, conf_data, priors)
            loss_l, loss_c = criterion(output, targets)
            
            # ISD sup loss
            score = F.softmax(conf_data, dim=-1)
            conf_class = score[:,:,1:]
            background_score = score[:, :, 0]
            each_val, each_index = torch.max(conf_class, dim=2)
            mask_val = (each_val > background_score).data
            
            score_shuffle = F.softmax(conf_shuffle_data, dim=-1)
            conf_class_shuffle = score_shuffle[:,:,1:]
            background_score_shuffle = score_shuffle[:, :, 0]
            each_val_shuffle, each_index_shuffle = torch.max(conf_class_shuffle, dim=2)
            mask_val_shuffle = (each_val_shuffle > background_score_shuffle).data
            
            mask_left_right = mask_val.float() * mask_val_shuffle.float()
            mask_left_right = mask_left_right.bool()       
            
            mask_only_left = mask_val.float() * (1 - mask_val_shuffle.float())
            mask_only_left = mask_only_left.bool()
            
            if mask_left_right.sum() > 0:
                mask_left_right_conf_index = mask_left_right.unsqueeze(2).expand_as(conf_data)
                score_sampled = score[mask_left_right_conf_index].view(-1, 21)
                score_shuffle_sampled = score_shuffle[mask_left_right_conf_index].view(-1, 21)
                conf_mix_sampled = conf_mix_data[mask_left_right_conf_index].view(-1, 21)
                score_mix_sampled = lam * score_sampled + (1 - lam) * score_shuffle_sampled + 1e-7
                loss_c_unsup_isd_type1_1a = conf_consistency_criterion(score_mix_sampled.log(), F.softmax(conf_mix_sampled.detach(), dim=-1) + 1e-7).sum(-1).mean()
                loss_c_unsup_isd_type1_1b = conf_consistency_criterion((F.softmax(conf_mix_sampled, dim=-1) + 1e-7).log(), score_mix_sampled.detach()).sum(-1).mean()
                loss_c_unsup_isd_type1_1 = torch.div(loss_c_unsup_isd_type1_1a + loss_c_unsup_isd_type1_1b, 2)
                     
            if mask_only_left.sum() > 0:
                mask_only_left_loc_index = mask_only_left.unsqueeze(2).expand_as(loc_data)
                mask_only_left_conf_index = mask_only_left.unsqueeze(2).expand_as(conf_data)
                loc_sampled = loc_data[mask_only_left_loc_index].view(-1, 4)
                loc_mix_sampled = loc_mix_data[mask_only_left_loc_index].view(-1, 4)
                score_sampled = score[mask_only_left_conf_index].view(-1, 21)
                conf_mix_sampled = conf_mix_data[mask_only_left_conf_index].view(-1, 21)
                loss_l_unsup_isd_type2_1 = F.mse_loss(loc_mix_sampled, loc_sampled.detach())
                loss_c_unsup_isd_type2_1 = conf_consistency_criterion(F.log_softmax(conf_mix_sampled), score_sampled.detach()).sum(-1).mean()
                
            
            
        if len(sup_image_index2) > 0:
            loc2_data, conf2_data = loc2[sup_image_index2,:,:], conf2[sup_image_index2,:,:]
            loc2_mix_data, conf2_mix_data = loc2_mix[sup_image_index2,:,:], conf2_mix[sup_image_index2,:,:]
            loc2_shuffle_data, conf2_shuffle_data = loc2_shuffle[sup_image_index2,:,:], conf2_shuffle[sup_image_index2,:,:]
            output2 = (loc2_data, conf2_data, priors)
            loss_l2, loss_c2 = criterion(output2, targets2)
            
            # ISD sup loss
            score2 = F.softmax(conf2_data, dim=-1)
            conf_class2 = score2[:,:,1:]
            background_score2 = score2[:, :, 0]
            each_val2, each_index2 = torch.max(conf_class2, dim=2)
            mask_val2 = (each_val2 > background_score2).data
            
            score_shuffle2 = F.softmax(conf2_shuffle_data, dim=-1)
            conf_class_shuffle2 = score_shuffle2[:,:,1:]
            background_score_shuffle2 = score_shuffle2[:, :, 0]
            each_val_shuffle2, each_index_shuffle2 = torch.max(conf_class_shuffle2, dim=2)
            mask_val_shuffle2 = (each_val_shuffle2 > background_score_shuffle2).data
            
            mask_left_right2 = mask_val2.float() * mask_val_shuffle2.float()
            mask_left_right2 = mask_left_right2.bool()       
            
            mask_only_left2 = mask_val2.float() * (1 - mask_val_shuffle2.float())
            mask_only_left2 = mask_only_left2.bool()
            
            
            if mask_left_right2.sum() > 0:
                mask_left_right_conf_index2 = mask_left_right2.unsqueeze(2).expand_as(conf2_data)
                score_sampled2 = score2[mask_left_right_conf_index2].view(-1, 21)
                score_shuffle_sampled2 = score_shuffle2[mask_left_right_conf_index2].view(-1, 21)
                conf_mix_sampled2 = conf2_mix_data[mask_left_right_conf_index2].view(-1, 21)
                score_mix_sampled2 = lam * score_sampled2 + (1 - lam) * score_shuffle_sampled2 + 1e-7
                loss_c_unsup_isd_type1_2a = conf_consistency_criterion(score_mix_sampled2.log(), F.softmax(conf_mix_sampled2.detach(), dim=-1) + 1e-7).sum(-1).mean()
                loss_c_unsup_isd_type1_2b = conf_consistency_criterion((F.softmax(conf_mix_sampled2, dim=-1) + 1e-7).log(), score_mix_sampled2.detach()).sum(-1).mean()
                loss_c_unsup_isd_type1_2 = torch.div(loss_c_unsup_isd_type1_2a + loss_c_unsup_isd_type1_2b, 2)
                     
            if mask_only_left2.sum() > 0:
                mask_only_left_loc_index2 = mask_only_left2.unsqueeze(2).expand_as(loc2_data)
                mask_only_left_conf_index2 = mask_only_left2.unsqueeze(2).expand_as(conf2_data)
                loc_sampled2 = loc2_data[mask_only_left_loc_index2].view(-1, 4)
                loc_mix_sampled2 = loc2_mix_data[mask_only_left_loc_index2].view(-1, 4)
                score_sampled2 = score2[mask_only_left_conf_index2].view(-1, 21)
                conf_mix_sampled2 = conf2_mix_data[mask_only_left_conf_index2].view(-1, 21)
                loss_l_unsup_isd_type2_2 = F.mse_loss(loc_mix_sampled2, loc_sampled2.detach())
                loss_c_unsup_isd_type2_2 = conf_consistency_criterion(F.log_softmax(conf_mix_sampled2), score_sampled2.detach()).sum(-1).mean()



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

        loss_c_unsup_isd_type1_1 = torch.mul(loss_c_unsup_isd_type1_1, ramp_weight)
        loss_l_unsup_isd_type2_1 = torch.mul(loss_l_unsup_isd_type2_1, ramp_weight)
        loss_c_unsup_isd_type2_1 = torch.mul(loss_c_unsup_isd_type2_1, ramp_weight)
        loss_c_unsup_isd_type1_2 = torch.mul(loss_c_unsup_isd_type1_2, ramp_weight)
        loss_l_unsup_isd_type2_2 = torch.mul(loss_l_unsup_isd_type2_2, ramp_weight)
        loss_c_unsup_isd_type2_2 = torch.mul(loss_c_unsup_isd_type2_2, ramp_weight)

        loss  = loss_l  + loss_c  + torch.div(loss_l_unsup_x  + loss_l_unsup_y  + loss_l_unsup_w  + loss_l_unsup_h,  4) + loss_c_unsup  + loss_c_unsup_isd_type1_1 * 0.1 + loss_l_unsup_isd_type2_1 + loss_c_unsup_isd_type2_1
        loss2 = loss_l2 + loss_c2 + torch.div(loss_l_unsup2_x + loss_l_unsup2_y + loss_l_unsup2_w + loss_l_unsup2_h, 4) + loss_c_unsup2 + loss_c_unsup_isd_type1_2 * 0.1 + loss_l_unsup_isd_type2_2 + loss_c_unsup_isd_type2_2
        
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
            print("iter {} || loss: {:.4f} | loss_c: {:.4f}, loss_l: {:.4f}, loss_l_unsup: {:.4f}, loss_c_unsup: {:.4f}, loss_isd_type1_c: {:.4f}, loss_isd_type2_l: {:.4f}, loss_isd_type2_c: {:.4f}, lr: {:.4f}, super_len: {}, unsuper_len: {}".format(iteration, loss.item(), loss_c.item(), loss_l.item(), torch.div(loss_l_unsup_x + loss_l_unsup_y + loss_l_unsup_w + loss_l_unsup_h, 4).item(), loss_c_unsup.item(), loss_c_unsup_isd_type1_1.item(), loss_l_unsup_isd_type2_1.item(), loss_c_unsup_isd_type2_1.item(), float(optimizer.param_groups[0]['lr']), len(sup_image_index), len(unsup_image_index)))
            print("loss2: {:.4f} | loss_c2: {:.4f}, loss_l2: {:.4f}, loss_l_unsup2: {:.4f}, loss_c_unsup2: {:.4f}, loss_isd_type1_c2: {:.4f}, loss_isd_type2_l2: {:.4f}, loss_isd_type2_c2: {:.4f}, lr2: {:.4f}, super_len: {}, unsuper_len: {}".format(loss2.item(), loss_c2.item(), loss_l2.item(), torch.div(loss_l_unsup2_x + loss_l_unsup2_y + loss_l_unsup2_w + loss_l_unsup2_h, 4).item(), loss_c_unsup2.item(), loss_c_unsup_isd_type1_2.item(), loss_l_unsup_isd_type2_2.item(), loss_c_unsup_isd_type2_2.item(), float(optimizer2.param_groups[0]['lr']), len(sup_image_index), len(unsup_image_index)))

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
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo151_student_A_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))
            torch.save(ssd_net2.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo151_student_B_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))
            torch.save(ssd_net_teacher.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo151_teacher_A_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))
            torch.save(ssd_net2_teacher.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), save_path, "ssd300_pseudo151_teacher_B_{}_{}_".format(args.ema_rate, args.ema_step) + repr(iteration+1) + '.pth'))

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

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]
             
if __name__ == '__main__':
    train()

