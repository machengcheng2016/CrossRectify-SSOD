import warnings
warnings.filterwarnings("ignore")

from data import *
from utils.augmentations import *
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import math
import copy


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
parser.add_argument('--resume', default=None, type=str,  # None  'weights/ssd300_COCO_80000.pth'
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--beta_dis', default=100.0, type=float,
                    help='beta distribution')
parser.add_argument('--lam', default=-1.0, type=float,
                    help='beta distribution')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--type1coef', default=0.1, type=float,
                    help='type1coef')
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
parser.add_argument('--debug', action='store_true')
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

def shuffle_tensor(tensor):
    tensor_tmp, tensor_shuffle = tensor.detach().clone(), tensor.detach().clone()
    tensor_shuffle[:int(args.batch_size/2)] = tensor_tmp[int(args.batch_size/2):]
    tensor_shuffle[int(args.batch_size/2):] = tensor_tmp[:int(args.batch_size/2)]
    return tensor_shuffle

def adjust_score(scores):
    scores_copy = scores.copy()
    scores_copy[:,1:] = np.sort(scores[:,1:], axis=1)[:,::-1]
    return scores_copy

def print_grad(net):
    for name, param in net.named_parameters():
        if param.grad is not None:
            print(name, param.grad)
        else:
            print(name, None)

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

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    setup_seed(args.seed)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)
                             
    conf_consistency_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    net.train()
    print('Loading the dataset...')

    step_index = 0

    supervised_batch = args.batch_size

    if args.warmup and args.start_iter == 0:
        supervised_dataset = VOCDetection_con_init(root=args.dataset_root,
                                                   transform=img_transform_sup)
    else:
        supervised_dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                          transform=img_transform_sup)

    supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                             num_workers=args.num_workers,
                                             shuffle=True, collate_fn=detection_collate,
                                             pin_memory=True, drop_last=True)


    batch_iterator = iter(supervised_data_loader)

    for iteration in range(cfg['max_iter'] // (args.batch_size // 32)):

        if iteration * (args.batch_size // 32) in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        if args.resume and iteration < int((args.resume).split('_')[-1].split('.')[0]):
            #print("Skipping iteration {}".format(iteration))
            continue

        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            supervised_dataset = VOCDetection(root=args.dataset_root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                              transform=img_transform_sup)
            supervised_data_loader = data.DataLoader(supervised_dataset, supervised_batch,
                                                     num_workers=args.num_workers,
                                                     shuffle=True, collate_fn=detection_collate,
                                                     pin_memory=True, drop_last=True)
            batch_iterator = iter(supervised_data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = images.cuda()
            with torch.no_grad():
                targets = [ann.cuda() for ann in targets]
        else:
            with torch.no_grad():
                targets = [ann for ann in targets]

        # forward
        t0 = time.time()

        output = net(images)
        loc, conf, priors = output

        # backprop
        # loss = Variable(torch.cuda.FloatTensor([0]))
        loss_l = Variable(torch.cuda.FloatTensor([0]))
        loss_c = Variable(torch.cuda.FloatTensor([0]))

        loss_l, loss_c = criterion(output, targets)
        
        images_shuffle = shuffle_tensor(images)
        images_shuffle = flip(images_shuffle, 3)
        
        output_shuffle = net(images_shuffle)
        loc_shuffle, conf_shuffle, priors = output_shuffle
        
        lam = np.random.beta(args.beta_dis, args.beta_dis)
        if args.lam >= 0.0:
            lam = args.lam
        images_mix = lam * images.clone() + (1 - lam) * images_shuffle.clone()
        
        loss_c_unsup1 = torch.cuda.FloatTensor([0])
        loss_l_unsup = torch.cuda.FloatTensor([0])
        loss_c_unsup = torch.cuda.FloatTensor([0])
        if not args.cuda:
            loss_c_unsup1 = loss_c_unsup1.cpu()
            loss_l_unsup = loss_l_unsup.cpu()
            loss_c_unsup = loss_c_unsup.cpu()
                       
        score = F.softmax(conf, dim=-1)
        conf_class = score[:,:,1:]
        background_score = score[:, :, 0]
        each_val, each_index = torch.max(conf_class, dim=2)
        mask_val = (each_val > background_score).data
        
        score_shuffle = F.softmax(conf_shuffle, dim=-1)
        conf_class_shuffle = score_shuffle[:,:,1:]
        background_score_shuffle = score_shuffle[:, :, 0]
        each_val_shuffle, each_index_shuffle = torch.max(conf_class_shuffle, dim=2)
        mask_val_shuffle = (each_val_shuffle > background_score_shuffle).data
        
        mask_left_right = mask_val.float() * mask_val_shuffle.float()
        mask_left_right = mask_left_right.bool()     
        
        mask_only_left = mask_val.float() * (1 - mask_val_shuffle.float())
        mask_only_left = mask_only_left.bool()
        
        if mask_left_right.sum() > 0 or mask_only_left.sum() > 0:
            out_mix = net(images_mix)
            loc_mix, conf_mix, _ = out_mix
        
        if mask_left_right.sum() > 0:
            mask_left_right_conf_index = mask_left_right.unsqueeze(2).expand_as(conf)
            score_sampled = score[mask_left_right_conf_index].view(-1, 21)
            score_shuffle_sampled = score_shuffle[mask_left_right_conf_index].view(-1, 21)
            conf_mix_sampled = conf_mix[mask_left_right_conf_index].view(-1, 21)
            score_mix_sampled = lam * score_sampled + (1 - lam) * score_shuffle_sampled + 1e-7
            loss_c_unsup1a = conf_consistency_criterion(score_mix_sampled.log(), F.softmax(conf_mix_sampled.detach(), dim=-1) + 1e-7).sum(-1).mean()
            loss_c_unsup1b = conf_consistency_criterion((F.softmax(conf_mix_sampled, dim=-1) + 1e-7).log(), score_mix_sampled.detach()).sum(-1).mean()
            loss_c_unsup1 = torch.div(loss_c_unsup1a + loss_c_unsup1b, 2)
        
        if mask_only_left.sum() > 0:
            mask_only_left_loc_index = mask_only_left.unsqueeze(2).expand_as(loc)
            mask_only_left_conf_index = mask_only_left.unsqueeze(2).expand_as(conf)
            loc_sampled = loc[mask_only_left_loc_index].view(-1, 4)
            loc_mix_sampled = loc_mix[mask_only_left_loc_index].view(-1, 4)
            score_sampled = score[mask_only_left_conf_index].view(-1, 21)
            conf_mix_sampled = conf_mix[mask_only_left_conf_index].view(-1, 21)
            loss_l_unsup = F.mse_loss(loc_mix_sampled, loc_sampled.detach())
            loss_c_unsup = conf_consistency_criterion(F.log_softmax(conf_mix_sampled), score_sampled.detach()).sum(-1).mean()

            if args.debug:
                conf_mix_sampled_np = F.softmax(conf_mix_sampled, dim=-1).detach().cpu().numpy()
                score_sampled_np = score_sampled.detach().cpu().numpy()
                conf_mix_sampled_np = adjust_score(conf_mix_sampled_np)
                score_sampled_np = adjust_score(score_sampled_np)
                try:
                    conf_mix_sampled_np_collector.append(conf_mix_sampled_np)
                    score_sampled_np_collector.append(score_sampled_np)
                except:
                    conf_mix_sampled_np_collector = [conf_mix_sampled_np]
                    score_sampled_np_collector = [score_sampled_np]
                #if np.concatenate(conf_mix_sampled_np_collector).shape[0] >= 1000:
                if len(conf_mix_sampled_np_collector) == 20:
                    conf_mix_sampled_np_collector = np.concatenate(conf_mix_sampled_np_collector)#[:1000]
                    score_sampled_np_collector = np.concatenate(score_sampled_np_collector)#[:1000]
                    from scipy.io import savemat
                    savemat("lambda/confirm-{}-{}.mat".format(args.lam, int((args.resume).split('_')[-1].split('.')[0])), {"conf_mix_sampled_np_collector":conf_mix_sampled_np_collector, "score_sampled_np_collector":score_sampled_np_collector})
                    assert False
                    
        consistency_loss = loss_l_unsup + loss_c_unsup
        ramp_weight = rampweight(iteration)
        consistency_loss = torch.mul(consistency_loss, ramp_weight)

        loss = loss_l + loss_c + consistency_loss + loss_c_unsup1 * args.type1coef * ramp_weight

        if float(loss.item()) > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t1 = time.time()

        if args.debug or (iteration % 200 == 0):
            print('timer: %.4f sec.' % (t1 - t0))
            print("iter {} || loss: {:.4f}, loss_c: {:.4f}, loss_l: {:.4f}, loss_c_unsup1: {:.4f}, loss_c_unsup: {:.4f}, loss_l_unsup: {:.4f}, lr: {:.4f}, mask_only_left.sum() = {}, lam = {}\n".format(iteration, loss.item(), loss_c.item(), loss_l.item(), loss_c_unsup1.item(), loss_c_unsup.item(), loss_l_unsup.item(), float(optimizer.param_groups[0]['lr']), mask_only_left.sum().item(), lam))


        if float(loss.item()) > 1000:
            break

        if iteration != 0 and (iteration+1) % args.save_interval == 0:
            print('Saving state, iter:', iteration)
            if not os.path.exists(os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type))):
                os.makedirs(os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type)))
            if args.lam < 0:
                torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), 'ssd300_isd_sup_0712_' + repr(iteration+1) + '.pth'))
            else:
                torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, "{}+{}".format(args.sup_aug_type, args.unsup_aug_type), "ssd300_isd_{}_sup_0712_".format(args.lam) + repr(iteration+1) + '.pth'))
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

