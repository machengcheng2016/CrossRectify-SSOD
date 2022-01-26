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
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


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
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
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
parser.add_argument('--save_interval', default=12000, type=int,
                    help='Interval of save weights')
parser.add_argument('--sup_aug_type', default='default', type=str,
                    help='default | autoaugment | gridmask')
parser.add_argument('--num_gpus', default=4, type=int,
                    help='num_gpus')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed')
args = parser.parse_args()


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

    if args.dataset == 'COCO':
        dataset = COCODetection(root=args.dataset_root,
                                transform=img_transform_sup)
    elif args.dataset == 'VOC300':
        dataset = VOCDetection_con(root=args.dataset_root,
                               img_transform_sup=img_transform_sup, img_transform_unsup=img_transform_sup)
    elif args.dataset == 'VOC512':
        dataset = VOCDetection(root=args.dataset_root,
                               transform=img_transform_sup)
    else:
        raise KeyError("There is no such dataset")


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

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    print('Loading the dataset...')

    print('Training SSD on:', args.dataset)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(cfg['max_iter'] // (args.batch_size // 32)):

        if iteration * (args.batch_size // 32) in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        if args.resume and iteration < int((args.resume).split('_')[-1].split('.')[0]):
            continue

        try:
            images, targets, semis = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets, semis = next(batch_iterator)
    
        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        # forward
        out = net(images)

        sup_image_binary_index = np.zeros([len(semis),1])

        for super_image in range(len(semis)):
            if int(semis[super_image]) == 1 :
                sup_image_binary_index[super_image] = 1
            else:
                sup_image_binary_index[super_image] = 0

            if int(semis[len(semis)-1-super_image]) == 0:
                del targets[len(semis)-1-super_image]

        sup_image_index = np.where(sup_image_binary_index == 1)[0]

        loc_data, conf_data, priors = out

        if len(sup_image_index) > 0:
            loc_data = loc_data[sup_image_index,:,:]
            conf_data = conf_data[sup_image_index,:,:]
            output = (
                loc_data,
                conf_data,
                priors
            )

        # backprop
        loss_l = Variable(torch.cuda.FloatTensor([0]))
        loss_c = Variable(torch.cuda.FloatTensor([0]))
        loss = Variable(torch.cuda.FloatTensor([0]))

        if len(sup_image_index) > 0:
            loss_l, loss_c = criterion(output, targets)
            loss = loss_l + loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        if loss.item() > 1000:
            break

        if iteration % 200 == 0:
            print("iter {} || loss: {:.4f}, loss_c: {:.4f}, loss_l: {:.4f}, super_len: {}".format(iteration, loss.item(), loss_c.item(), loss_l.item(), len(sup_image_index)))

        if (iteration != 0 and (iteration+1) % args.save_interval == 0) or (iteration+1 in range(12000-10, 12000)) or (iteration+1 in range(60000-10, 60000)) or (iteration+1 in range(120000-10, 120000)):
            print('Saving state, iter:', iteration+1)
            if not os.path.exists(os.path.join(args.save_folder, args.sup_aug_type)):
                os.makedirs(os.path.join(args.save_folder, args.sup_aug_type))
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, args.sup_aug_type, "ssd300_"+repr(iteration+1) + '.pth'))


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
