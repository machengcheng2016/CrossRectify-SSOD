import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc300, voc512, coco
import os
import warnings
import math
import numpy as np
import cv2
import copy


class SSD_CON(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_CON, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if(size==300):
            self.cfg = (coco, voc300)[num_classes == 21]
        else:
            self.cfg = (coco, voc512)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if phase == 'test':
            # self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.vgg_t = copy.deepcopy(self.vgg)
            self.extras_t = copy.deepcopy(self.extras)
            self.loc_t = copy.deepcopy(self.loc)
            self.conf_t = copy.deepcopy(self.conf)

        ### mt
        self.ema_factor = 0.999
        self.global_step = 0


    def forward(self, x, x_flip, x_shuffle):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """



        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # zero_mask = torch.cat([o.view(o.size(0), -1) for o in zero_mask], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        loc = loc.view(loc.size(0), -1, 4)
        conf = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
        # basic

        if (x_flip is None) and (x_shuffle is None):
            return output, conf, None, loc, None, None, None, None, None

        sources_flip = list()
        loc_flip = list()
        conf_flip = list()
        loc_shuffle = list()
        conf_shuffle = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x_flip = self.vgg[k](x_flip)

        s_flip = self.L2Norm(x_flip)
        sources_flip.append(s_flip)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x_flip = self.vgg[k](x_flip)
        sources_flip.append(x_flip)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_flip = F.relu(v(x_flip), inplace=True)
            if k % 2 == 1:
                sources_flip.append(x_flip)

        # apply multibox head to source layers
        for (x_flip, l, c) in zip(sources_flip, self.loc, self.conf):
            loc_shuffle.append(l(x_flip).permute(0, 2, 3, 1).contiguous())
            conf_shuffle.append(c(x_flip).permute(0, 2, 3, 1).contiguous())
            append_loc = l(x_flip).permute(0, 2, 3, 1).contiguous()
            append_conf = c(x_flip).permute(0, 2, 3, 1).contiguous()
            append_loc = flip(append_loc,2)
            append_conf = flip(append_conf,2)
            loc_flip.append(append_loc)
            conf_flip.append(append_conf)

        loc_shuffle = torch.cat([o.view(o.size(0), -1) for o in loc_shuffle], 1)
        conf_shuffle = torch.cat([o.view(o.size(0), -1) for o in conf_shuffle], 1)

        loc_flip = torch.cat([o.view(o.size(0), -1) for o in loc_flip], 1)
        conf_flip = torch.cat([o.view(o.size(0), -1) for o in conf_flip], 1)

        loc_shuffle = loc_flip.view(loc_shuffle.size(0), -1, 4)
        conf_shuffle = self.softmax(conf_shuffle.view(conf_shuffle.size(0), -1, self.num_classes))

        loc_flip = loc_flip.view(loc_flip.size(0), -1, 4)
        conf_flip = self.softmax(conf_flip.view(conf_flip.size(0), -1, self.num_classes))



        sources_interpolation = list()
        loc_interpolation = list()
        conf_interpolation = list()

#        # apply vgg up to conv4_3 relu
        for k in range(23):
            x_shuffle = self.vgg[k](x_shuffle)

        s_shuffle = self.L2Norm(x_shuffle)
        sources_interpolation.append(s_shuffle)


#        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x_shuffle = self.vgg[k](x_shuffle)
        sources_interpolation.append(x_shuffle)



#        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_shuffle = F.relu(v(x_shuffle), inplace=True)
            if k % 2 == 1:
                sources_interpolation.append(x_shuffle)



#        # apply multibox head to source layers
        for (x_shuffle, l, c) in zip(sources_interpolation, self.loc, self.conf):
            loc_interpolation.append(l(x_shuffle).permute(0, 2, 3, 1).contiguous())
            conf_interpolation.append(c(x_shuffle).permute(0, 2, 3, 1).contiguous())


        loc_interpolation = torch.cat([o.view(o.size(0), -1) for o in loc_interpolation], 1)
        conf_interpolation = torch.cat([o.view(o.size(0), -1) for o in conf_interpolation], 1)

        loc_interpolation = loc_interpolation.view(loc_interpolation.size(0), -1, 4)

        conf_interpolation = self.softmax(conf_interpolation.view(conf_interpolation.size(0), -1, self.num_classes))


        if self.phase == "test":
            return output
        else:
            return output, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



class SSD_CON_SIMPLE(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD_CON_SIMPLE, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        if(size==300):
            self.cfg = (coco, voc300)[num_classes == 21]
        else:
            self.cfg = (coco, voc512)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.softmax = nn.Softmax(dim=-1)

        if phase == 'test':
            # self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)


    def forward(self, x, x_mix):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # zero_mask = torch.cat([o.view(o.size(0), -1) for o in zero_mask], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        loc = loc.view(loc.size(0), -1, 4)
        conf = self.softmax(conf.view(conf.size(0), -1, self.num_classes))
        # basic

        if x_mix is None:
            if self.phase == "test":
                return output
            else:
                return output, conf, loc, None, None
            
        sources_mix = list()
        loc_mix = list()
        conf_mix = list()

#        # apply vgg up to conv4_3 relu
        for k in range(23):
            x_mix = self.vgg[k](x_mix)

        s_mix = self.L2Norm(x_mix)
        sources_mix.append(s_mix)


#        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x_mix = self.vgg[k](x_mix)
        sources_mix.append(x_mix)



#        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x_mix = F.relu(v(x_mix), inplace=True)
            if k % 2 == 1:
                sources_mix.append(x_mix)



#        # apply multibox head to source layers
        for (x_mix, l, c) in zip(sources_mix, self.loc, self.conf):
            loc_mix.append(l(x_mix).permute(0, 2, 3, 1).contiguous())
            conf_mix.append(c(x_mix).permute(0, 2, 3, 1).contiguous())


        loc_mix = torch.cat([o.view(o.size(0), -1) for o in loc_mix], 1)
        conf_mix = torch.cat([o.view(o.size(0), -1) for o in conf_mix], 1)

        loc_mix = loc_mix.view(loc_mix.size(0), -1, 4)

        conf_mix = self.softmax(conf_mix.view(conf_mix.size(0), -1, self.num_classes))


        if self.phase == "test":
            return output
        else:
            return output, conf, loc, conf_mix, loc_mix

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v=='K':
                layers += [nn.Conv2d(in_channels, 256,
                           kernel_size=4, stride=1, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers



def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K'],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 6, 4, 4],
}

def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
             else torch.arange(x.size(i)-1, -1, -1).long()
             for i in range(x.dim()))]

class GaussianNoise(nn.Module):
    def __init__(self, batch_size, input_size=(3, 300, 300), mean=0, std=0.15):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size, ) + input_size
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.mean = mean
        self.std = std

    def forward(self, x):
        self.noise.data.normal_(self.mean, std=self.std)
        if x.size(0) == self.noise.size(0):
            return x + self.noise
        else:
            #print('---- Noise Size ')
            return x + self.noise[:x.size(0)]


def build_ssd_con(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD_CON(phase, size, base_, extras_, head_, num_classes)



def build_ssd_con_simple(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    # if size != 300:
    #     print("ERROR: You specified size " + repr(size) + ". However, " +
    #           "currently only SSD300 (size=300) is supported!")
    #     return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD_CON_SIMPLE(phase, size, base_, extras_, head_, num_classes)


def compute_mixup_map_i(lam, net, images_left, images_right, conf_consistency_criterion, lr, num_steps_i):
    conf_left = net(images_left, None, None)[1]
    conf_right = net(images_right, None, None)[1]
    
    ## original background elimination
    left_conf_class = conf_left[:, :, 1:].clone()
    left_background_score = conf_left[:, :, 0].clone()
    left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
    left_mask_val = left_each_val > left_background_score
    left_mask_val = left_mask_val.data

    ## flip background elimination
    right_conf_class = conf_right[:, :, 1:].clone()
    right_background_score = conf_right[:, :, 0].clone()
    right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
    right_mask_val = right_each_val > right_background_score
    right_mask_val = right_mask_val.data

    '''
    ## both background elimination
    only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
    only_right_mask_val = right_mask_val.float() * (1 - left_mask_val.float())
    only_left_mask_val = only_left_mask_val.bool()
    only_right_mask_val = only_right_mask_val.bool()
    '''
    
    intersection_mask_val = left_mask_val * right_mask_val
    
    ### Type-I mask
    intersection_mask_conf_index = intersection_mask_val.unsqueeze(2).expand_as(conf_left)
    conf_left_i = conf_left[intersection_mask_conf_index].view(-1, 21)
    conf_right_i = conf_right[intersection_mask_conf_index].view(-1, 21)
    mix_conf_i = lam * conf_left_i + (1 - lam) * conf_right_i

    images_mix_ = lam * images_left.clone() + (1 - lam) * images_right.clone()
    conf_mix_ = net(images_mix_, None, None)[1]
    conf_mix_i_ = conf_mix_[intersection_mask_conf_index].view(-1, 21)
    loss = conf_consistency_criterion(conf_mix_i_.log(), mix_conf_i.detach()).sum(-1).mean()
    print(loss.item())

    mask_i = Variable(torch.ones_like(images_left) * lam, requires_grad=True)
    optimizer_mask_i = optim.Adam([mask_i], lr=lr, betas=(0.0, 0.0))
    for s in range(num_steps_i):
        images_mix = mask_i * images_left + (1 - mask_i) * images_right
        conf_mix = net(images_mix, None, None)[1]
        conf_mix_i = conf_mix[intersection_mask_conf_index].view(-1, 21)
        with torch.enable_grad():
            loss_i = conf_consistency_criterion(conf_mix_i.log(), mix_conf_i.detach()).sum(-1).mean()
        loss_i.backward(retain_graph=True if s != num_steps_i - 1 else False)
        optimizer_mask_i.step()
        print(mask_i.mean())
        
        mask_i.data = torch.sigmoid(mask_i.data)
        sum_masks = mask_i.data.sum(0, keepdim=True)
        mask_i.data /= sum_masks
        print(s, loss_i.item())

    return mask_i, loss_i.item(), loss.item()
    
def compute_mixup_map_ii(lam, net, images_left, images_right, conf_consistency_criterion, lr, num_steps_ii):
    conf_left = net(images_left, None, None)[1]
    conf_right = net(images_right, None, None)[1]
    
    ## original background elimination
    left_conf_class = conf_left[:, :, 1:].clone()
    left_background_score = conf_left[:, :, 0].clone()
    left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
    left_mask_val = left_each_val > left_background_score
    left_mask_val = left_mask_val.data

    ## flip background elimination
    right_conf_class = conf_right[:, :, 1:].clone()
    right_background_score = conf_right[:, :, 0].clone()
    right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
    right_mask_val = right_each_val > right_background_score
    right_mask_val = right_mask_val.data

    ## both background elimination
    only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
    only_left_mask_val = only_left_mask_val.bool()
    
    ### Type-II mask
    only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf_left)
    
    mix_conf_i = lam * conf_left_i + (1 - lam) * conf_right_i
    
    mask_i = Variable(torch.ones_like(images_left) * lam, requires_grad=True)
    optimizer_mask_i = optim.Adam([mask_i], lr=lr, betas=(0.0, 0.0))
    for s in range(num_steps_i):
        images_mix = mask_i * images_left + (1 - mask_i) * images_right
        conf_mix = net(images_mix, None, None)[1]
        conf_mix_i = conf_mix[only_left_mask_conf_index].view(-1, 21)
        with torch.enable_grad():
            loss_i = conf_consistency_criterion(conf_mix_i.log(), mix_conf_i.detach()).sum(-1).mean()
        loss.backward(retain_graph=True if s != num_steps_i - 1 else False)
        optimizer_mask.step()
    return mask_i
