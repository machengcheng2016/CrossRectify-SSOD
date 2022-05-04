# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp

def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input
 
def replicate_input(x):
    return x.detach().clone()
    
def to_one_hot(y, num_classes=21):
    """
    Take a batch of label y with n dims and convert it to
    1-hot representation with n+1 dims.
    Link: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/24
    """
    if not isinstance(y, torch.Tensor):
        y = torch.LongTensor([y])
    y = replicate_input(y).view(-1, 1)
    y_one_hot = y.new_zeros((y.size()[0], num_classes)).scatter_(1, y, 1)
    return y_one_hot

class ISDLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion):

        ### interpolation regularization
        # out, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation
        reduced_batch_size = conf.shape[0]
        conf_temp = conf_shuffle.clone()
        loc_temp = loc_shuffle.clone()
        conf_temp[:int(reduced_batch_size / 2), :, :] = conf_shuffle[int(reduced_batch_size / 2):, :, :]
        conf_temp[int(reduced_batch_size / 2):, :, :] = conf_shuffle[:int(reduced_batch_size / 2), :, :]
        loc_temp[:int(reduced_batch_size / 2), :, :] = loc_shuffle[int(reduced_batch_size / 2):, :, :]
        loc_temp[int(reduced_batch_size / 2):, :, :] = loc_shuffle[:int(reduced_batch_size / 2), :, :]

        ## original background elimination
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data

        ## flip background elimination
        right_conf_class = conf_temp[:, :, 1:].clone()
        right_background_score = conf_temp[:, :, 0].clone()
        right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
        right_mask_val = right_each_val > right_background_score
        right_mask_val = right_mask_val.data

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_right_mask_val = right_mask_val.float() * (1 - left_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        only_right_mask_val = only_right_mask_val.bool()

        intersection_mask_val = left_mask_val * right_mask_val

        ##################    Type-I_######################
        intersection_mask_conf_index = intersection_mask_val.unsqueeze(2).expand_as(conf)

        intersection_left_conf_mask_sample = conf.clone()
        intersection_left_conf_sampled = intersection_left_conf_mask_sample[intersection_mask_conf_index].view(-1,
                                                                                                               21)

        intersection_right_conf_mask_sample = conf_temp.clone()
        intersection_right_conf_sampled = intersection_right_conf_mask_sample[intersection_mask_conf_index].view(-1,
                                                                                                                 21)

        intersection_intersection_conf_mask_sample = conf_interpolation.clone()
        intersection_intersection_sampled = intersection_intersection_conf_mask_sample[
            intersection_mask_conf_index].view(-1, 21)

        if (intersection_mask_val.sum() > 0):

            mixed_val = lam * intersection_left_conf_sampled + (1 - lam) * intersection_right_conf_sampled

            mixed_val = mixed_val + 1e-7
            intersection_intersection_sampled = intersection_intersection_sampled + 1e-7

            interpolation_consistency_conf_loss_a = conf_consistency_criterion(mixed_val.log(),
                                                                               intersection_intersection_sampled.detach()).sum(
                -1).mean()
            interpolation_consistency_conf_loss_b = conf_consistency_criterion(
                intersection_intersection_sampled.log(),
                mixed_val.detach()).sum(-1).mean()
            interpolation_consistency_conf_loss = interpolation_consistency_conf_loss_a + interpolation_consistency_conf_loss_b
            interpolation_consistency_conf_loss = torch.div(interpolation_consistency_conf_loss, 2)
        else:
            interpolation_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            interpolation_consistency_conf_loss = interpolation_consistency_conf_loss.data[0]

        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)
        only_left_mask_loc_index = only_left_mask_val.unsqueeze(2).expand_as(loc)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_loc_mask_sample = loc.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)
        ori_fixmatch_loc_sampled = ori_fixmatch_loc_mask_sample[only_left_mask_loc_index].view(-1, 4)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_loc_mask_sample_interpolation = loc_interpolation.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)
        ori_fixmatch_loc_sampled_interpolation = ori_fixmatch_loc_mask_sample_interpolation[
            only_left_mask_loc_index].view(-1, 4)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

            ## LOC LOSS
            only_left_consistency_loc_loss_x = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 0] - ori_fixmatch_loc_sampled[:, 0].detach(),
                exponent=2))
            only_left_consistency_loc_loss_y = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 1] - ori_fixmatch_loc_sampled[:, 1].detach(),
                exponent=2))
            only_left_consistency_loc_loss_w = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 2] - ori_fixmatch_loc_sampled[:, 2].detach(),
                exponent=2))
            only_left_consistency_loc_loss_h = torch.mean(torch.pow(
                ori_fixmatch_loc_sampled_interpolation[:, 3] - ori_fixmatch_loc_sampled[:, 3].detach(),
                exponent=2))

            only_left_consistency_loc_loss = torch.div(
                only_left_consistency_loc_loss_x + only_left_consistency_loc_loss_y + only_left_consistency_loc_loss_w + only_left_consistency_loc_loss_h,
                4)

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]
            only_left_consistency_loc_loss = only_left_consistency_loc_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss + only_left_consistency_loc_loss




        ##################    Type-II_B ######################

        only_right_mask_conf_index = only_right_mask_val.unsqueeze(2).expand_as(conf)
        only_right_mask_loc_index = only_right_mask_val.unsqueeze(2).expand_as(loc)

        flip_fixmatch_conf_mask_sample = conf_temp.clone()
        flip_fixmatch_loc_mask_sample = loc_temp.clone()
        flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 21)
        flip_fixmatch_loc_sampled = flip_fixmatch_loc_mask_sample[only_right_mask_loc_index].view(-1, 4)

        flip_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        flip_fixmatch_loc_mask_sample_interpolation = loc_interpolation.clone()
        flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
            only_right_mask_conf_index].view(-1, 21)
        flip_fixmatch_loc_sampled_interpolation = flip_fixmatch_loc_mask_sample_interpolation[
            only_right_mask_loc_index].view(-1, 4)

        if (only_right_mask_val.sum() > 0):
            ## KLD !!!!!1
            flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_sampled_interpolation + 1e-7
            flip_fixmatch_conf_sampled = flip_fixmatch_conf_sampled + 1e-7
            only_right_consistency_conf_loss_a = conf_consistency_criterion(
                flip_fixmatch_conf_sampled_interpolation.log(),
                flip_fixmatch_conf_sampled.detach()).sum(-1).mean()
            # consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
            #                                                      conf_sampled.detach()).sum(-1).mean()
            # consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b
            only_right_consistency_conf_loss = only_right_consistency_conf_loss_a

            ## LOC LOSS
            only_right_consistency_loc_loss_x = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 0] - flip_fixmatch_loc_sampled[:, 0].detach(),
                    exponent=2))
            only_right_consistency_loc_loss_y = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 1] - flip_fixmatch_loc_sampled[:, 1].detach(),
                    exponent=2))
            only_right_consistency_loc_loss_w = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 2] - flip_fixmatch_loc_sampled[:, 2].detach(),
                    exponent=2))
            only_right_consistency_loc_loss_h = torch.mean(
                torch.pow(
                    flip_fixmatch_loc_sampled_interpolation[:, 3] - flip_fixmatch_loc_sampled[:, 3].detach(),
                    exponent=2))

            only_right_consistency_loc_loss = torch.div(
                only_right_consistency_loc_loss_x + only_right_consistency_loc_loss_y + only_right_consistency_loc_loss_w + only_right_consistency_loc_loss_h,
                4)

        else:
            only_right_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_conf_loss = only_right_consistency_conf_loss.data[0]
            only_right_consistency_loc_loss = only_right_consistency_loc_loss.data[0]

        # consistency_loss = consistency_conf_loss  # consistency_loc_loss
        only_right_consistency_loss = only_right_consistency_conf_loss + only_right_consistency_loc_loss
        #            only_right_consistency_loss = only_right_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss + only_right_consistency_loss
        return interpolation_consistency_conf_loss, fixmatch_loss
