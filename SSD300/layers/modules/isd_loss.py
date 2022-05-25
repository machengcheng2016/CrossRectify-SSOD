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



class ISDLoss_select(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_select, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion, sup_image_index):
        sup_image_index_A = sup_image_index
        sup_image_index_B = np.sort((sup_image_index + 16) % 32)
        unsup_image_index_A = np.array(list(set(range(32)) - set(sup_image_index_A.tolist())))
        unsup_image_index_B = np.array(list(set(range(32)) - set(sup_image_index_B.tolist())))
        
        ### interpolation regularization
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
        left_mask_val_select = left_mask_val.clone().detach()
        left_mask_val_select[unsup_image_index_A, :] = False

        ## flip background elimination
        right_conf_class = conf_temp[:, :, 1:].clone()
        right_background_score = conf_temp[:, :, 0].clone()
        right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
        right_mask_val = right_each_val > right_background_score
        right_mask_val = right_mask_val.data
        right_mask_val_select = right_mask_val.clone().detach()
        right_mask_val_select[unsup_image_index_B, :] = False

        ## both background elimination
        only_left_mask_val = left_mask_val_select.float() * (1 - right_mask_val.float())
        only_right_mask_val = right_mask_val_select.float() * (1 - left_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        only_right_mask_val = only_right_mask_val.bool()

        intersection_mask_val = left_mask_val_select * right_mask_val_select

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



class ISDLoss_only_type1(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type1, self).__init__()
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

        return interpolation_consistency_conf_loss



class ISDLoss_only_type2_conf_only_ori(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion):
    
        conf_temp = conf_shuffle.clone()
        conf_temp[:int(reduced_batch_size / 2), :, :] = conf_shuffle[int(reduced_batch_size / 2):, :, :]
        conf_temp[int(reduced_batch_size / 2):, :, :] = conf_shuffle[:int(reduced_batch_size / 2), :, :]

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
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss
        
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_only_ori_select(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_select, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion, sup_image_index):
        sup_image_index_A = sup_image_index
        unsup_image_index_A = np.array(list(set(range(32)) - set(sup_image_index_A.tolist())))
    
        conf_temp = conf_shuffle.clone()
        conf_temp[:int(reduced_batch_size / 2), :, :] = conf_shuffle[int(reduced_batch_size / 2):, :, :]
        conf_temp[int(reduced_batch_size / 2):, :, :] = conf_shuffle[:int(reduced_batch_size / 2), :, :]

        ## original background elimination
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        left_mask_val_select = left_mask_val.clone().detach()
        left_mask_val_select[unsup_image_index_A, :] = False

        ## flip background elimination
        right_conf_class = conf_temp[:, :, 1:].clone()
        right_background_score = conf_temp[:, :, 0].clone()
        right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
        right_mask_val = right_each_val > right_background_score
        right_mask_val = right_mask_val.data

        ## both background elimination        
        only_left_mask_val = left_mask_val_select.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss
        
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_both_ori_and_flip(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_both_ori_and_flip, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion):

        reduced_batch_size = conf.shape[0]
        conf_temp = conf_shuffle.clone()
        conf_temp[:int(reduced_batch_size / 2), :, :] = conf_shuffle[int(reduced_batch_size / 2):, :, :]
        conf_temp[int(reduced_batch_size / 2):, :, :] = conf_shuffle[:int(reduced_batch_size / 2), :, :]

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

        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss

        ##################    Type-II_B ######################

        only_right_mask_conf_index = only_right_mask_val.unsqueeze(2).expand_as(conf)

        flip_fixmatch_conf_mask_sample = conf_temp.clone()
        flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 21)

        flip_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
            only_right_mask_conf_index].view(-1, 21)

        if (only_right_mask_val.sum() > 0):
            ## KLD !!!!!1
            flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_sampled_interpolation + 1e-7
            flip_fixmatch_conf_sampled = flip_fixmatch_conf_sampled + 1e-7
            only_right_consistency_conf_loss_a = conf_consistency_criterion(
                flip_fixmatch_conf_sampled_interpolation.log(),
                flip_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_right_consistency_conf_loss = only_right_consistency_conf_loss_a

        else:
            only_right_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_conf_loss = only_right_consistency_conf_loss.data[0]

        only_right_consistency_loss = only_right_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss + only_right_consistency_loss
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_both_ori_and_flip_select(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_both_ori_and_flip_select, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion, sup_image_index):
        sup_image_index_A = sup_image_index
        sup_image_index_B = np.sort((sup_image_index + 16) % 32)
        unsup_image_index_A = np.array(list(set(range(32)) - set(sup_image_index_A.tolist())))
        unsup_image_index_B = np.array(list(set(range(32)) - set(sup_image_index_B.tolist())))

        reduced_batch_size = conf.shape[0]
        conf_temp = conf_shuffle.clone()
        conf_temp[:int(reduced_batch_size / 2), :, :] = conf_shuffle[int(reduced_batch_size / 2):, :, :]
        conf_temp[int(reduced_batch_size / 2):, :, :] = conf_shuffle[:int(reduced_batch_size / 2), :, :]

        ## original background elimination
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        left_mask_val_select = left_mask_val.clone().detach()
        left_mask_val_select[unsup_image_index_A, :] = False

        ## flip background elimination
        right_conf_class = conf_temp[:, :, 1:].clone()
        right_background_score = conf_temp[:, :, 0].clone()
        right_each_val, right_each_index = torch.max(right_conf_class, dim=2)
        right_mask_val = right_each_val > right_background_score
        right_mask_val = right_mask_val.data
        right_mask_val_select = right_mask_val.clone().detach()
        right_mask_val_select[unsup_image_index_B, :] = False

        ## both background elimination
        only_left_mask_val = left_mask_val_select.float() * (1 - right_mask_val.float())
        only_right_mask_val = right_mask_val_select.float() * (1 - left_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        only_right_mask_val = only_right_mask_val.bool()

        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss

        ##################    Type-II_B ######################

        only_right_mask_conf_index = only_right_mask_val.unsqueeze(2).expand_as(conf)

        flip_fixmatch_conf_mask_sample = conf_temp.clone()
        flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 21)

        flip_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
            only_right_mask_conf_index].view(-1, 21)

        if (only_right_mask_val.sum() > 0):
            ## KLD !!!!!1
            flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_sampled_interpolation + 1e-7
            flip_fixmatch_conf_sampled = flip_fixmatch_conf_sampled + 1e-7
            only_right_consistency_conf_loss_a = conf_consistency_criterion(
                flip_fixmatch_conf_sampled_interpolation.log(),
                flip_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_right_consistency_conf_loss = only_right_consistency_conf_loss_a

        else:
            only_right_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_conf_loss = only_right_consistency_conf_loss.data[0]

        only_right_consistency_loss = only_right_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss + only_right_consistency_loss
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_both_ori_and_flip_with_mix_adv_true(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_both_ori_and_flip_with_mix_adv_true, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, net, img_mix, lam, conf, conf_flip, loc, loc_flip, conf_shuffle, conf_interpolation, loc_shuffle, loc_interpolation, conf_consistency_criterion):

        reduced_batch_size = conf.shape[0]
        conf_temp = conf_shuffle.clone()
        conf_temp[:int(reduced_batch_size / 2), :, :] = conf_shuffle[int(reduced_batch_size / 2):, :, :]
        conf_temp[int(reduced_batch_size / 2):, :, :] = conf_shuffle[:int(reduced_batch_size / 2), :, :]

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

        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a
            
            ## adversarial loss
            if not args.no_random_start:
                img_adv = replicate_input(img_mix) + 0.1 * torch.randn_like(img_mix)
            else:
                img_adv = replicate_input(img_mix)
            for i in range(args.num_iters):
                img_adv.requires_grad_()
                logits_adv = net(img_adv, None, None)[0][1]
                logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)
                with torch.enable_grad():
                    if args.loss_type == "pgd":
                        if args.onehot:
                            if args.targeted:
                                loss_adv = -F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), torch.zeros_like(ori_fixmatch_conf_sampled.detach().argmax(dim=-1)))
                            else:
                                loss_adv = F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach().argmax(dim=-1))
                        else:
                            if args.no_random_start:
                                raise ValueError("no random start but do KL attack")
                            else:
                                loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach())
                    elif args.loss_type == "cw":
                        if args.targeted:
                            y_onehot = torch.zeros_like(logits_adv_sampled)
                            y_onehot[:,0] = 1.0
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                        else:
                            y_onehot = to_one_hot(ori_fixmatch_conf_sampled.detach().argmax(dim=-1))
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                grad = torch.autograd.grad(loss_adv, [img_adv])[0].detach().data
                if args.pert_type == "l_inf":
                    img_adv = replicate_input(img_adv) + args.stepsize * torch.sign(grad)
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                elif args.pert_type == "l_2":
                    grad_norms = grad.view(img_mix.shape[0], -1).norm(p=2, dim=1)
                    grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])
                    grad.renorm_(p=2, dim=0, maxnorm=args.eps)
                    img_adv = replicate_input(img_adv) + args.stepsize * grad
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                else:
                    raise ValueError("No such pert_type {}".format(args.pert_type))
                net.zero_grad()

            logits_adv = net(img_adv, None, None)[0][1]
            logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)
            
            loss_adv_left = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach())
            
        else:
        
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]
            loss_adv_left = Variable(torch.cuda.FloatTensor([0]))
            loss_adv_left = loss_adv_left.data[0]

        only_left_consistency_loss = only_left_consistency_conf_loss

        ##################    Type-II_B ######################

        only_right_mask_conf_index = only_right_mask_val.unsqueeze(2).expand_as(conf)

        flip_fixmatch_conf_mask_sample = conf_temp.clone()
        flip_fixmatch_conf_sampled = flip_fixmatch_conf_mask_sample[only_right_mask_conf_index].view(-1, 21)

        flip_fixmatch_conf_mask_sample_interpolation = conf_interpolation.clone()
        flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_mask_sample_interpolation[
            only_right_mask_conf_index].view(-1, 21)

        if (only_right_mask_val.sum() > 0):
            ## KLD !!!!!1
            flip_fixmatch_conf_sampled_interpolation = flip_fixmatch_conf_sampled_interpolation + 1e-7
            flip_fixmatch_conf_sampled = flip_fixmatch_conf_sampled + 1e-7
            only_right_consistency_conf_loss_a = conf_consistency_criterion(
                flip_fixmatch_conf_sampled_interpolation.log(),
                flip_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_right_consistency_conf_loss = only_right_consistency_conf_loss_a

            ## adversarial loss
            if not args.no_random_start:
                img_adv = replicate_input(img_mix) + 0.1 * torch.randn_like(img_mix)
            else:
                img_adv = replicate_input(img_mix)
            for i in range(args.num_iters):
                img_adv.requires_grad_()
                logits_adv = net(img_adv, None, None)[0][1]
                logits_adv_sampled = logits_adv[only_right_mask_conf_index].view(-1, 21)
                with torch.enable_grad():
                    if args.loss_type == "pgd":
                        if args.onehot:
                            if args.targeted:
                                loss_adv = -F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), torch.zeros_like(flip_fixmatch_conf_sampled.detach().argmax(dim=-1)))
                            else:
                                loss_adv = F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), flip_fixmatch_conf_sampled.detach().argmax(dim=-1))
                        else:
                            if args.no_random_start:
                                raise ValueError("no random start but do KL attack")
                            else:
                                loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), flip_fixmatch_conf_sampled.detach())
                    elif args.loss_type == "cw":
                        if args.targeted:
                            y_onehot = torch.zeros_like(logits_adv_sampled)
                            y_onehot[:,0] = 1.0
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                        else:
                            y_onehot = to_one_hot(flip_fixmatch_conf_sampled.detach().argmax(dim=-1))
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                grad = torch.autograd.grad(loss_adv, [img_adv])[0].detach().data
                if args.pert_type == "l_inf":
                    img_adv = replicate_input(img_adv) + args.stepsize * torch.sign(grad)
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                elif args.pert_type == "l_2":
                    grad_norms = grad.view(img_mix.shape[0], -1).norm(p=2, dim=1)
                    grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])
                    grad.renorm_(p=2, dim=0, maxnorm=args.eps)
                    img_adv = replicate_input(img_adv) + args.stepsize * grad
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                else:
                    raise ValueError("No such pert_type {}".format(args.pert_type))
                net.zero_grad()

            logits_adv = net(img_adv, None, None)[0][1]
            logits_adv_sampled = logits_adv[only_right_mask_conf_index].view(-1, 21)
            
            loss_adv_right = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), flip_fixmatch_conf_sampled.detach())
        else:
        
            only_right_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_right_consistency_conf_loss = only_right_consistency_conf_loss.data[0]
            loss_adv_right = Variable(torch.cuda.FloatTensor([0]))
            loss_adv_right = loss_adv_right.data[0]

        only_right_consistency_loss = only_right_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss + only_right_consistency_loss
        
        loss_adv = loss_adv_left + loss_adv_right
        
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss, loss_adv



class ISDLoss_only_type2_conf_only_ori_no_flip(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data

        ## both background elimination
        only_left_mask_val = left_mask_val.float()
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]


        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss
        
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_only_ori_no_flip_both_mask(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip_both_mask, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        reduced_batch_size = conf.shape[0]
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        
        ## right background elimination
        right_mask_val = left_mask_val.clone()
        right_mask_val[:int(reduced_batch_size / 2), :] = left_mask_val[int(reduced_batch_size / 2):, :]
        right_mask_val[int(reduced_batch_size / 2):, :] = left_mask_val[:int(reduced_batch_size / 2), :]

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]

        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss
        
        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_wrong(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_wrong, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        reduced_batch_size = conf.shape[0]
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data

        ## right background elimination
        right_mask_val = left_mask_val.detach()
        right_mask_val[:int(reduced_batch_size / 2), :] = left_mask_val[int(reduced_batch_size / 2):, :]
        right_mask_val[int(reduced_batch_size / 2):, :] = left_mask_val[:int(reduced_batch_size / 2), :]

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()

        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)
            
        if (only_left_mask_val.sum() > 0):
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a

        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]

        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss

        return Variable(torch.cuda.FloatTensor([0])), fixmatch_loss



class ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_with_mix_adv(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_with_mix_adv, self).__init__()
        self.use_gpu = use_gpu
        

    def forward(self, args, net, img_mix, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        reduced_batch_size = conf.shape[0]
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        
        ## right background elimination
        right_mask_val = left_mask_val.clone()
        right_mask_val[:int(reduced_batch_size / 2), :] = left_mask_val[int(reduced_batch_size / 2):, :]
        right_mask_val[int(reduced_batch_size / 2):, :] = left_mask_val[:int(reduced_batch_size / 2), :]

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
        
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a
            
            ## adversarial loss
            
            if not args.no_random_start:
                img_adv = replicate_input(img_mix) + 0.1 * torch.randn_like(img_mix)
            else:
                img_adv = replicate_input(img_mix)
            for i in range(args.num_iters):
                img_adv.requires_grad_()
                logits_adv = net(img_adv, None)[0][1]
                logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)
                with torch.enable_grad():
                    if args.loss_type == "pgd":
                        if args.onehot:
                            if args.targeted:
                                loss_adv = -F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), torch.zeros_like(ori_fixmatch_conf_sampled_interpolation.detach().argmax(dim=-1)))
                            else:
                                loss_adv = F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled_interpolation.detach().argmax(dim=-1))
                        else:
                            if args.no_random_start:
                                raise ValueError("no random start but do KL attack")
                            else:
                                loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled_interpolation.detach())
                    elif args.loss_type == "cw":
                        if args.targeted:
                            y_onehot = torch.zeros_like(logits_adv_sampled)
                            y_onehot[:,0] = 1.0
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                        else:
                            y_onehot = to_one_hot(ori_fixmatch_conf_sampled_interpolation.detach().argmax(dim=-1))
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                grad = torch.autograd.grad(loss_adv, [img_adv])[0].detach().data
                if args.pert_type == "l_inf":
                    img_adv = replicate_input(img_adv) + args.stepsize * torch.sign(grad)
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                elif args.pert_type == "l_2":
                    grad_norms = grad.view(img_mix.shape[0], -1).norm(p=2, dim=1)
                    grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])
                    grad.renorm_(p=2, dim=0, maxnorm=args.eps)
                    img_adv = replicate_input(img_adv) + args.stepsize * grad
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                else:
                    raise ValueError("No such pert_type {}".format(args.pert_type))
                net.zero_grad()

            logits_adv = net(img_adv, None)[0][1]
            logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)

            loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled_interpolation.detach())
            
        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]
            loss_adv = Variable(torch.cuda.FloatTensor([0]))
            loss_adv = loss_adv.data[0]

        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss
        
        return Variable(torch.cuda.FloatTensor([0])).data[0], fixmatch_loss, loss_adv



class ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_only_mix_adv(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_only_mix_adv, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, net, img_mix, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        reduced_batch_size = conf.shape[0]
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        
        ## right background elimination
        right_mask_val = left_mask_val.clone()
        right_mask_val[:int(reduced_batch_size / 2), :] = left_mask_val[int(reduced_batch_size / 2):, :]
        right_mask_val[int(reduced_batch_size / 2):, :] = left_mask_val[:int(reduced_batch_size / 2), :]

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        loss_adv = Variable(torch.cuda.FloatTensor([0])).data[0]

        if (only_left_mask_val.sum() > 0):
        
            ## adversarial loss
            
            if not args.no_random_start:
                img_adv = replicate_input(img_mix) + 0.1 * torch.randn_like(img_mix)
            else:
                img_adv = replicate_input(img_mix)
            for i in range(args.num_iters):
                img_adv.requires_grad_()
                logits_adv = net(img_adv, None)[0][1]
                logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)
                with torch.enable_grad():
                    if args.loss_type == "pgd":
                        if args.onehot:
                            if args.targeted:
                                loss_adv = -F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), torch.zeros_like(ori_fixmatch_conf_sampled_interpolation.detach().argmax(dim=-1)))
                            else:
                                loss_adv = F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled_interpolation.detach().argmax(dim=-1))
                        else:
                            if args.no_random_start:
                                raise ValueError("no random start but do KL attack")
                            else:
                                loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled_interpolation.detach())
                    elif args.loss_type == "cw":
                        if args.targeted:
                            y_onehot = torch.zeros_like(logits_adv_sampled)
                            y_onehot[:,0] = 1.0
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                        else:
                            y_onehot = to_one_hot(ori_fixmatch_conf_sampled_interpolation.detach().argmax(dim=-1))
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(real - other, min=args.kappa).mean()                            
                grad = torch.autograd.grad(loss_adv, [img_adv])[0].detach().data
                if args.pert_type == "l_inf":
                    img_adv = replicate_input(img_adv) + args.stepsize * torch.sign(grad)
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                elif args.pert_type == "l_2":
                    grad_norms = grad.view(img_mix.shape[0], -1).norm(p=2, dim=1)
                    grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])
                    grad.renorm_(p=2, dim=0, maxnorm=args.eps)
                    img_adv = replicate_input(img_adv) + args.stepsize * grad
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                else:
                    raise ValueError("No such pert_type {}".format(args.pert_type))
                net.zero_grad()

            logits_adv = net(img_adv, None)[0][1]
            logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)

            loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled_interpolation.detach())
            
        return Variable(torch.cuda.FloatTensor([0])).data[0], Variable(torch.cuda.FloatTensor([0])).data[0], loss_adv



class ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_with_mix_adv_true(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_with_mix_adv_true, self).__init__()
        self.use_gpu = use_gpu
        

    def forward(self, args, net, img_mix, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        reduced_batch_size = conf.shape[0]
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        
        ## right background elimination
        right_mask_val = left_mask_val.clone()
        right_mask_val[:int(reduced_batch_size / 2), :] = left_mask_val[int(reduced_batch_size / 2):, :]
        right_mask_val[int(reduced_batch_size / 2):, :] = left_mask_val[:int(reduced_batch_size / 2), :]

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        if (only_left_mask_val.sum() > 0):
        
            ## KLD !!!!!1
            ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_sampled_interpolation + 1e-7
            ori_fixmatch_conf_sampled = ori_fixmatch_conf_sampled + 1e-7
            only_left_consistency_conf_loss_a = conf_consistency_criterion(
                ori_fixmatch_conf_sampled_interpolation.log(),
                ori_fixmatch_conf_sampled.detach()).sum(-1).mean()
            only_left_consistency_conf_loss = only_left_consistency_conf_loss_a
            
            ## adversarial loss
            
            if not args.no_random_start:
                img_adv = replicate_input(img_mix) + 0.1 * torch.randn_like(img_mix)
            else:
                img_adv = replicate_input(img_mix)
            for i in range(args.num_iters):
                img_adv.requires_grad_()
                logits_adv = net(img_adv, None)[0][1]
                logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)
                with torch.enable_grad():
                    if args.loss_type == "pgd":
                        if args.onehot:
                            if args.targeted:
                                loss_adv = -F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), torch.zeros_like(ori_fixmatch_conf_sampled.detach().argmax(dim=-1)))
                            else:
                                loss_adv = F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach().argmax(dim=-1))
                        else:
                            if args.no_random_start:
                                raise ValueError("no random start but do KL attack")
                            else:
                                loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach())
                    elif args.loss_type == "cw":
                        if args.targeted:
                            y_onehot = torch.zeros_like(logits_adv_sampled)
                            y_onehot[:,0] = 1.0
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                        else:
                            y_onehot = to_one_hot(ori_fixmatch_conf_sampled.detach().argmax(dim=-1))
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                grad = torch.autograd.grad(loss_adv, [img_adv])[0].detach().data
                if args.pert_type == "l_inf":
                    img_adv = replicate_input(img_adv) + args.stepsize * torch.sign(grad)
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                elif args.pert_type == "l_2":
                    grad_norms = grad.view(img_mix.shape[0], -1).norm(p=2, dim=1)
                    grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])
                    grad.renorm_(p=2, dim=0, maxnorm=args.eps)
                    img_adv = replicate_input(img_adv) + args.stepsize * grad
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                else:
                    raise ValueError("No such pert_type {}".format(args.pert_type))
                net.zero_grad()

            logits_adv = net(img_adv, None)[0][1]
            logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)

            loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach())
            
        else:
            only_left_consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            only_left_consistency_conf_loss = only_left_consistency_conf_loss.data[0]
            loss_adv = Variable(torch.cuda.FloatTensor([0]))
            loss_adv = loss_adv.data[0]

        only_left_consistency_loss = only_left_consistency_conf_loss

        fixmatch_loss = only_left_consistency_loss
        
        return Variable(torch.cuda.FloatTensor([0])).data[0], fixmatch_loss, loss_adv



class ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_only_mix_adv_true(nn.Module):
    def __init__(self, use_gpu=True):
        super(ISDLoss_only_type2_conf_only_ori_no_flip_both_mask_only_mix_adv_true, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args, net, img_mix, lam, conf, loc, conf_mix, loc_mix, conf_consistency_criterion):

        ## original background elimination
        reduced_batch_size = conf.shape[0]
        left_conf_class = conf[:, :, 1:].clone()
        left_background_score = conf[:, :, 0].clone()
        left_each_val, left_each_index = torch.max(left_conf_class, dim=2)
        left_mask_val = left_each_val > left_background_score
        left_mask_val = left_mask_val.data
        
        ## right background elimination
        right_mask_val = left_mask_val.clone()
        right_mask_val[:int(reduced_batch_size / 2), :] = left_mask_val[int(reduced_batch_size / 2):, :]
        right_mask_val[int(reduced_batch_size / 2):, :] = left_mask_val[:int(reduced_batch_size / 2), :]

        ## both background elimination
        only_left_mask_val = left_mask_val.float() * (1 - right_mask_val.float())
        only_left_mask_val = only_left_mask_val.bool()
        
        ##################    Type-II_A ######################

        only_left_mask_conf_index = only_left_mask_val.unsqueeze(2).expand_as(conf)

        ori_fixmatch_conf_mask_sample = conf.clone()
        ori_fixmatch_conf_sampled = ori_fixmatch_conf_mask_sample[only_left_mask_conf_index].view(-1, 21)

        ori_fixmatch_conf_mask_sample_interpolation = conf_mix.clone()
        ori_fixmatch_conf_sampled_interpolation = ori_fixmatch_conf_mask_sample_interpolation[
            only_left_mask_conf_index].view(-1, 21)

        loss_adv = Variable(torch.cuda.FloatTensor([0])).data[0]

        if (only_left_mask_val.sum() > 0):
        
            ## adversarial loss
            
            if not args.no_random_start:
                img_adv = replicate_input(img_mix) + 0.1 * torch.randn_like(img_mix)
            else:
                img_adv = replicate_input(img_mix)
            for i in range(args.num_iters):
                img_adv.requires_grad_()
                logits_adv = net(img_adv, None)[0][1]
                logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)
                with torch.enable_grad():
                    if args.loss_type == "pgd":
                        if args.onehot:
                            if args.targeted:
                                loss_adv = -F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), torch.zeros_like(ori_fixmatch_conf_sampled.detach().argmax(dim=-1)))
                            else:
                                loss_adv = F.nll_loss(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach().argmax(dim=-1))
                        else:
                            if args.no_random_start:
                                raise ValueError("no random start but do KL attack")
                            else:
                                loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach())
                    elif args.loss_type == "cw":
                        if args.targeted:
                            y_onehot = torch.zeros_like(logits_adv_sampled)
                            y_onehot[:,0] = 1.0
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(other - real, min=args.kappa)
                        else:
                            y_onehot = to_one_hot(ori_fixmatch_conf_sampled.detach().argmax(dim=-1))
                            real = (y_onehot * logits_adv_sampled).sum(dim=1)
                            other = (logits_adv_sampled * (1.0 - y_onehot) - (1e8 * y_onehot)).max(1)[0]         
                            loss_adv = clamp(real - other, min=args.kappa).mean()                            
                grad = torch.autograd.grad(loss_adv, [img_adv])[0].detach().data
                if args.pert_type == "l_inf":
                    img_adv = replicate_input(img_adv) + args.stepsize * torch.sign(grad)
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                elif args.pert_type == "l_2":
                    grad_norms = grad.view(img_mix.shape[0], -1).norm(p=2, dim=1)
                    grad.div_(grad_norms.view(-1, 1, 1, 1))
                    # avoid nan or inf if gradient is 0
                    if (grad_norms == 0).any():
                        grad[grad_norms == 0] = torch.randn_like(grad[grad_norms == 0])
                    grad.renorm_(p=2, dim=0, maxnorm=args.eps)
                    img_adv = replicate_input(img_adv) + args.stepsize * grad
                    img_adv = torch.min(torch.max(img_adv, img_mix - args.eps), img_mix + args.eps)
                else:
                    raise ValueError("No such pert_type {}".format(args.pert_type))
                net.zero_grad()

            logits_adv = net(img_adv, None)[0][1]
            logits_adv_sampled = logits_adv[only_left_mask_conf_index].view(-1, 21)

            loss_adv = F.kl_div(F.log_softmax(logits_adv_sampled, dim=-1), ori_fixmatch_conf_sampled.detach())
            
        return Variable(torch.cuda.FloatTensor([0])).data[0], Variable(torch.cuda.FloatTensor([0])).data[0], loss_adv


