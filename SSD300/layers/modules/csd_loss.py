# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class CSDLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(CSDLoss, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, args,conf, conf_flip, loc, loc_flip, conf_consistency_criterion):
        conf_class = conf[:, :, 1:].clone()
        background_score = conf[:, :, 0].clone()
        each_val, each_index = torch.max(conf_class, dim=2)
        mask_val = each_val > background_score
        mask_val = mask_val.data

        mask_conf_index = mask_val.unsqueeze(2).expand_as(conf)
        mask_loc_index = mask_val.unsqueeze(2).expand_as(loc)


        conf_sampled = conf[mask_conf_index].view(-1, 21).clone()
        loc_sampled = loc[mask_loc_index].view(-1, 4).clone()


        conf_sampled_flip = conf_flip[mask_conf_index].view(-1, 21).clone()
        loc_sampled_flip = loc_flip[mask_loc_index].view(-1, 4).clone()

        if (mask_val.sum() > 0):
            ## JSD !!!!!1
            conf_sampled_flip = conf_sampled_flip + 1e-7
            conf_sampled = conf_sampled + 1e-7
            consistency_conf_loss_a = conf_consistency_criterion(conf_sampled.log(),
                                                                 conf_sampled_flip.detach()).sum(-1).mean()
            consistency_conf_loss_b = conf_consistency_criterion(conf_sampled_flip.log(),
                                                                 conf_sampled.detach()).sum(-1).mean()
            consistency_conf_loss = consistency_conf_loss_a + consistency_conf_loss_b
            consistency_conf_loss = torch.div(consistency_conf_loss, 2)

            ## LOC LOSS
            consistency_loc_loss_x = torch.mean(torch.pow(loc_sampled[:, 0] + loc_sampled_flip[:, 0], exponent=2))
            consistency_loc_loss_y = torch.mean(torch.pow(loc_sampled[:, 1] - loc_sampled_flip[:, 1], exponent=2))
            consistency_loc_loss_w = torch.mean(torch.pow(loc_sampled[:, 2] - loc_sampled_flip[:, 2], exponent=2))
            consistency_loc_loss_h = torch.mean(torch.pow(loc_sampled[:, 3] - loc_sampled_flip[:, 3], exponent=2))

            consistency_loc_loss = torch.div(
                consistency_loc_loss_x + consistency_loc_loss_y + consistency_loc_loss_w + consistency_loc_loss_h,
                4)

        else:
            consistency_conf_loss = Variable(torch.cuda.FloatTensor([0]))
            consistency_loc_loss = Variable(torch.cuda.FloatTensor([0]))
            consistency_conf_loss = consistency_conf_loss.data[0]
            consistency_loc_loss = consistency_loc_loss.data[0]

        consistency_loss = consistency_conf_loss + consistency_loc_loss

        return consistency_loss