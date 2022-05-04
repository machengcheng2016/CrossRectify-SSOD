# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector


@DETECTORS.register_module()
class TwoStageDetectorDouble(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone1,
                 backbone2,
                 neck1=None,
                 neck2=None,
                 rpn_head1=None,
                 rpn_head2=None,
                 roi_head1=None,
                 roi_head2=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained1=None,
                 pretrained2=None,
                 init_cfg=None):
        super(TwoStageDetectorDouble, self).__init__(init_cfg)
        if pretrained1:
            warnings.warn('DeprecationWarning: pretrained1 is deprecated, '
                          'please use "init_cfg" instead')
            backbone1.pretrained1 = pretrained1
        if pretrained2:
            warnings.warn('DeprecationWarning: pretrained2 is deprecated, '
                          'please use "init_cfg" instead')
            backbone2.pretrained2 = pretrained2
            
        self.backbone1 = build_backbone(backbone1)
        self.backbone2 = build_backbone(backbone2)

        if neck1 is not None:
            self.neck1 = build_neck(neck1)
        if neck2 is not None:
            self.neck2 = build_neck(neck2)

        if rpn_head1 is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head1.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head1 = build_head(rpn_head_)
        if rpn_head2 is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head2.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head2 = build_head(rpn_head_)

        if roi_head1 is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head1.update(train_cfg=rcnn_train_cfg)
            roi_head1.update(test_cfg=test_cfg.rcnn)
            roi_head1.pretrained = pretrained1
            self.roi_head1 = build_head(roi_head1)
        if roi_head2 is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head2.update(train_cfg=rcnn_train_cfg)
            roi_head2.update(test_cfg=test_cfg.rcnn)
            roi_head2.pretrained = pretrained2
            self.roi_head2 = build_head(roi_head2)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck1') and hasattr(self, 'neck2') and self.neck1 is not None and self.neck2 is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head1') and hasattr(self, 'roi_head2') and self.roi_head1.with_shared_head and self.roi_head2.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head1') and self.roi_head1.with_bbox) or (hasattr(self, 'bbox_head1') and self.bbox_head1 is not None)) and ((hasattr(self, 'roi_head2') and self.roi_head2.with_bbox) or (hasattr(self, 'bbox_head2') and self.bbox_head2 is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head1') and self.roi_head1.with_mask) or (hasattr(self, 'mask_head1') and self.mask_head1 is not None)) and ((hasattr(self, 'roi_head2') and self.roi_head2.with_mask) or (hasattr(self, 'mask_head2') and self.mask_head2 is not None))

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head1') and self.rpn_head1 is not None and hasattr(self, 'rpn_head2') and self.rpn_head2

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head1') and self.roi_head1 is not None and hasattr(self, 'roi_head2') and self.roi_head2 is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x1 = self.backbone1(img)
        x2 = self.backbone2(img)
        if self.with_neck:
            x1 = self.neck1(x1)
            x2 = self.neck2(x2)
        return x1, x2

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs1, outs2 = (), ()
        # backbone
        x1, x2 = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs1 = self.rpn_head1(x1)
            rpn_outs2 = self.rpn_head2(x2)
            outs1 = outs1 + (rpn_outs1, )
            outs2 = outs2 + (rpn_outs2, )
        proposals1 = torch.randn(1000, 4).to(img.device)
        proposals2 = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs1 = self.roi_head1.forward_dummy(x1, proposals1)
        roi_outs2 = self.roi_head2.forward_dummy(x2, proposals2)
        outs1 = outs1 + (roi_outs1, )
        outs2 = outs2 + (roi_outs2, )
        return outs1, outs2

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x1, x2 = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses1, proposal_list1 = self.rpn_head1.forward_train(
                x1,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            rpn_losses2, proposal_list2 = self.rpn_head2.forward_train(
                x2,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            for key, value in rpn_losses1.items():
                if not key.endswith('1'):
                    losses.update({key+'1':value})
                else:
                    print(key)
                    losses.update({key:value})
            for key, value in rpn_losses2.items():
                if not key.endswith('2'):
                    losses.update({key+'2':value})
                else:
                    print(key)
                    losses.update({key:value})
        else:
            proposal_list1 = proposal_list2 = proposals

        roi_losses1 = self.roi_head1.forward_train(x1, img_metas, proposal_list1,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        roi_losses2 = self.roi_head2.forward_train(x2, img_metas, proposal_list2,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        for key, value in roi_losses1.items():
            losses.update({key+'1':value})
        for key, value in roi_losses2.items():
            losses.update({key+'2':value})

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x1, x2 = self.extract_feat(img)

        if proposals is None:
            proposal_list1 = await self.rpn_head1.async_simple_test_rpn(
                x1, img_meta)
            proposal_list2 = await self.rpn_head2.async_simple_test_rpn(
                x2, img_meta)
        else:
            proposal_list1 = proposal_list2 = proposals

        return await self.roi_head1.async_simple_test(x1, proposal_list1, img_meta, rescale=rescale), self.roi_head2.async_simple_test(x2, proposal_list2, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x1, x2 = self.extract_feat(img)
        if proposals is None:
            proposal_list1 = self.rpn_head1.simple_test_rpn(x1, img_metas)
            proposal_list2 = self.rpn_head2.simple_test_rpn(x2, img_metas)
        else:
            proposal_list1 = proposal_list2 = proposals

        return self.roi_head1.simple_test(x1, proposal_list1, img_metas, rescale=rescale), self.roi_head2.simple_test(x2, proposal_list2, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x1, x2 = self.extract_feats(imgs)
        proposal_list1 = self.rpn_head1.aug_test_rpn(x1, img_metas)
        proposal_list2 = self.rpn_head2.aug_test_rpn(x2, img_metas)
        return self.roi_head1.aug_test(x1, proposal_list1, img_metas, rescale=rescale), self.roi_head2.aug_test(x2, proposal_list2, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x1, x2 = self.extract_feat(img)
        proposals1 = self.rpn_head1.onnx_export(x1, img_metas)
        proposals2 = self.rpn_head2.onnx_export(x2, img_metas)
        if hasattr(self.roi_head1, 'onnx_export'):
            return self.roi_head1.onnx_export(x1, proposals1, img_metas)
        if hasattr(self.roi_head2, 'onnx_export'):
            return self.roi_head2.onnx_export(x2, proposals2, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
