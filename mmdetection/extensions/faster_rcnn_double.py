# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .two_stage_double import TwoStageDetectorDouble


@DETECTORS.register_module()
class FasterRCNNDouble(TwoStageDetectorDouble):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone1,
                 backbone2,
                 rpn_head1,
                 rpn_head2,
                 roi_head1,
                 roi_head2,
                 train_cfg,
                 test_cfg,
                 neck1=None,
                 neck2=None,
                 pretrained1=None,
                 pretrained2=None,
                 init_cfg=None):
        super(FasterRCNNDouble, self).__init__(
            backbone1=backbone1,
            backbone2=backbone2,
            neck1=neck1,
            neck2=neck2,
            rpn_head1=rpn_head1,
            rpn_head2=rpn_head2,
            roi_head1=roi_head1,
            roi_head2=roi_head2,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained1=pretrained1,
            pretrained2=pretrained2,
            init_cfg=init_cfg)
