# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN

from detectron2.config import configurable

from copy import deepcopy

@META_ARCH_REGISTRY.register()
class DoubleGeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone1 = backbone
        self.backbone2 = deepcopy(backbone)
        self.proposal_generator1 = proposal_generator
        self.proposal_generator2 = deepcopy(proposal_generator)
        self.roi_heads1 = roi_heads
        self.roi_heads2 = deepcopy(roi_heads)

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features1 = self.backbone1(images.tensor)
        features2 = self.backbone2(images.tensor)

        if self.proposal_generator1 is not None:
            proposals1, proposal_losses1 = self.proposal_generator1(images, features1, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals1 = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses1 = {}

        if self.proposal_generator2 is not None:
            proposals2, proposal_losses2 = self.proposal_generator2(images, features2, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals2 = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses2 = {}

        proposal_losses1_prefix = {}
        for key in proposal_losses1.keys():
            proposal_losses1_prefix[key + "1"] = proposal_losses1[key]

        proposal_losses2_prefix = {}
        for key in proposal_losses2.keys():
            proposal_losses2_prefix[key + "2"] = proposal_losses2[key]

        _, detector_losses1 = self.roi_heads1(images, features1, proposals1, gt_instances)
        _, detector_losses2 = self.roi_heads2(images, features2, proposals2, gt_instances)
        
        detector_losses1_prefix = {}
        for key in detector_losses1.keys():
            detector_losses1_prefix[key + "1"] = detector_losses1[key]
        
        detector_losses2_prefix = {}
        for key in detector_losses2.keys():
            detector_losses2_prefix[key + "2"] = detector_losses2[key]
        
        losses = {}
        losses.update(proposal_losses1_prefix)
        losses.update(detector_losses1_prefix)
        losses.update(proposal_losses2_prefix)
        losses.update(detector_losses2_prefix)
        
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features1 = self.backbone1(images.tensor)
        features2 = self.backbone2(images.tensor)

        if detected_instances is None:
            if self.proposal_generator1 is not None:
                proposals1, _ = self.proposal_generator1(images, features1, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals1 = [x["proposals"].to(self.device) for x in batched_inputs]
            results1, _ = self.roi_heads1(images, features1, proposals1, None)
            
            if self.proposal_generator2 is not None:
                proposals2, _ = self.proposal_generator2(images, features2, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals2 = [x["proposals"].to(self.device) for x in batched_inputs]
            results2, _ = self.roi_heads2(images, features2, proposals2, None)
            
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results1 = self.roi_heads1.forward_with_given_boxes(features1, detected_instances)
            results2 = self.roi_heads2.forward_with_given_boxes(features2, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results1, batched_inputs, images.image_sizes), GeneralizedRCNN._postprocess(results2, batched_inputs, images.image_sizes)
        else:
            return results1, results2

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone1.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabDoubleGeneralizedRCNN(DoubleGeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, do_A=True, do_B=True
    ):
        assert do_A or do_B
        
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if do_A:
            features1 = self.backbone1(images.tensor)
        if do_B:
            features2 = self.backbone2(images.tensor)

        if branch == "supervised":
            # Region proposal network
            if do_A:
                proposals_rpn1, proposal_losses1 = self.proposal_generator1(
                    images, features1, gt_instances
                )
                proposal_losses1_prefix = {}
                for key in proposal_losses1.keys():
                    proposal_losses1_prefix[key + "1"] = proposal_losses1[key]
            if do_B:
                proposals_rpn2, proposal_losses2 = self.proposal_generator2(
                    images, features2, gt_instances
                )            
                proposal_losses2_prefix = {}
                for key in proposal_losses2.keys():
                    proposal_losses2_prefix[key + "2"] = proposal_losses2[key]

            # # roi_head lower branch
            if do_A:
                _, detector_losses1 = self.roi_heads1(
                    images, features1, proposals_rpn1, gt_instances, branch=branch
                )
                detector_losses1_prefix = {}
                for key in detector_losses1.keys():
                    detector_losses1_prefix[key + "1"] = detector_losses1[key]
            if do_B:
                _, detector_losses2 = self.roi_heads2(
                    images, features2, proposals_rpn2, gt_instances, branch=branch
                )
                detector_losses2_prefix = {}
                for key in detector_losses2.keys():
                    detector_losses2_prefix[key + "2"] = detector_losses2[key]

            losses = {}
            if do_A:
                losses.update(proposal_losses1_prefix)
                losses.update(detector_losses1_prefix)
            if do_B:
                losses.update(proposal_losses2_prefix)
                losses.update(detector_losses2_prefix)
            return losses, [], [], None

        elif branch == "unsup_data_weak":            
            # Region proposal network
            if do_A:
                proposals_rpn1, _ = self.proposal_generator1(
                    images, features1, None, compute_loss=False
                )
            if do_B:
                proposals_rpn2, _ = self.proposal_generator2(
                    images, features2, None, compute_loss=False
                )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            if do_A:
                proposals_roih1, ROI_predictions1 = self.roi_heads1(
                    images,
                    features1,
                    proposals_rpn1,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )
            if do_B:
                proposals_roih2, ROI_predictions2 = self.roi_heads2(
                    images,
                    features2,
                    proposals_rpn2,
                    targets=None,
                    compute_loss=False,
                    branch=branch,
                )
            if do_A and not do_B:
                return ({}, proposals_rpn1, proposals_roih1, ROI_predictions1)
            if not do_A and do_B:
                return ({}, proposals_rpn2, proposals_roih2, ROI_predictions2)
            if do_A and do_B:
                return ({}, proposals_rpn1, proposals_roih1, ROI_predictions1), ({}, proposals_rpn2, proposals_roih2, ROI_predictions2)

        elif branch == "val_loss":
            assert do_A and do_B
            # Region proposal network
            proposals_rpn1, proposal_losses1 = self.proposal_generator1(
                images, features1, gt_instances, compute_val_loss=True
            )
            proposals_rpn2, proposal_losses2 = self.proposal_generator2(
                images, features2, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses1 = self.roi_heads1(
                images,
                features1,
                proposals_rpn1,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )
            _, detector_losses2 = self.roi_heads2(
                images,
                features2,
                proposals_rpn2,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            proposal_losses1_prefix = {}
            for key in proposal_losses1.keys():
                proposal_losses1_prefix[key + "1"] = proposal_losses1[key]
            proposal_losses2_prefix = {}
            for key in proposal_losses2.keys():
                proposal_losses2_prefix[key + "2"] = proposal_losses2[key]

            detector_losses1_prefix = {}
            for key in detector_losses1.keys():
                detector_losses1_prefix[key + "1"] = detector_losses1[key]
            detector_losses2_prefix = {}
            for key in detector_losses2.keys():
                detector_losses2_prefix[key + "2"] = detector_losses2[key]

            losses = {}
            losses.update(proposal_losses1_prefix)
            losses.update(detector_losses1_prefix)
            losses.update(proposal_losses2_prefix)
            losses.update(detector_losses2_prefix)
            return losses, [], [], None
