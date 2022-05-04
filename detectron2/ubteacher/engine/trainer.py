# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from ubteacher.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
from ubteacher.evaluation import DatasetEvaluator, print_csv_format, inference_on_dataset, generate_on_dataset, evaluate_on_dataset
#from ubteacher.solver import build_lr_scheduler, build_optimizer
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

# for measuring pseudo accuracy
from detectron2.modeling.matcher import Matcher
from detectron2.structures.boxes import pairwise_iou
from copy import deepcopy

from ubteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
)
from ubteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate
from ubteacher.engine.hooks import LossEvalHook
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from ubteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from ubteacher.engine.train_loop import DoubleTrainer
from ensemble_boxes import *

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Unbiased Teacher Trainer
class UBTeacherTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)
        data_loader2 = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)

        # create an student ema model
        model_ema = self.build_model(cfg)
        self.model_ema = model_ema

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else DoubleTrainer)(
            model, data_loader, data_loader2, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model, model_ema)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())
        
        self.proposal_matcher = Matcher(self.cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS, self.cfg.MODEL.ROI_HEADS.IOU_LABELS, allow_low_quality_matches=False)            

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            if TORCH_VERSION >= (1, 7):
                self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapperTwoCropSeparate(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            valid_map = proposal_bbox_inst.scores > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map]

        return new_proposal_inst

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if psedo_label_method == "thresholding":
                proposal_bbox_inst = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabeld_data, label):
        '''
        for unlabel_datum, lab_inst in zip(unlabeld_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabeld_data
        '''
        for idx in range(len(unlabeld_data)-1, -1, -1):
            unlabel_datum, lab_inst = unlabeld_data[idx], label[idx]
            if not lab_inst.gt_boxes.tensor.shape[0]:
                del(unlabeld_data[idx])
            else:
                unlabel_datum["instances"] = lab_inst
        return unlabeld_data
                
    def resize_to_small(self, lists, image_size):
        weight, height = image_size
        for i in range(len(lists)):
            lists[i][0] /= weight
            lists[i][1] /= height
            lists[i][2] /= weight
            lists[i][3] /= height
        return lists
                
    def resize_to_large(self, lists, image_size):
        weight, height = image_size
        for i in range(len(lists)):
            lists[i][0] *= weight
            lists[i][1] *= height
            lists[i][2] *= weight
            lists[i][3] *= height
        return lists


    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data1 = next(self._trainer._data_loader_iter)
        data2 = next(self._trainer._data_loader_iter2)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        _, label_data_q1, unlabel_data_q1, unlabel_data_k1 = data1
        _, label_data_q2, unlabel_data_q2, unlabel_data_k2 = data2
        data_time = time.perf_counter() - start

        # save unlabeled data's label for use
        unlabeled_infos1 = [deepcopy(item["instances"]) for item in unlabel_data_q1]
        unlabeled_infos2 = [deepcopy(item["instances"]) for item in unlabel_data_q2]
        
        # remove unlabeled data labels
        unlabel_data_k1 = self.remove_label(unlabel_data_k1)
        unlabel_data_k2 = self.remove_label(unlabel_data_k2)
        unlabel_data_q1 = self.remove_label(unlabel_data_q1)
        unlabel_data_q2 = self.remove_label(unlabel_data_q2)

        # burn-in stage (supervised training with labeled data)
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            record_dict = {}
            record_dict1, _, _, _ = self.model(label_data_q1, branch="supervised", do_B=False)
            record_dict2, _, _, _ = self.model(label_data_q2, branch="supervised", do_A=False)

            # weight losses
            loss_dict = {}
            for key in record_dict1.keys():
                if key[:4] == "loss" and key[-1] == '1':
                    loss_dict[key] = record_dict1[key]
                    record_dict[key] = record_dict1[key]
            for key in record_dict2.keys():
                if key[:4] == "loss" and key[-1] == '2':
                    loss_dict[key] = record_dict2[key]
                    record_dict[key] = record_dict2[key]
            losses = sum(loss_dict.values())
            
        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_student_ema_model(keep_rate=0.00)

            elif (self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_student_ema_model(keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}
            #  generate the pseudo-label using teacher model
            # note that we do not convert to eval mode, as 1) there is no gradient computed in
            # teacher model and 2) batch norm layers are not updated as well
            with torch.no_grad():
                (_, _, proposals_roih_unsup_k1, _), (_, _, proposals_roih_unsup_k1_ema, _) = self.model_ema(unlabel_data_k1, branch="unsup_data_weak")
                (_, _, proposals_roih_unsup_k2_ema, _), (_, _, proposals_roih_unsup_k2, _) = self.model_ema(unlabel_data_k2, branch="unsup_data_weak")

            #  Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            # Pseudo_labeling for ROI head (bbox location/objectness)
            pseudo_proposals_roih_unsup_k1, _ = self.process_pseudo_label(proposals_roih_unsup_k1, cur_threshold, "roih", "thresholding")
            pseudo_proposals_roih_unsup_k2, _ = self.process_pseudo_label(proposals_roih_unsup_k2, cur_threshold, "roih", "thresholding")
            pseudo_proposals_roih_unsup_k1_ema, _ = self.process_pseudo_label(proposals_roih_unsup_k1_ema, cur_threshold, "roih", "thresholding")
            pseudo_proposals_roih_unsup_k2_ema, _ = self.process_pseudo_label(proposals_roih_unsup_k2_ema, cur_threshold, "roih", "thresholding")

            # assigning labels for the counterpart, then evaluating pseudo label accuracy.
            image_sizes1 = [item.image_size for item in pseudo_proposals_roih_unsup_k1]
            pseudo_boxes1 = [item.gt_boxes for item in pseudo_proposals_roih_unsup_k1]
            pseudo_scores1 = [item.scores for item in pseudo_proposals_roih_unsup_k1]
            pseudo_labels1 = [item.gt_classes for item in pseudo_proposals_roih_unsup_k1]

            image_sizes1_ema = [item.image_size for item in pseudo_proposals_roih_unsup_k1_ema]
            pseudo_boxes1_ema = [item.gt_boxes for item in pseudo_proposals_roih_unsup_k1_ema]
            pseudo_scores1_ema = [item.scores for item in pseudo_proposals_roih_unsup_k1_ema]
            pseudo_labels1_ema = [item.gt_classes for item in pseudo_proposals_roih_unsup_k1_ema]

            for (image_size1, image_size1_ema) in zip(image_sizes1, image_sizes1_ema):
                assert image_size1 == image_size1_ema

            pseudo_proposals_roih_unsup_k1_from_ema = []
            for iii, (image_size1, pseudo_box1, pseudo_label1, pseudo_score1, pseudo_box1_ema, pseudo_label1_ema, pseudo_score1_ema) in enumerate(zip(image_sizes1, pseudo_boxes1, pseudo_labels1, pseudo_scores1, pseudo_boxes1_ema, pseudo_labels1_ema, pseudo_scores1_ema)):
                pseudo_proposals_roih_unsup_k1_from_ema.append(Instances(image_size1))
                if pseudo_label1.numel() == 0 or pseudo_label1_ema.numel() == 0:    # neither model A nor model B can see any objects.
                    pseudo_proposals_roih_unsup_k1_from_ema[iii]._fields = {'gt_boxes': Boxes(torch.FloatTensor([]).to(pseudo_proposals_roih_unsup_k1[iii].gt_boxes.device)), 'gt_classes': torch.LongTensor([]).to(pseudo_proposals_roih_unsup_k1[iii].gt_classes.device), 'scores': torch.FloatTensor([]).to(pseudo_proposals_roih_unsup_k1[iii].scores.device)}
                else:
                    match_quality_matrix1_ema_for_1 = pairwise_iou(pseudo_box1_ema, pseudo_box1)
                    matched_idxs1_ema_for_1, matched_labels1_ema_for_1 = self.proposal_matcher(match_quality_matrix1_ema_for_1)
                    assert not (-1 in matched_labels1_ema_for_1)
                    
                    pseudo_box1_ema_matched = pseudo_box1_ema[matched_idxs1_ema_for_1][matched_labels1_ema_for_1 == 1]
                    pseudo_label1_ema_matched = pseudo_label1_ema[matched_idxs1_ema_for_1][matched_labels1_ema_for_1 == 1]
                    pseudo_score1_ema_matched = pseudo_score1_ema[matched_idxs1_ema_for_1][matched_labels1_ema_for_1 == 1]
                    pseudo_box1 = pseudo_box1[matched_labels1_ema_for_1 == 1]
                    pseudo_label1 = pseudo_label1[matched_labels1_ema_for_1 == 1]
                    pseudo_score1 = pseudo_score1[matched_labels1_ema_for_1 == 1]
                    
                    mask = (pseudo_score1_ema_matched > pseudo_score1).float()
                    pseudo_label1 = (pseudo_label1 * (mask == 0).float() + pseudo_label1_ema_matched * (mask == 1).float()).long()
                    pseudo_box1 = Boxes(((pseudo_box1.tensor*(1-mask.unsqueeze(1).expand_as(pseudo_box1.tensor)) + pseudo_box1_ema_matched.tensor*(mask.unsqueeze(1).expand_as(pseudo_box1_ema_matched.tensor)))).to(pseudo_proposals_roih_unsup_k1[iii].gt_boxes.device))
                    #'''
                    remain_idx_in_1_ema = list(set(range(len(pseudo_label1_ema))) - set(matched_idxs1_ema_for_1[matched_labels1_ema_for_1 == 1].tolist()))
                    if len(remain_idx_in_1_ema):
                        pseudo_label1_ema_remain = pseudo_label1_ema[remain_idx_in_1_ema]
                        pseudo_box1_ema_remain = pseudo_box1_ema[remain_idx_in_1_ema]
                        pseudo_label1 = torch.cat((pseudo_label1, pseudo_label1_ema_remain))
                        pseudo_box1 = Boxes(torch.cat((pseudo_box1.tensor, pseudo_box1_ema_remain.tensor), dim=0).to(pseudo_proposals_roih_unsup_k1[iii].gt_boxes.device))
                    #'''
                    pseudo_proposals_roih_unsup_k1_from_ema[iii]._fields = {'gt_boxes': deepcopy(pseudo_box1), 'gt_classes': deepcopy(pseudo_label1)}
            
            
            
            # assigning labels for the counterpart, then evaluating pseudo label accuracy.
            image_sizes2 = [item.image_size for item in pseudo_proposals_roih_unsup_k2]
            pseudo_boxes2 = [item.gt_boxes for item in pseudo_proposals_roih_unsup_k2]
            pseudo_scores2 = [item.scores for item in pseudo_proposals_roih_unsup_k2]
            pseudo_labels2 = [item.gt_classes for item in pseudo_proposals_roih_unsup_k2]

            image_sizes2_ema = [item.image_size for item in pseudo_proposals_roih_unsup_k2_ema]
            pseudo_boxes2_ema = [item.gt_boxes for item in pseudo_proposals_roih_unsup_k2_ema]
            pseudo_scores2_ema = [item.scores for item in pseudo_proposals_roih_unsup_k2_ema]
            pseudo_labels2_ema = [item.gt_classes for item in pseudo_proposals_roih_unsup_k2_ema]

            for (image_size2, image_size2_ema) in zip(image_sizes2, image_sizes2_ema):
                assert image_size2 == image_size2_ema

            pseudo_proposals_roih_unsup_k2_from_ema = []
            for jjj, (image_size2, pseudo_box2, pseudo_label2, pseudo_score2, pseudo_box2_ema, pseudo_label2_ema, pseudo_score2_ema) in enumerate(zip(image_sizes2, pseudo_boxes2, pseudo_labels2, pseudo_scores2, pseudo_boxes2_ema, pseudo_labels2_ema, pseudo_scores2_ema)):
                pseudo_proposals_roih_unsup_k2_from_ema.append(Instances(image_size2))
                if pseudo_label2.numel() == 0 or pseudo_label2_ema.numel() == 0:    # neither model A nor model B can see any objects.
                    pseudo_proposals_roih_unsup_k2_from_ema[jjj]._fields = {'gt_boxes': Boxes(torch.FloatTensor([]).to(pseudo_proposals_roih_unsup_k2[jjj].gt_boxes.device)), 'gt_classes': torch.LongTensor([]).to(pseudo_proposals_roih_unsup_k2[jjj].gt_classes.device), 'scores': torch.FloatTensor([]).to(pseudo_proposals_roih_unsup_k2[jjj].scores.device)}
                else:
                    match_quality_matrix2_ema_for_2 = pairwise_iou(pseudo_box2_ema, pseudo_box2)
                    matched_idxs2_ema_for_2, matched_labels2_ema_for_2 = self.proposal_matcher(match_quality_matrix2_ema_for_2)
                    assert not (-1 in matched_labels2_ema_for_2)
                    
                    pseudo_box2_ema_matched = pseudo_box2_ema[matched_idxs2_ema_for_2][matched_labels2_ema_for_2 == 1]
                    pseudo_label2_ema_matched = pseudo_label2_ema[matched_idxs2_ema_for_2][matched_labels2_ema_for_2 == 1]
                    pseudo_score2_ema_matched = pseudo_score2_ema[matched_idxs2_ema_for_2][matched_labels2_ema_for_2 == 1]
                    pseudo_box2 = pseudo_box2[matched_labels2_ema_for_2 == 1]
                    pseudo_label2 = pseudo_label2[matched_labels2_ema_for_2 == 1]
                    pseudo_score2 = pseudo_score2[matched_labels2_ema_for_2 == 1]    
                    
                    mask2 = (pseudo_score2_ema_matched > pseudo_score2).float()
                    pseudo_label2 = (pseudo_label2 * (mask2 == 0).float() + pseudo_label2_ema_matched * (mask2 == 1).float()).long()
                    pseudo_box2 = Boxes(((pseudo_box2.tensor*(1-mask2.unsqueeze(1).expand_as(pseudo_box2.tensor)) + pseudo_box2_ema_matched.tensor*(mask2.unsqueeze(1).expand_as(pseudo_box2_ema_matched.tensor)))).to(pseudo_proposals_roih_unsup_k2[jjj].gt_boxes.device))
                    #'''
                    remain_idx_in_2_ema = list(set(range(len(pseudo_label2_ema))) - set(matched_idxs2_ema_for_2[matched_labels2_ema_for_2 == 1].tolist()))
                    if len(remain_idx_in_2_ema):
                        pseudo_label2_ema_remain = pseudo_label2_ema[remain_idx_in_2_ema]
                        pseudo_box2_ema_remain = pseudo_box2_ema[remain_idx_in_2_ema]
                        pseudo_label2 = torch.cat((pseudo_label2, pseudo_label2_ema_remain))
                        pseudo_box2 = Boxes(torch.cat((pseudo_box2.tensor, pseudo_box2_ema_remain.tensor), dim=0).to(pseudo_proposals_roih_unsup_k2[jjj].gt_boxes.device))                        
                    #'''
                    pseudo_proposals_roih_unsup_k2_from_ema[jjj]._fields = {'gt_boxes': deepcopy(pseudo_box2), 'gt_classes': deepcopy(pseudo_label2)}


            groundtruth_boxes1 = [item.gt_boxes for item in unlabeled_infos1]
            groundtruth_labels1 = [item.gt_classes for item in unlabeled_infos1]

            image_sizes1 = [item.image_size for item in pseudo_proposals_roih_unsup_k1_from_ema]
            pseudo_boxes1 = [item.gt_boxes for item in pseudo_proposals_roih_unsup_k1_from_ema]
            pseudo_labels1 = [item.gt_classes for item in pseudo_proposals_roih_unsup_k1_from_ema]

            groundtruth_boxes2 = [item.gt_boxes for item in unlabeled_infos2]
            groundtruth_labels2 = [item.gt_classes for item in unlabeled_infos2]
            
            image_sizes2 = [item.image_size for item in pseudo_proposals_roih_unsup_k2_from_ema]
            pseudo_boxes2 = [item.gt_boxes for item in pseudo_proposals_roih_unsup_k2_from_ema]
            pseudo_labels2 = [item.gt_classes for item in pseudo_proposals_roih_unsup_k2_from_ema]
            


            N1, n1, m1, N2, n2, m2 = 0, 0, 0, 0, 0, 0
            for groundtruth_box1, groundtruth_label1, pseudo_box1, pseudo_label1 in zip(groundtruth_boxes1, groundtruth_labels1, pseudo_boxes1, pseudo_labels1):
                N1 += pseudo_box1.tensor.shape[0]
                if pseudo_box1.tensor.shape[0] == 0:
                    continue
                groundtruth_box1 = groundtruth_box1.to(pseudo_box1.device)
                groundtruth_label1 = groundtruth_label1.to(pseudo_label1.device)
                match_quality_matrix1 = pairwise_iou(groundtruth_box1, pseudo_box1)
                matched_idxs1, matched_labels1 = self.proposal_matcher(match_quality_matrix1)
                has_gt1 = groundtruth_label1.numel() > 0
                n1 += groundtruth_label1.numel()
                if has_gt1:
                    groundtruth_label1 = groundtruth_label1[matched_idxs1]
                    # Label unmatched proposals (0 label from matcher) as background (label=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
                    groundtruth_label1[matched_labels1 == 0] = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
                    # Label ignore proposals (-1 label)
                    groundtruth_label1[matched_labels1 == -1] = -1
                else:
                    groundtruth_label1 = torch.zeros_like(matched_idxs1) + self.cfg.MODEL.ROI_HEADS.NUM_CLASSES                    
                    print("no has_gt1")
                m1 += torch.eq(groundtruth_label1, pseudo_label1).sum().item()
                
            for groundtruth_box2, groundtruth_label2, pseudo_box2, pseudo_label2 in zip(groundtruth_boxes2, groundtruth_labels2, pseudo_boxes2, pseudo_labels2):
                N2 += pseudo_box2.tensor.shape[0]
                if pseudo_box2.tensor.shape[0] == 0:
                    continue
                groundtruth_box2 = groundtruth_box2.to(pseudo_box2.device)
                groundtruth_label2 = groundtruth_label2.to(pseudo_label2.device)
                match_quality_matrix2 = pairwise_iou(groundtruth_box2, pseudo_box2)
                matched_idxs2, matched_labels2 = self.proposal_matcher(match_quality_matrix2)
                has_gt2 = groundtruth_label2.numel() > 0
                n2 += groundtruth_label2.numel()
                if has_gt2:
                    groundtruth_label2 = groundtruth_label2[matched_idxs2]
                    # Label unmatched proposals (0 label from matcher) as background (label=self.cfg.MODEL.ROI_HEADS.NUM_CLASSES)
                    groundtruth_label2[matched_labels2 == 0] = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
                    # Label ignore proposals (-1 label)
                    groundtruth_label2[matched_labels2 == -1] = -1
                else:
                    groundtruth_label2 = torch.zeros_like(matched_idxs2) + self.cfg.MODEL.ROI_HEADS.NUM_CLASSES                    
                    print("no has_gt2")
                m2 += torch.eq(groundtruth_label2, pseudo_label2).sum().item()
                
            print(m1, n1, N1, m2, n2, N2)

            record_dict1, _, _, _ = self.model(label_data_q1, branch="supervised", do_B=False)
            record_dict2, _, _, _ = self.model(label_data_q2, branch="supervised", do_A=False)
            # weight losses
            record_all_label_data = {}
            for key in record_dict1.keys():
                if key[:4] == "loss" and key[-1] == '1':
                    record_all_label_data[key] = record_dict1[key]
            for key in record_dict2.keys():
                if key[:4] == "loss" and key[-1] == '2':
                    record_all_label_data[key] = record_dict2[key]
            record_dict.update(record_all_label_data)

            #  add pseudo-label to unlabeled data
            unlabel_data_q1 = self.add_label(unlabel_data_q1, pseudo_proposals_roih_unsup_k1_from_ema)
            new_record_all_unlabel_data1 = {}
            if len(unlabel_data_q1):
                record_all_unlabel_data1, _, _, _ = self.model(unlabel_data_q1, branch="supervised", do_B=False)
                for key in record_all_unlabel_data1.keys():
                    if key[-1] == '1':
                        new_record_all_unlabel_data1[key + "_pseudo"] = record_all_unlabel_data1[key]
            else:
                for key in record_all_label_data.keys():
                    if key[-1] == '1':
                        new_record_all_unlabel_data1[key + "_pseudo"] = 0.0
            record_dict.update(new_record_all_unlabel_data1)
            
            unlabel_data_q2 = self.add_label(unlabel_data_q2, pseudo_proposals_roih_unsup_k2_from_ema)
            new_record_all_unlabel_data2 = {}
            if len(unlabel_data_q2):
                record_all_unlabel_data2, _, _, _ = self.model(unlabel_data_q2, branch="supervised", do_A=False)
                for key in record_all_unlabel_data2.keys():
                    if key[-1] == '2':
                        new_record_all_unlabel_data2[key + "_pseudo"] = record_all_unlabel_data2[key]
            else:
                for key in record_all_label_data.keys():
                    if key[-1] == '2':
                        new_record_all_unlabel_data2[key + "_pseudo"] = 0.0
                    
            record_dict.update(new_record_all_unlabel_data2)
            
            
            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    # pseudo bbox regression <- 0
                    if key in ["loss_rpn_loc1_pseudo", "loss_box_reg1_pseudo", "loss_rpn_loc2_pseudo", "loss_box_reg2_pseudo"]:
                        loss_dict[key] = record_dict[key] * 0.0
                    # unsupervised loss
                    elif key[-6:] == "pseudo":  
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                    # supervised loss
                    else:  
                        loss_dict[key] = record_dict[key] * 1.0

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_student_ema_model(self, keep_rate=0.996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_student_ema_dict = OrderedDict()
        for key, value in self.model_ema.state_dict().items():
            if key in student_model_dict.keys():
                new_student_ema_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_ema.load_state_dict(new_student_ema_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_ema.load_state_dict(rename_model_dict)
        else:
            self.model_ema.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            return _last_eval_results_student

        def test_and_save_results_student_ema():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_ema)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student_ema))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


    @classmethod
    def generate(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
            generate_on_dataset(model, data_loader, evaluator)

    @classmethod
    def evaluate(cls, cfg, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            evaluate_on_dataset(data_loader, evaluator)


    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:

        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)
