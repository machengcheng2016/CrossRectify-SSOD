#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer

# hacky way to register
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin

from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        Trainer = UBTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")
    
    res = []
    if cfg.SEMISUPNET.Trainer == "ubteacher":
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        #res = Trainer.test(cfg, ensem_ts_model.modelStudent)#Teacher)

        ensem_ts_model.modelStudent.eval()
        ensem_ts_model.modelStudent.train(False)
        import torch
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = Trainer.build_test_loader(cfg, dataset_name)
            #results_i = inference_on_dataset(model, data_loader, evaluator)
            for idx, inputs in enumerate(data_loader):
                with torch.no_grad():
                    outputs = ensem_ts_model.modelStudent(inputs)
                pred_classes = outputs[0]['instances'].pred_classes.tolist()
                scores = outputs[0]['instances'].scores.tolist()
                pred_boxes = outputs[0]['instances'].pred_boxes.tensor.tolist()
                assert len(pred_classes) == len(scores) and len(scores) == len(pred_boxes)
                for iii in range(len(pred_classes)):
                    x, y, w, h = pred_boxes[iii][0], pred_boxes[iii][1], pred_boxes[iii][2]-pred_boxes[iii][0], pred_boxes[iii][3]-pred_boxes[iii][1]
                    pred_class = pred_classes[iii] + 1
                    score = scores[iii]
                    tmp_dir = {'image_id':int(inputs[0]['image_id']), 'category_id':pred_class, 'bbox':[x, y, w, h], 'score':score}
                    res += [tmp_dir]
                if (idx+1) % 500 == 0:
                    print("Test Dataset: {}, {} images are processed by modelStudent in {}".format(dataset_name, idx+1, (cfg.MODEL.WEIGHTS).split('/')[-1]))
                    
    else:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        #res = Trainer.test(cfg, model)
        
        model.eval()
        model.train(False)
        import torch
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = Trainer.build_test_loader(cfg, dataset_name)
            #results_i = inference_on_dataset(model, data_loader, evaluator)
            for idx, inputs in enumerate(data_loader):
                with torch.no_grad():
                    outputs = model(inputs)
                pred_classes = outputs[0]['instances'].pred_classes.tolist()
                scores = outputs[0]['instances'].scores.tolist()
                pred_boxes = outputs[0]['instances'].pred_boxes.tensor.tolist()
                assert len(pred_classes) == len(scores) and len(scores) == len(pred_boxes)
                for iii in range(len(pred_classes)):
                    x, y, w, h = pred_boxes[iii][0], pred_boxes[iii][1], pred_boxes[iii][2]-pred_boxes[iii][0], pred_boxes[iii][3]-pred_boxes[iii][1]
                    pred_class = pred_classes[iii] + 1
                    score = scores[iii]
                    tmp_dir = {'image_id':int(inputs[0]['image_id']), 'category_id':pred_class, 'bbox':[x, y, w, h], 'score':score}
                    res += [tmp_dir]
                if (idx+1) % 500 == 0:
                    print("Test Dataset: {}, {} images are processed by modelStudent in {}".format(dataset_name, idx+1, (cfg.MODEL.WEIGHTS).split('/')[-1]))
                    
    import json
    import os
    json_name = (cfg.MODEL.WEIGHTS).split('/')[-1][:-4]
    with open(os.path.join("output", json_name+"-student-output.json"), 'w') as fid:
        json.dump(res, fid)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
