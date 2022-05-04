import torch
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid

from mmdet.core.bbox.iou_calculators import bbox_overlaps

@DETECTORS.register_module()
class SoftTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        super(SoftTeacher, self).__init__(
            dict(teacher=build_detector(model), student=build_detector(model)),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        #! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        #! it means that at least one sample for each group should be provided on each gpu.
        #! In some situation, we can only put one image per gpu, we have to return the sum of loss
        #! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num1": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(teacher_data["img"][torch.Tensor(tidx).to(teacher_data["img"].device).long()], [teacher_data["img_metas"][idx] for idx in tidx], [teacher_data["proposals"][idx] for idx in tidx] if ("proposals" in teacher_data) and (teacher_data["proposals"] is not None) else None)
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info)

    def compute_pseudo_label_loss(self, student_info, teacher_info):
        M1 = self._get_trans_mat(teacher_info["transform_matrix1"], student_info["transform_matrix1"])
        M2 = self._get_trans_mat(teacher_info["transform_matrix2"], student_info["transform_matrix2"])

        pseudo_bboxes1 = self._transform_bbox(teacher_info["det_bboxes1"], M1, [meta["img_shape"] for meta in student_info["img_metas"]])
        pseudo_bboxes2 = self._transform_bbox(teacher_info["det_bboxes2"], M2, [meta["img_shape"] for meta in student_info["img_metas"]])
        
        pseudo_labels1 = teacher_info["det_labels1"]
        pseudo_labels2 = teacher_info["det_labels2"]
        
        loss = {}
        rpn_loss1, proposal_list1 = self.rpn_loss1(student_info["rpn_out1"], pseudo_bboxes1, student_info["img_metas"], student_info=student_info)
        rpn_loss2, proposal_list2 = self.rpn_loss2(student_info["rpn_out2"], pseudo_bboxes2, student_info["img_metas"], student_info=student_info)
        
        loss.update(rpn_loss1)
        loss.update(rpn_loss2)
        
        if proposal_list1 is not None:
            student_info["proposals1"] = proposal_list1
        if proposal_list2 is not None:
            student_info["proposals2"] = proposal_list2
            
        if self.train_cfg.use_teacher_proposal:
            proposals1 = self._transform_bbox(teacher_info["proposals1"], M1, [meta["img_shape"] for meta in student_info["img_metas"]])
        else:
            proposals1 = student_info["proposals1"]
            
        if self.train_cfg.use_teacher_proposal:
            proposals2 = self._transform_bbox(teacher_info["proposals2"], M2, [meta["img_shape"] for meta in student_info["img_metas"]])
        else:
            proposals2 = student_info["proposals2"]

        loss.update(
            self.unsup_rcnn_cls_loss1(
                student_info["backbone_feature1"],
                student_info["img_metas"],
                proposals1,
                pseudo_bboxes1,
                pseudo_labels1,
                teacher_info["transform_matrix1"],
                student_info["transform_matrix1"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature1"],
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup_rcnn_cls_loss2(
                student_info["backbone_feature2"],
                student_info["img_metas"],
                proposals2,
                pseudo_bboxes2,
                pseudo_labels2,
                teacher_info["transform_matrix2"],
                student_info["transform_matrix2"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature2"],
                student_info=student_info,
            )
        )
                
        loss.update(
            self.unsup_rcnn_reg_loss1(
                student_info["backbone_feature1"],
                student_info["img_metas"],
                proposals1,
                pseudo_bboxes1,
                pseudo_labels1,
                student_info=student_info,
            )
        )
        
        loss.update(
            self.unsup_rcnn_reg_loss2(
                student_info["backbone_feature2"],
                student_info["img_metas"],
                proposals2,
                pseudo_bboxes2,
                pseudo_labels2,
                student_info=student_info,
            )
        )
        
        return loss

    def rpn_loss1(
        self,
        rpn_out1,
        pseudo_bboxes1,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes1:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num1": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs1 = rpn_out1 + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head1.loss(
                *loss_inputs1, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head1.get_bboxes(
                *rpn_out1, img_metas, cfg=proposal_cfg
            )
            #log_image_with_boxes(
            #    "rpn",
            #    student_info["img"][0],
            #    pseudo_bboxes1[0][:, :4],
            #    bbox_tag="rpn_pseudo_label",
            #    scores=pseudo_bboxes1[0][:, 4],
            #    interval=500,
            #    img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            #)
            losses1 = {}
            for key, value in losses.items():
                losses1.update({key+'1': value})
            return losses1, proposal_list
        else:
            return {}, None

    def rpn_loss2(
        self,
        rpn_out2,
        pseudo_bboxes2,
        img_metas,
        gt_bboxes_ignore=None,
        student_info=None,
        **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes2:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                        :, 4
                    ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num2": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs2 = rpn_out2 + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head2.loss(
                *loss_inputs2, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head2.get_bboxes(
                *rpn_out2, img_metas, cfg=proposal_cfg
            )
            #log_image_with_boxes(
            #    "rpn",
            #    student_info["img"][0],
            #    pseudo_bboxes2[0][:, :4],
            #    bbox_tag="rpn_pseudo_label",
            #    scores=pseudo_bboxes2[0][:, 4],
            #    interval=500,
            #    img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            #)
            losses2 = {}
            for key, value in losses.items():
                losses2.update({key+'2': value})
            return losses2, proposal_list
        else:
            return {}, None


    def unsup_rcnn_cls_loss1(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num1": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result1(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head1._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head1.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head1.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head1.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head1.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        #if len(gt_bboxes[0]) > 0:
        #    log_image_with_boxes(
        #        "rcnn_cls",
        #        student_info["img"][0],
        #        gt_bboxes[0],
        #        bbox_tag="pseudo_label",
        #        labels=gt_labels[0],
        #        class_names=self.CLASSES,
        #        interval=500,
        #        img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #    )
        loss1 = {}
        for key, value in loss.items():
            loss1.update({key+'1': value})
        return loss1

    def unsup_rcnn_cls_loss2(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        teacher_transMat,
        student_transMat,
        teacher_img_metas,
        teacher_feat,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num2": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result2(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head2._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head2.bbox_head.get_targets(
            sampling_results, gt_bboxes, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head2.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head2.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head2.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["loss_cls"] = loss["loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["loss_bbox"] = loss["loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        #if len(gt_bboxes[0]) > 0:
        #    log_image_with_boxes(
        #        "rcnn_cls",
        #        student_info["img"][0],
        #        gt_bboxes[0],
        #        bbox_tag="pseudo_label",
        #        labels=gt_labels[0],
        #        class_names=self.CLASSES,
        #        interval=500,
        #        img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #    )
        loss2 = {}
        for key, value in loss.items():
            loss2.update({key+'2': value})
        return loss2

    def unsup_rcnn_reg_loss1(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num1": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head1.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        #if len(gt_bboxes[0]) > 0:
        #    log_image_with_boxes(
        #        "rcnn_reg",
        #        student_info["img"][0],
        #        gt_bboxes[0],
        #        bbox_tag="pseudo_label",
        #        labels=gt_labels[0],
        #        class_names=self.CLASSES,
        #        interval=500,
        #        img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #    )
        return {"loss_bbox1": loss_bbox}

    def unsup_rcnn_reg_loss2(
        self,
        feat,
        img_metas,
        proposal_list,
        pseudo_bboxes,
        pseudo_labels,
        student_info=None,
        **kwargs,
    ):
        gt_bboxes, gt_labels, _ = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num2": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        loss_bbox = self.student.roi_head2.forward_train(
            feat, img_metas, proposal_list, gt_bboxes, gt_labels, **kwargs
        )["loss_bbox"]
        #if len(gt_bboxes[0]) > 0:
        #    log_image_with_boxes(
        #        "rcnn_reg",
        #        student_info["img"][0],
        #        gt_bboxes[0],
        #        bbox_tag="pseudo_label",
        #        labels=gt_labels[0],
        #        class_names=self.CLASSES,
        #        interval=500,
        #        img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #    )
        return {"loss_bbox2": loss_bbox}

    def get_sampling_result1(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head1.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head1.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    def get_sampling_result2(
        self,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head2.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head2.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat1, feat2 = self.student.extract_feat(img)
        student_info["backbone_feature1"] = feat1
        student_info["backbone_feature2"] = feat2
        if self.student.with_rpn:
            rpn_out1 = self.student.rpn_head1(feat1)
            rpn_out2 = self.student.rpn_head2(feat2)
            student_info["rpn_out1"] = list(rpn_out1)
            student_info["rpn_out2"] = list(rpn_out2)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix1"] = [torch.from_numpy(meta["transform_matrix"]).float().to(feat1[0][0].device) for meta in img_metas]
        student_info["transform_matrix2"] = [torch.from_numpy(meta["transform_matrix"]).float().to(feat2[0][0].device) for meta in img_metas]
        return student_info

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):
        teacher_info = {}
        feat1, feat2 = self.teacher.extract_feat(img)
        teacher_info["backbone_feature1"], teacher_info["backbone_feature2"] = feat1, feat2
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get("rpn_proposal", self.teacher.test_cfg.rpn)
            rpn_out1 = list(self.teacher.rpn_head1(feat1))
            rpn_out2 = list(self.teacher.rpn_head2(feat2))
            proposal_list1 = self.teacher.rpn_head1.get_bboxes(*rpn_out1, img_metas, cfg=proposal_cfg)
            proposal_list2 = self.teacher.rpn_head2.get_bboxes(*rpn_out2, img_metas, cfg=proposal_cfg)
        else:
            proposal_list1 = proposal_list2 = proposals
        teacher_info["proposals1"] = proposal_list1
        teacher_info["proposals2"] = proposal_list2

        proposal_list1, proposal_label_list1 = self.teacher.roi_head1.simple_test_bboxes(feat1, img_metas, proposal_list1, self.teacher.test_cfg.rcnn, rescale=False)
        proposal_list2, proposal_label_list2 = self.teacher.roi_head2.simple_test_bboxes(feat2, img_metas, proposal_list2, self.teacher.test_cfg.rcnn, rescale=False)

        proposal_list1 = [p.to(feat1[0].device) for p in proposal_list1]
        proposal_list2 = [p.to(feat2[0].device) for p in proposal_list2]
        proposal_list1 = [p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list1]
        proposal_list2 = [p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list2]
        proposal_label_list1 = [p.to(feat1[0].device) for p in proposal_label_list1]
        proposal_label_list2 = [p.to(feat2[0].device) for p in proposal_label_list2]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list1, proposal_label_list1, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal1,
                        proposal_label1,
                        proposal1[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal1, proposal_label1 in zip(
                        proposal_list1, proposal_label_list1
                    )
                ]
            )
        )
        proposal_list2, proposal_label_list2, _ = list(
            zip(
                *[
                    filter_invalid(
                        proposal2,
                        proposal_label2,
                        proposal2[:, -1],
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal2, proposal_label2 in zip(
                        proposal_list2, proposal_label_list2
                    )
                ]
            )
        )
        det_bboxes1 = proposal_list1
        det_bboxes2 = proposal_list2
        reg_unc1 = self.compute_uncertainty_with_aug1(feat1, img_metas, proposal_list1, proposal_label_list1)
        reg_unc2 = self.compute_uncertainty_with_aug2(feat2, img_metas, proposal_list2, proposal_label_list2)
        det_bboxes1 = [torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes1, reg_unc1)]
        det_bboxes2 = [torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes2, reg_unc2)]
        det_labels1 = proposal_label_list1
        det_labels2 = proposal_label_list2
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CROSS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        det_bboxes1_cross = []
        det_bboxes2_cross = []
        det_labels1_cross = []
        det_labels2_cross = []
        
        assert len(det_bboxes1) == len(det_bboxes2)
        for iii in range(len(det_bboxes1)):
            if det_bboxes1[iii].shape[0] > 0 and det_bboxes2[iii].shape[0] > 0:
                det_score1 = det_bboxes1[iii][:,4]
                det_score2 = det_bboxes2[iii][:,4]
                
                ious = bbox_overlaps(det_bboxes1[iii][:,:4], det_bboxes2[iii][:,:4])
                ious_max1, ious_idx1 = ious.max(1)
                ious_max2, ious_idx2 = ious.max(0)
                assert len(ious_max1) == len(det_bboxes1[iii]) == len(det_labels1[iii]) and len(ious_max2) == len(det_bboxes2[iii]) == len(det_labels2[iii])
                
                # as for 1
                matched1 = ious_max1 > 0.5
                comparison1 = det_score1[matched1] > det_score2[ious_idx1[matched1]]
                det_bbox1_match = torch.cat((det_bboxes1[iii][matched1][comparison1], det_bboxes2[iii][ious_idx1[matched1]][~comparison1]))
                det_bbox1_cross = torch.cat((det_bboxes1[iii][~matched1], det_bbox1_match))
                det_label1_match = torch.cat((det_labels1[iii][matched1][comparison1], det_labels2[iii][ious_idx1[matched1]][~comparison1]))
                det_label1_cross = torch.cat((det_labels1[iii][~matched1], det_label1_match))
                            
                # as for 2
                matched2 = ious_max2 > 0.5
                comparison2 = det_score2[matched2] > det_score1[ious_idx2[matched2]]
                det_bbox2_match = torch.cat((det_bboxes2[iii][matched2][comparison2], det_bboxes1[iii][ious_idx2[matched2]][~comparison2]))
                det_bbox2_cross = torch.cat((det_bboxes2[iii][~matched2], det_bbox2_match))
                det_label2_match = torch.cat((det_labels2[iii][matched2][comparison2], det_labels1[iii][ious_idx2[matched2]][~comparison2]))
                det_label2_cross = torch.cat((det_labels2[iii][~matched2], det_label2_match))
                
                det_bboxes1_cross.append(det_bbox1_cross)
                det_bboxes2_cross.append(det_bbox2_cross)
                det_labels1_cross.append(det_label1_cross)
                det_labels2_cross.append(det_label2_cross)
                '''
                if matched1.sum() > 0:
                    print("="*30, det_bboxes1[iii], det_bboxes1_cross[iii], det_labels1[iii], det_labels1_cross[iii])
                    assert False
                if matched2.sum() > 0:
                    print("="*30, det_bboxes2[iii], det_bboxes2_cross[iii], det_labels1[iii], det_labels1_cross[iii])
                    assert False
                '''
            else:
                det_bboxes1_cross.append(det_bboxes1[iii])
                det_bboxes2_cross.append(det_bboxes2[iii])
                det_labels1_cross.append(det_labels1[iii])
                det_labels2_cross.append(det_labels2[iii])
                
        teacher_info["det_bboxes1"] = det_bboxes1_cross
        teacher_info["det_bboxes2"] = det_bboxes2_cross
        teacher_info["det_labels1"] = det_labels1_cross
        teacher_info["det_labels2"] = det_labels2_cross
        teacher_info["transform_matrix1"] = [torch.from_numpy(meta["transform_matrix"]).float().to(feat1[0][0].device) for meta in img_metas]
        teacher_info["transform_matrix2"] = [torch.from_numpy(meta["transform_matrix"]).float().to(feat2[0][0].device) for meta in img_metas]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug1(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head1.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    def compute_uncertainty_with_aug2(
        self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head2.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc


    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                torch.randn(times, box.shape[0], 4, device=box.device)
                * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
