"""Shared two-stage OBB dataflow for independently implemented official-core ports."""

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmrotate.core import rbbox2roi

from sfod.rotated_unbiased_teacher import UnbiasedTeacher


def view_scale(meta, reference):
    # The current strong pipeline has no resize and therefore no scale_factor.
    scale = reference.new_tensor(meta.get('scale_factor', 1.0)).flatten()
    return scale.repeat(4) if scale.numel() == 1 else scale


def map_shared_geometry(boxes, weak_meta, strong_meta):
    """Map resized OBBs between the existing shared-flip/photometric views."""
    if (weak_meta['ori_filename'] != strong_meta['ori_filename']
            or weak_meta.get('flip', False) != strong_meta.get('flip', False)
            or weak_meta.get('flip_direction') != strong_meta.get('flip_direction')):
        raise ValueError('Proposal ports require the shared geometric augmentation')
    mapped = boxes.clone()
    mapped[:, :4] *= view_scale(strong_meta, boxes) / view_scale(weak_meta, boxes)
    return mapped


def preclassifier_roi_forward(roi_head, features, proposals):
    """Capture the actual shared-FC output, not pooled256-D ROI tensors."""
    captured = []
    hook = roi_head.bbox_head.fc_cls.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0]))
    try:
        result = roi_head._bbox_forward(features, rbbox2roi(proposals))
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected exactly one pre-classifier ROI tensor')
    return {'cls_score': result['cls_score'], 'bbox_pred': result['bbox_pred'],
            'preclassifier': captured[0]}


class ProposalAlignedTeacher(UnbiasedTeacher):
    """Image-only pseudo detection plus per-image, proposal-aligned auxiliary loss.

    The teacher is frozen during forward/backward. A method-specific hook updates
    it after an optimizer step or at epoch end. Auxiliary modules belong to the
    Student object, not its plain-detector EMA model.
    """

    epoch_teacher_momentum = None
    iteration_teacher_momentum = None

    def __init__(self, *args, cfg, **kwargs):
        if not cfg.get('strict_source_free') or not cfg.get('use_bbox_reg'):
            raise ValueError('Official proposal ports require image-only data and all four pseudo losses')
        if cfg.get('semantic_reweight') or cfg.get('dynamic_threshold'):
            raise ValueError('Official proposal ports do not use CGA or adaptive pseudo thresholds')
        super().__init__(*args, cfg=cfg, **kwargs)
        if self.ema_model is None:
            raise ValueError('A fixed-source initialized teacher is required')
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

    def forward_train_semi(
            self, img, img_metas, gt_bboxes, gt_labels,
            img_unlabeled, img_metas_unlabeled, gt_bboxes_unlabeled, gt_labels_unlabeled,
            img_unlabeled_1, img_metas_unlabeled_1, gt_bboxes_unlabeled_1, gt_labels_unlabeled_1):
        self._assert_strict_empty_targets(
            gt_bboxes=gt_bboxes, gt_labels=gt_labels,
            gt_bboxes_unlabeled=gt_bboxes_unlabeled, gt_labels_unlabeled=gt_labels_unlabeled,
            gt_bboxes_unlabeled_1=gt_bboxes_unlabeled_1, gt_labels_unlabeled_1=gt_labels_unlabeled_1)
        self.cur_iter += 1
        self.image_num += len(img_metas_unlabeled)
        teacher = getattr(self.ema_model, 'module', self.ema_model)
        teacher.eval()
        with torch.no_grad():
            teacher_features = teacher.extract_feat(img_unlabeled)
            all_proposals = teacher.rpn_head.simple_test_rpn(
                teacher_features, img_metas_unlabeled)
            # Keep the common full pseudo/NMS pipeline; cap only the auxiliary branch.
            detections = teacher.roi_head.simple_test(
                teacher_features, all_proposals, img_metas_unlabeled, rescale=True)
        pseudo_boxes, pseudo_labels = self.create_pseudo_results(
            img_unlabeled_1, detections, [], img_unlabeled.device)
        for boxes, meta in zip(pseudo_boxes, img_metas_unlabeled_1):
            boxes[:, :4] *= view_scale(meta, boxes)
        self.analysis()

        student_features = self.extract_feat(img_unlabeled_1)
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, student_proposals = self.rpn_head.forward_train(
            student_features, img_metas_unlabeled_1, pseudo_boxes, gt_labels=None,
            gt_bboxes_ignore=None, proposal_cfg=proposal_cfg)
        roi_losses = self.roi_head.forward_train(
            student_features, img_metas_unlabeled_1, student_proposals,
            pseudo_boxes, pseudo_labels, gt_bboxes_ignore=None, gt_masks=None)
        losses = self.parse_loss({**rpn_losses, **roi_losses})
        losses = {f'{name}_unlabeled': value * self.weight_u if 'loss' in name else value
                  for name, value in losses.items()}
        losses.update(self.auxiliary_losses(
            teacher, teacher_features, student_features, all_proposals, pseudo_boxes,
            img_metas_unlabeled, img_metas_unlabeled_1))
        losses['pseudo_num'] = img_unlabeled.new_tensor(self.pseudo_num.sum() / self.image_num)
        return losses

    def auxiliary_losses(self, teacher, teacher_features, student_features,
                         all_proposals, pseudo_boxes, img_metas_unlabeled, img_metas_unlabeled_1):
        proposals = [p[:300, :5].detach() for p in all_proposals]
        strong_proposals = [
            map_shared_geometry(p, weak, strong) for p, weak, strong in
            zip(proposals, img_metas_unlabeled, img_metas_unlabeled_1)]
        with torch.no_grad():
            teacher_roi = preclassifier_roi_forward(teacher.roi_head, teacher_features, proposals)
        student_roi = preclassifier_roi_forward(self.roi_head, student_features, strong_proposals)
        counts = [len(p) for p in proposals]
        per_image = []
        start = 0
        for index, count in enumerate(counts):
            selected = slice(start, start + count)
            per_image.append(self.proposal_loss(
                proposals[index], pseudo_boxes[index],
                {k: teacher_roi[k][selected] for k in ('cls_score', 'bbox_pred', 'preclassifier')},
                {k: student_roi[k][selected] for k in ('cls_score', 'bbox_pred', 'preclassifier')},
                img_metas_unlabeled[index], img_metas_unlabeled_1[index]))
            start += count
        return {name: sum(row[name] for row in per_image) / len(per_image)
                for name in per_image[0]}

    def proposal_loss(self, proposals, hpl, teacher_roi, student_roi, weak_meta, strong_meta):
        raise NotImplementedError


@HOOKS.register_module()
class EpochFinalTeacherHook(Hook):
    """Configure at HIGH priority, before the NORMAL checkpoint hook."""

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        if not isinstance(model, ProposalAlignedTeacher):
            raise TypeError('EpochFinalTeacherHook requires a proposal-port model')
        if model.epoch_teacher_momentum is None:
            raise ValueError('The official method must declare its epoch EMA retention')

    def after_train_epoch(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        model.update_ema_model(model.epoch_teacher_momentum)
        model.ema_model.eval()


@HOOKS.register_module()
class AfterOptimizerTeacherHook(Hook):
    """Priority45: after the standard optimizer40, before checkpoints50."""

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        if not isinstance(model, ProposalAlignedTeacher) or model.iteration_teacher_momentum is None:
            raise TypeError('AfterOptimizerTeacherHook requires an iteration-EMA port model')

    def after_train_iter(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        model.update_ema_model(model.iteration_teacher_momentum)
        model.ema_model.eval()
