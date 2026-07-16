# Copyright (c) OpenMMLab. All rights reserved.
"""Oriented R-CNN RoI head that supports per-pseudo-GT semantic reweighting.

This head is used by the SARCLIP Semantic Reliability Reweighting (SRW)
method.  The detector score alone decides whether a pseudo box is admitted
(handled upstream in ``UnbiasedTeacher.create_pseudo_results``).  SARCLIP only
supplies a *semantic reliability weight* per pseudo GT, which is applied to the
**ROI positive classification loss** and nowhere else:

- RPN classification / bbox regression are untouched (SRW never reaches RPN).
- ROI negative samples keep ``label_weight = 1``.
- ROI bbox regression is untouched (the SFOD recipe uses ``use_bbox_reg=False``
  anyway, but even if enabled, only ``label_weights`` are scaled here).

When ``gt_semantic_weights`` is ``None`` this head is byte-for-byte equivalent
to :class:`OrientedStandardRoIHead`.
"""
import torch

from mmrotate.core import rbbox2roi
from mmrotate.models.builder import ROTATED_HEADS
from mmrotate.models.roi_heads.oriented_standard_roi_head import (
    OrientedStandardRoIHead,
)


@ROTATED_HEADS.register_module()
class SemanticWeightedOrientedStandardRoIHead(OrientedStandardRoIHead):
    """Oriented RCNN roi head with optional per-GT semantic reweighting."""

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_weights=None):
        """Same as the base head, plus an optional ``gt_semantic_weights``.

        Args:
            gt_semantic_weights (list[Tensor] | None): Per-image, per-GT
                semantic reliability weights aligned to ``gt_bboxes`` /
                ``gt_labels``.  Each tensor has shape ``(num_gt,)``.  When
                ``None`` the behaviour is identical to the base head.
        """
        if gt_semantic_weights is None:
            return super().forward_train(
                x, img_metas, proposal_list, gt_bboxes, gt_labels,
                gt_bboxes_ignore, gt_masks)

        if not self.with_bbox:
            return dict()

        num_imgs = len(img_metas)
        if len(gt_semantic_weights) != num_imgs:
            raise ValueError(
                'gt_semantic_weights must have one entry per image: '
                f'got {len(gt_semantic_weights)} for {num_imgs} images')
        for i in range(num_imgs):
            if gt_semantic_weights[i].shape[0] != gt_bboxes[i].shape[0]:
                raise ValueError(
                    'gt_semantic_weights[%d] length %d does not match '
                    'gt_bboxes length %d'
                    % (i, gt_semantic_weights[i].shape[0],
                       gt_bboxes[i].shape[0]))

        # assign gts and sample proposals (identical to the base head)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])

            if gt_bboxes[i].numel() == 0:
                sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                    (0, gt_bboxes[0].size(-1))).zero_()
            else:
                sampling_result.pos_gt_bboxes = \
                    gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

            sampling_results.append(sampling_result)

        losses = dict()
        bbox_results = self._bbox_forward_train_semantic_weighted(
            x, sampling_results, gt_bboxes, gt_labels, img_metas,
            gt_semantic_weights)
        losses.update(bbox_results['loss_bbox'])
        return losses

    def _bbox_forward_train_semantic_weighted(self, x, sampling_results,
                                              gt_bboxes, gt_labels, img_metas,
                                              gt_semantic_weights):
        """Box forward + loss, scaling positive ROI ``label_weights`` by the
        semantic reliability weight of the pseudo GT each positive ROI was
        assigned to."""
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        # concat=False so we can edit each image's positive label_weights
        # before concatenation, using its own pos_assigned_gt_inds mapping.
        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list = \
            self.bbox_head.get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg,
                concat=False)

        for i, sampling_result in enumerate(sampling_results):
            num_pos = sampling_result.pos_bboxes.size(0)
            if num_pos == 0:
                continue
            assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            sem_w = gt_semantic_weights[i].to(
                device=label_weights_list[i].device,
                dtype=label_weights_list[i].dtype)
            positive_weights = sem_w[assigned_gt_inds]
            # Only positive ROIs (first num_pos rows) are scaled; negative ROI
            # label weights (the tail) are left at 1.
            label_weights_list[i][:num_pos] = \
                label_weights_list[i][:num_pos] * positive_weights

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        labels, label_weights, bbox_targets,
                                        bbox_weights)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
