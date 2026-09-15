"""Independent LPLD OBB port; official algorithm commit ebdc813805870ec20de91ecfb03705c24e4d51cf."""

import torch
from mmdet.models.builder import DETECTORS
from mmrotate.core.bbox import rbbox_overlaps

from .lpld_losses import lpld_candidate_mask, lpld_loss, select_teacher_decoded_boxes
from .proposal_teacher import ProposalAlignedTeacher, map_shared_geometry


@DETECTORS.register_module()
class LPLDOBB(ProposalAlignedTeacher):
    epoch_teacher_momentum = 0.75

    def proposal_loss(self, proposals, hpl, teacher_roi, student_roi, weak_meta, strong_meta):
        if proposals.numel() == 0 or hpl.numel() == 0:
            zero = student_roi['cls_score'].sum() * 0.0
            return {'loss_lpld': zero, 'lpld_kept': zero.detach()}
        with torch.no_grad():
            teacher = getattr(self.ema_model, 'module', self.ema_model)
            head = teacher.roi_head.bbox_head
            decoded = head.bbox_coder.decode(
                proposals, teacher_roi['bbox_pred'], max_shape=weak_meta['img_shape'])
            selected = select_teacher_decoded_boxes(proposals, decoded, teacher_roi['cls_score'])
            selected = map_shared_geometry(selected, weak_meta, strong_meta)
            overlaps = rbbox_overlaps(selected, hpl)
            mask = lpld_candidate_mask(teacher_roi['cls_score'], overlaps)
        loss = lpld_loss(
            student_roi['cls_score'], teacher_roi['cls_score'],
            student_roi['preclassifier'], teacher_roi['preclassifier'], mask)
        return {'loss_lpld': loss, 'lpld_kept': mask.float().sum()}
