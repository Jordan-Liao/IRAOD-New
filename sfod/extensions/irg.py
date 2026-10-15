"""Independent code-defined IRG OBB port; original algorithm commit82c84894."""

from mmdet.models.builder import DETECTORS

from .irg_losses import IRGLosses
from .proposal_teacher import ProposalAlignedTeacher


@DETECTORS.register_module()
class IRGOBB(ProposalAlignedTeacher):
    epoch_teacher_momentum = 0.9

    def __init__(self, *args, cfg, **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        self.irg = IRGLosses(self.roi_head.bbox_head.fc_cls.in_features)

    def proposal_loss(self, proposals, hpl, teacher_roi, student_roi, weak_meta, strong_meta):
        if proposals.numel() == 0:
            zero = (student_roi['cls_score'].sum()
                    + sum(parameter.sum() for parameter in self.irg.parameters())) * 0.0
            losses = {name: zero for name in (
                'loss_irg_direct', 'loss_irg_student_graph',
                'loss_irg_teacher_graph', 'loss_irg_contrast')}
        else:
            teacher = getattr(self.ema_model, 'module', self.ema_model)
            losses = self.irg(
                student_roi['preclassifier'], teacher_roi['preclassifier'],
                student_roi['cls_score'], teacher_roi['cls_score'],
                self.roi_head.bbox_head.fc_cls, teacher.roi_head.bbox_head.fc_cls)
        losses['irg_proposals'] = student_roi['cls_score'].new_tensor(float(len(proposals)))
        return losses
