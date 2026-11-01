"""Paper-defined SF-UT OBB, arXiv:2407.07586 Sec3.2/Eq3; not the frozen-teacher code variant."""

from mmdet.models.builder import DETECTORS

from .proposal_teacher import ProposalAlignedTeacher


@DETECTORS.register_module()
class SFUTOBB(ProposalAlignedTeacher):
    iteration_teacher_momentum = 0.9996

    def auxiliary_losses(self, teacher, teacher_features, student_features,
                         all_proposals, pseudo_boxes, img_metas_unlabeled, img_metas_unlabeled_1):
        # SF-UT has no graph, low-confidence distillation, or extra projection head.
        return {}
