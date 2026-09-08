"""Independent tensor implementation of the LPLD paper's distillation algorithm.

Algorithm evidence supplied in the task: official LPLD commit
ebdc813805870ec20de91ecfb03705c24e4d51cf, student_sfda_rcnn.py:145-221.
Implemented from the supplied mathematical specification, not upstream source.

Inputs describe ONE image, in exact teacher RPN proposal order. Teacher weak
and student strong ROI tensors must already be aligned at those same indices.
Decoding, coordinate mapping, rotated IoU, and image averaging belong to callers.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def select_teacher_decoded_boxes(proposals, decoded_boxes, teacher_logits):
    """Select teacher boxes without NMS or matching, returning detached [N, 5].

    ``proposals`` is [N, 5]; ``decoded_boxes`` is already bbox-coder-decoded,
    either [N, 5] or [N, 5*C]; ``teacher_logits`` is [N, C+1], background last.
    A background argmax retains the original proposal, not a decoded box.
    """
    labels = teacher_logits.argmax(dim=-1)
    background = labels == teacher_logits.shape[-1] - 1
    if decoded_boxes.shape[-1] == 5:
        selected = decoded_boxes
    else:
        boxes = decoded_boxes.reshape(
            proposals.shape[0], teacher_logits.shape[-1] - 1, 5)
        # Background has no regression slot; its selected value is replaced below.
        class_indices = labels.masked_fill(background, 0)
        selected = boxes[torch.arange(labels.shape[0], device=labels.device),
                         class_indices]
    return torch.where(background[:, None], proposals, selected)


@torch.no_grad()
def lpld_candidate_mask(teacher_logits, iou_to_hpl):
    """Return a detached [N] bool mask, preserving proposal order.

    ``iou_to_hpl`` is [N, H]: caller-computed rotated IoU between selected
    teacher boxes and HPL boxes in the same coordinates. H=0 keeps nothing.
    Teacher logits are [N, C+1], with background last.
    """
    if iou_to_hpl.shape[1] == 0:
        return teacher_logits.new_zeros((teacher_logits.shape[0],),
                                       dtype=torch.bool)
    background_prob = teacher_logits.softmax(dim=-1)[:, -1]
    foreground_confidence = teacher_logits[:, :-1].softmax(dim=-1).amax(dim=-1)
    return ((iou_to_hpl.amax(dim=-1) <= 0.4)
            & (background_prob <= 0.99)
            & (foreground_confidence >= 0.9))


def lpld_loss(student_logits, teacher_logits, student_features, teacher_features,
              candidate_mask):
    """Return scalar LPLD for an image and its precomputed candidate mask.

    Logits are [N, C+1], features are pre-classifier ROI vectors [N, D].
    The target is foreground softmax plus background mass 1e-10, WITHOUT
    renormalization. Detached weights are 1-cos(teacher, student), using
    torch's standard cosine epsilon. Reduction is sum / kept count / 10.
    Only student logits receive gradients, including when no proposals remain.
    """
    if not candidate_mask.any():
        return student_logits.sum() * 0.0
    with torch.no_grad():
        foreground = teacher_logits[candidate_mask, :-1].softmax(dim=-1)
        target = torch.cat((foreground, foreground.new_full(
            (foreground.shape[0], 1), 1e-10)), dim=-1)
        weights = 1.0 - F.cosine_similarity(
            teacher_features[candidate_mask], student_features[candidate_mask],
            dim=-1)
    log_probs = student_logits[candidate_mask].log_softmax(dim=-1)
    per_proposal = F.kl_div(log_probs, target, reduction="none").sum(dim=-1)
    return (weights * per_proposal).sum() / per_proposal.shape[0] / 10.0
