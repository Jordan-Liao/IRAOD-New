"""PETS (ICCV 2023) mechanics, ported to oriented boxes.

Official code: github.com/RainbowNebula/PETS @ 1e87e45 (ubteacher/engine/
trainer.py, wbf.py). Every HBB op that matches, fuses or suppresses boxes is
replaced by its rotated equivalent (mmcv ``box_iou_rotated``). Deviations are
listed in experiments/comparison/pets_obb_port.md.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from mmcv.ops import box_iou_rotated


def exchange_iterations(total_iters, epochs=10, warmup_epochs=2, period=1):
    """Iterations (0-based, before the step) at which Static Teacher <-> Student.

    Official loop (trainer.py:181-184) swaps at the start of every epoch with
    ``epoch >= WARM_UP_EPOCH and epoch % Period == 0`` over ``EPOCHS`` epochs.
    The 1-epoch budget keeps that proportion: virtual epoch k starts at
    ``total_iters * k // epochs``.
    """
    return sorted({total_iters * k // epochs
                   for k in range(warmup_epochs, epochs) if k % period == 0})


@torch.no_grad()
def exchange_weights_(student, static_teacher):
    """Swap every Static Teacher tensor with the Student's, in place.

    In place so the optimizer and DDP keep pointing at the student tensors.
    The official ``swap_weight`` (trainer.py:566-577) copies through aliased
    ``state_dict`` views and so ends as a one-way Student -> Static copy; this
    is the two-way exchange of paper Eq. (2).
    """
    student_sd = student.state_dict()
    for key, static_value in static_teacher.state_dict().items():
        if key not in student_sd:
            raise KeyError(f'{key} is not found in student model')
        student_value = student_sd[key]
        tmp = student_value.clone()
        student_value.copy_(static_value)
        static_value.copy_(tmp)


def _norm_le90(angle):
    return (angle + math.pi / 2) % math.pi - math.pi / 2


def _align_to(box, ref_angle):
    """Equivalent (cx, cy, w, h, a) of ``box`` whose angle is nearest ref."""
    cx, cy, w, h, a = box
    candidates = [(w, h, a + k * math.pi) for k in (-1, 0, 1)]
    candidates += [(h, w, a + math.pi / 2 + k * math.pi) for k in (-1, 0)]
    w2, h2, a2 = min(candidates, key=lambda c: abs(c[2] - ref_angle))
    return np.array([cx, cy, w2, h2, a2], dtype=np.float64)


def _fuse_cluster(members):
    """Score-weighted box and max score (official get_weighted_box, 'max')."""
    ref_angle = members[0][1][4]
    boxes = np.stack([_align_to(box, ref_angle) for _, box in members])
    scores = np.array([score for score, _ in members], dtype=np.float64)
    fused = (scores[:, None] * boxes).sum(0) / scores.sum()
    fused[4] = _norm_le90(fused[4])
    return fused, scores.max()


def rotated_weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                  iou_thr=0.55, skip_box_thr=0.0):
    """Rotated port of the official WBF call (conf_type='max', weights=None).

    Returns ``(boxes[N, 5], scores[N], labels[N])`` sorted by score.
    """
    by_label = {}
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for box, score, label in zip(boxes, scores, labels):
            box = np.asarray(box, dtype=np.float64)
            # wbf.py:24-26 score skip, wbf.py:65-67 zero-area skip.
            if score < skip_box_thr or not np.isfinite(box).all() \
                    or box[2] * box[3] <= 0:
                continue
            by_label.setdefault(int(label), []).append((float(score), box))

    out_boxes, out_scores, out_labels = [], [], []
    for label, items in by_label.items():
        items.sort(key=lambda item: -item[0])
        clusters, fused = [], []
        for score, box in items:
            index = -1
            if fused:
                ious = box_iou_rotated(
                    torch.tensor(np.stack([f for f, _ in fused]), dtype=torch.float32),
                    torch.tensor(box[None], dtype=torch.float32)).view(-1)
                best = int(torch.argmax(ious))
                if float(ious[best]) > iou_thr:  # wbf.py:146 `<= match_iou` -> no match
                    index = best
            if index == -1:
                clusters.append([(score, box)])
                fused.append((box.copy(), score))
            else:
                clusters[index].append((score, box))
                fused[index] = _fuse_cluster(clusters[index])
        for box, score in fused:
            out_boxes.append(box)
            out_scores.append(score)
            out_labels.append(label)
    if not out_boxes:
        return (np.zeros((0, 5), np.float32), np.zeros((0,), np.float32),
                np.zeros((0,), np.int64))
    order = np.argsort(-np.asarray(out_scores), kind='stable')
    return (np.asarray(out_boxes, np.float32)[order],
            np.asarray(out_scores, np.float32)[order],
            np.asarray(out_labels, np.int64)[order])


def _flatten_thresholded(result, score_thr):
    """Per-class [n, 6] results -> boxes/scores/labels with score > thr."""
    boxes = np.concatenate([r[:, :5] for r in result]).astype(np.float32)
    scores = np.concatenate([r[:, 5] for r in result]).astype(np.float32)
    labels = np.concatenate(
        [np.full(len(r), cls, dtype=np.int64) for cls, r in enumerate(result)])
    keep = scores > score_thr  # trainer.py:220 strict `>`
    return boxes[keep], scores[keep], labels[keep]


def pets_consensus(static_result, dynamic_result, score_thr=0.5,
                   pair_iou_thr=0.5, wbf_iou_thr=0.55, skip_box_thr=0.01):
    """Teacher Consensus for one image (official filter_pseudo_label).

    Inputs and output are per-class lists of ``[n, 6]`` arrays
    ``(cx, cy, w, h, a, score)`` as produced by ``rbbox2result``.
    1. threshold each teacher at ``score > score_thr`` (trainer.py:368-373);
    2. keep every (static, dynamic) pair with rotated IoU > pair_iou_thr,
       class-agnostic, duplicates kept (trainer.py:251-268);
    3. fuse the kept boxes of both teachers with rotated WBF
       (trainer.py:271-272).
    """
    num_classes = len(static_result)
    empty = [np.zeros((0, 6), np.float32) for _ in range(num_classes)]
    b1, s1, l1 = _flatten_thresholded(static_result, score_thr)
    b2, s2, l2 = _flatten_thresholded(dynamic_result, score_thr)
    if len(b1) == 0 or len(b2) == 0:
        return empty
    iou = box_iou_rotated(torch.from_numpy(b1), torch.from_numpy(b2))
    k1, k2 = torch.nonzero(iou > pair_iou_thr, as_tuple=True)
    if len(k1) == 0:
        return empty
    k1, k2 = k1.numpy(), k2.numpy()
    boxes, scores, labels = rotated_weighted_boxes_fusion(
        [b1[k1], b2[k2]], [s1[k1], s2[k2]], [l1[k1], l2[k2]],
        iou_thr=wbf_iou_thr, skip_box_thr=skip_box_thr)
    return [np.concatenate([boxes[labels == c], scores[labels == c, None]],
                           axis=1).astype(np.float32)
            for c in range(num_classes)]
