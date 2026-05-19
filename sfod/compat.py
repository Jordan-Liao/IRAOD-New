import torch
from mmcv.ops import nms_rotated


_NMS_DEVICE_PATCHED = False


def _multiclass_nms_rotated_device_safe(multi_bboxes,
                                        multi_scores,
                                        score_thr,
                                        nms,
                                        max_num=-1,
                                        score_factors=None,
                                        return_inds=False):
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = torch.arange(
        num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = scores > score_thr
    if score_factors is not None:
        score_factors = score_factors.to(scores.device)
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        return dets, labels

    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]

    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)
    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return torch.cat([bboxes, scores[:, None]], 1), labels, keep
    return torch.cat([bboxes, scores[:, None]], 1), labels


def patch_mmrotate_multiclass_nms_rotated():
    global _NMS_DEVICE_PATCHED
    if _NMS_DEVICE_PATCHED:
        return

    import mmrotate.core as core
    from mmrotate.core.post_processing import bbox_nms_rotated as nms_module
    from mmrotate.models.roi_heads.bbox_heads import rotated_bbox_head

    core.multiclass_nms_rotated = _multiclass_nms_rotated_device_safe
    nms_module.multiclass_nms_rotated = _multiclass_nms_rotated_device_safe
    rotated_bbox_head.multiclass_nms_rotated = (
        _multiclass_nms_rotated_device_safe)
    _NMS_DEVICE_PATCHED = True
