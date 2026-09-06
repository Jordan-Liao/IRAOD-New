"""Observe the existing bbox-head NMS without changing detector inference."""

from unittest.mock import patch

import numpy as np

from experiments.comparison.result_completion import SCHEMA


class AlignedRoICapture:
    """Single-image, single-augmentation capture at the input of fc_cls.

    The original get_bboxes still performs activation, decode and rescale.
    Only its NMS call is observed; no second NMS or box matching is performed.
    """

    def __init__(self, bbox_head, nms_module):
        self.bbox_head = bbox_head
        self.nms_module = nms_module
        self.reset()

    def reset(self):
        self.features = []
        self.detections = []

    def __enter__(self):
        original = self.nms_module.multiclass_nms_rotated

        def observe(multi_bboxes, multi_scores, *args, **kwargs):
            if kwargs.get("return_inds"):
                raise ValueError("Capture expects the standard bbox-head NMS call")
            dets, labels, flat_indices = original(
                multi_bboxes, multi_scores, *args, **kwargs, return_inds=True)
            self.detections.append((
                dets.detach().cpu().numpy(),
                labels.detach().cpu().numpy(),
                flat_indices.detach().cpu().numpy(),
                multi_scores.shape[1] - 1,
                multi_scores.shape[0],
            ))
            return dets, labels

        def capture(_module, inputs):
            self.features.append(inputs[0].detach().float().cpu().numpy().copy())

        self._patch = patch.object(
            self.nms_module, "multiclass_nms_rotated", observe)
        self._patch.start()
        self._hook = self.bbox_head.fc_cls.register_forward_pre_hook(capture)
        return self

    def __exit__(self, *exc):
        self._hook.remove()
        self._patch.stop()

    def aligned(self, image_id, per_class):
        """Consume exact kept indices, in global NMS order, including empties."""
        if len(self.features) != 1 or len(self.detections) != 1:
            raise ValueError("Expected one fc_cls and one bbox NMS call per image")
        raw = self.features[0]
        dets, labels, flat, classes, proposal_count = self.detections[0]
        proposals = flat // classes
        if raw.shape[0] != proposal_count or not np.array_equal(flat % classes, labels):
            raise ValueError("RoI/class mapping is inconsistent with NMS")
        if len(per_class) != classes:
            raise ValueError("Detector class count differs from captured NMS")
        for label, predictions in enumerate(per_class):
            if not np.array_equal(np.asarray(predictions), dets[labels == label]):
                raise ValueError("Captured detections differ from returned predictions")
        return {
            "features": raw[proposals],
            "labels": labels.astype(np.int64),
            "scores": dets[:, 5],
            "boxes": dets[:, :5],
            "proposal_indices": proposals.astype(np.int64),
            "flat_indices": flat.astype(np.int64),
            "detection_indices": np.arange(len(dets), dtype=np.int64),
            "image_ids": np.full(len(dets), image_id),
        }
