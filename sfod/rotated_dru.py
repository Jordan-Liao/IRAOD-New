"""DRU-OBB: Dynamic Retraining-Updating Mean Teacher on Oriented R-CNN.

Not method B. B is UnbiasedTeacher with fixed EMA and no historical student
loss. This detector applies DRU's two mechanisms on the shared OrthoNet-50
source.
"""
from __future__ import annotations

import torch
from mmdet.models.builder import DETECTORS

from .dru_dynamics import (
    adaptive_ema_momentum,
    historical_student_mse,
    should_retrain_student,
    unlabeled_loss_scalar,
)
from .rotated_unbiased_teacher import UnbiasedTeacher


@DETECTORS.register_module()
class DRUUnbiasedTeacher(UnbiasedTeacher):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("cfg", dict()) or {}
        super().__init__(*args, **kwargs)
        self.dru_hist_weight = float(cfg.get("dru_hist_weight", 0.5))
        self.dru_retrain_ratio = float(cfg.get("dru_retrain_ratio", 1.25))
        self.dru_momentum_step = float(cfg.get("dru_momentum_step", 0.02))
        self._dru_base_momentum = float(self.momentum)
        self._dru_ref_loss = None
        self._dru_last_student_loss = None
        self._dru_hist_pred = None
        self.dru_last_retrain_iter = -1
        self._dru_pending_retrain = False

    def _copy_teacher_to_student(self):
        ema = getattr(self.ema_model, "module", self.ema_model)
        student_sd = self.state_dict()
        patched = {}
        for key, value in ema.state_dict().items():
            student_key = key[7:] if key.startswith("module.") else key
            if student_key in student_sd and student_sd[student_key].shape == value.shape:
                patched[student_key] = value.detach().clone()
        if patched:
            student_sd.update(patched)
            self.load_state_dict(student_sd, strict=False)
        self.dru_last_retrain_iter = int(self.cur_iter)
        self._dru_pending_retrain = False

    def _capture_student_cls(self, fn):
        """Run fn while recording the last bbox-head cls scores (unlabeled pass)."""
        captured = {}
        head = getattr(getattr(self, "roi_head", None), "bbox_head", None)
        orig = getattr(head, "forward", None) if head is not None else None
        if orig is None:
            return fn(), None

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            cls = out[0] if isinstance(out, (tuple, list)) else out
            captured["cls"] = cls
            return out

        head.forward = wrapped
        try:
            result = fn()
        finally:
            head.forward = orig
        return result, captured.get("cls")

    def _historical_prediction_loss(self, cls_scores):
        if cls_scores is None or not torch.is_tensor(cls_scores):
            return None
        if cls_scores.dim() < 2 or cls_scores.numel() == 0:
            return None
        current = cls_scores.softmax(dim=-1).mean(dim=0)
        hist = self._dru_hist_pred
        loss = None
        if hist is not None and tuple(hist.shape) == tuple(current.shape):
            loss = self.dru_hist_weight * historical_student_mse(current, hist)
        self._dru_hist_pred = current.detach()
        return loss

    def forward_train_semi(self, *args, **kwargs):
        # Retrain must run before the new graph is built. Copying teacher
        # weights after super() but before backward in-places student params
        # that the current loss still points at.
        if self._dru_pending_retrain:
            self._copy_teacher_to_student()
        if self._dru_last_student_loss is not None and self._dru_ref_loss is not None:
            self.momentum = adaptive_ema_momentum(
                self._dru_base_momentum,
                self._dru_last_student_loss,
                self._dru_ref_loss,
                step=self.dru_momentum_step,
            )
        losses, cls_scores = self._capture_student_cls(
            lambda: super(DRUUnbiasedTeacher, self).forward_train_semi(*args, **kwargs)
        )
        student_loss = unlabeled_loss_scalar(losses)
        if self._dru_ref_loss is None:
            self._dru_ref_loss = student_loss
        else:
            if should_retrain_student(
                student_loss, self._dru_ref_loss, ratio=self.dru_retrain_ratio
            ):
                self._dru_pending_retrain = True
            self._dru_ref_loss = (
                0.9 * float(self._dru_ref_loss) + 0.1 * float(student_loss)
            )
        self._dru_last_student_loss = student_loss

        hist_loss = self._historical_prediction_loss(cls_scores)
        if hist_loss is not None:
            # Key must contain "loss" so MMCV includes it in backward.
            losses["loss_dru_historical"] = hist_loss
        return losses

