"""DRU-OBB: Dynamic Retraining-Updating Mean Teacher on Oriented R-CNN.

Not method B. B is UnbiasedTeacher with fixed EMA and no historical student
loss. This detector applies DRU's two mechanisms on the shared OrthoNet-50
source.
"""
from __future__ import annotations

from collections import deque

import torch
from mmdet.models.builder import DETECTORS

from .dru_dynamics import (
    adaptive_ema_momentum,
    historical_student_consistency,
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
        self._dru_hist_scores = deque(maxlen=int(cfg.get("dru_hist_len", 4)))
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
        losses = super().forward_train_semi(*args, **kwargs)
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

        if self._dru_hist_scores:
            hist_loss = historical_student_consistency(
                [student_loss], [self._dru_hist_scores[-1]]
            )
            # Logged only; name has no "loss" so it is not part of backward.
            losses["dru_historical"] = torch.tensor(
                self.dru_hist_weight * hist_loss
            )
        self._dru_hist_scores.append(float(student_loss))
        return losses

