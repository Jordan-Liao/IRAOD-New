"""PETS-OBB: Periodically Exchange Teacher-Student (ICCV 2023) on Oriented R-CNN.

Not method B. Three models: Student (trained), Dynamic Teacher (``ema_model``,
EMA of the Student, evaluated at the end) and Static Teacher (fixed within a
period, exchanged with the Student at each period boundary). Pseudo labels come
from the rotated Teacher Consensus of both teachers. Port notes:
experiments/comparison/pets_obb_port.md.
"""
from __future__ import annotations

import copy

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook
from mmdet.models.builder import DETECTORS
from mmdet.utils import get_root_logger

from .pets_dynamics import exchange_iterations, exchange_weights_, pets_consensus
from .rotated_unbiased_teacher import UnbiasedTeacher


@DETECTORS.register_module()
class PETSUnbiasedTeacher(UnbiasedTeacher):
    def __init__(self, *args, **kwargs):
        cfg = kwargs.get('cfg', dict()) or {}
        super().__init__(*args, **kwargs)
        # Same source weights as the Dynamic Teacher (trainer.py:165-174). Kept
        # off the module tree like ema_model: not optimized, not checkpointed.
        object.__setattr__(self, 'static_teacher', copy.deepcopy(self.ema_model))
        self.pets_epochs = int(cfg.get('pets_epochs', 10))
        self.pets_warmup_epochs = int(cfg.get('pets_warmup_epochs', 2))
        self.pets_period = int(cfg.get('pets_period', 1))
        self.pets_keep_rate = float(cfg.get('pets_keep_rate', 0.999))
        self.pets_pair_iou_thr = float(cfg.get('pets_pair_iou_thr', 0.5))
        self.pets_wbf_iou_thr = float(cfg.get('pets_wbf_iou_thr', 0.55))
        self.pets_skip_box_thr = float(cfg.get('pets_skip_box_thr', 0.01))
        self.momentum = float(cfg.get('pets_warmup_keep_rate', 0.9996))
        self.pets_total_iters = None
        self.pets_exchange_iters = None
        self.pets_num_exchanges = 0

    def cuda(self, device=None):
        self.static_teacher.cuda(device=device)
        return super().cuda(device=device)

    def set_pets_schedule(self, total_iters):
        self.pets_total_iters = int(total_iters)
        self.pets_exchange_iters = exchange_iterations(
            self.pets_total_iters, self.pets_epochs,
            self.pets_warmup_epochs, self.pets_period)
        get_root_logger().info(
            f'[PETS] total_iters={self.pets_total_iters} '
            f'virtual_epochs={self.pets_epochs} warmup={self.pets_warmup_epochs} '
            f'period={self.pets_period} exchange_before_iters={self.pets_exchange_iters}')

    def _pets_before_iter(self):
        """Exchange at a period boundary; then use the post-warm-up keep rate."""
        if self.pets_exchange_iters is None:
            raise RuntimeError('PETSScheduleHook must run before PETS training')
        if self.cur_iter in self.pets_exchange_iters:
            exchange_weights_(self, self.static_teacher)
            self.pets_num_exchanges += 1
            self.momentum = self.pets_keep_rate  # trainer.py:185
            get_root_logger().info(
                f'[PETS] exchange #{self.pets_num_exchanges} static<->student '
                f'before iter {self.cur_iter + 1}/{self.pets_total_iters} '
                f'keep_rate={self.momentum}')

    def forward_train_semi(self, img, img_metas, *args, **kwargs):
        # Before super(): the exchange must precede both the EMA update and
        # the graph of this iteration (trainer.py:182-191 order).
        self._pets_before_iter()
        losses = super().forward_train_semi(img, img_metas, *args, **kwargs)
        losses['pets_exchanges'] = torch.tensor(
            [float(self.pets_num_exchanges)], device=img.device)
        return losses

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True, **kwargs):
        """Teacher Consensus of the Static and Dynamic Teacher (weak view).

        Fused boxes still go through create_pseudo_results, so its degenerate
        rotated-box rejection applies to every PETS pseudo label.
        """
        dynamic = getattr(self.ema_model, 'module', self.ema_model)
        dynamic_results = dynamic.simple_test(img, img_metas, rescale=rescale)
        static_results = self.static_teacher.simple_test(
            img, img_metas, rescale=rescale)
        return [
            pets_consensus(s, d, score_thr=self.score_thr,
                           pair_iou_thr=self.pets_pair_iou_thr,
                           wbf_iou_thr=self.pets_wbf_iou_thr,
                           skip_box_thr=self.pets_skip_box_thr)
            for s, d in zip(static_results, dynamic_results)
        ]


@HOOKS.register_module()
class PETSScheduleHook(Hook):
    """Give the detector the run length so it can place the exchanges."""

    def before_run(self, runner):
        model = runner.model.module if is_module_wrapper(runner.model) else runner.model
        model.set_pets_schedule(runner.max_iters)
