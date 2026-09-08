"""Explicit labeled-TRAIN VLM oracles, isolated from the strict A–F models.

Import this module explicitly to register the appendix models. Target labels
supervise only the admitted visual adapter, never detector ground truth.
"""

from copy import deepcopy
import os
from pathlib import Path

import mmcv
import torch
from mmdet.models.builder import DETECTORS

from experiments.comparison.oracle_adapters import (
    attach_oracle_adapter, inspect_oracle_adapter)
from sfod.cga import CGA
from sfod.oriented_rcnn_cga import OrientedRCNN_CGA
from sfod.rotated_unbiased_teacher import UnbiasedTeacher
from sfod.unbiased_teacher_vlst import UnbiasedTeacherVLST


def _require_oracle_base(base_weights):
    """Bind the normal strict CGA build to the explicitly admitted base."""
    scorer = os.environ.get('CGA_SCORER', '').strip().lower()
    backend = os.environ.get('CGA_BACKEND', scorer).strip().lower()
    if (scorer != 'sarclip' or backend != 'sarclip'
            or os.environ.get('CGA_STRICT', '0').lower()
            not in ('1', 'true', 'yes')):
        raise ValueError('Oracle requires explicit strict SARCLIP CGA environment')
    if os.environ.get('SARCLIP_LORA', '').strip():
        raise RuntimeError('Oracle forbids ambient SARCLIP_LORA')
    pretrained = os.environ.get('SARCLIP_PRETRAINED', '').strip()
    if (not pretrained or not base_weights
            or Path(pretrained).expanduser().resolve()
            != Path(base_weights).expanduser().resolve()):
        raise ValueError('oracle_base_weights must match SARCLIP_PRETRAINED')
    if os.environ.get('SARCLIP_MODEL', 'ViT-B-32') != 'ViT-B-32':
        raise ValueError('Oracle requires the admitted ViT-B-32 SARCLIP model')


@DETECTORS.register_module()
class OrientedRCNNOracleCGA(OrientedRCNN_CGA):
    """The original teacher architecture with an explicitly attached oracle."""

    def __init__(self, *args, oracle_adapter, oracle_dataset,
                 oracle_base_weights, **kwargs):
        self.oracle_adapter = oracle_adapter
        self.oracle_dataset = oracle_dataset
        self.oracle_base_weights = oracle_base_weights
        super().__init__(*args, **kwargs)
        # Fail even if the first batch contains no detections. CGA is a plain
        # frozen scorer, not an added detector parameter or EMA state entry.
        self.cga, self.exclude_ids = self._build_cga(
            self.roi_head.bbox_head.num_classes)

    def _build_cga(self, num_classes):
        _require_oracle_base(self.oracle_base_weights)
        scorer, exclude_ids = super()._build_cga(num_classes)
        scorer = attach_oracle_adapter(
            scorer, self.oracle_adapter, self.oracle_dataset,
            self.oracle_base_weights)
        return scorer, exclude_ids


class _OracleStudent:
    fairness_group = 'Target-supervised'

    def __init__(self, *args, cfg, ema_config=None, ema_ckpt=None, **kwargs):
        for key in ('oracle_dataset', 'oracle_adapter', 'oracle_base_weights'):
            if not cfg.get(key):
                raise ValueError(f'Oracle requires cfg.{key}')
        if (cfg.get('strict_source_free') is not True
                or cfg.get('weight_l') != 0
                or cfg.get('use_bbox_reg', False) is not False):
            raise ValueError(
                'Oracle requires strict_source_free=True, weight_l=0 and '
                'use_bbox_reg=False')
        if ema_config is None or not ema_ckpt:
            raise ValueError('Oracle requires the original source teacher config/checkpoint')
        _require_oracle_base(cfg['oracle_base_weights'])
        evidence = inspect_oracle_adapter(
            cfg['oracle_adapter'], cfg['oracle_dataset'],
            cfg['oracle_base_weights'])
        teacher_config = (
            mmcv.Config.fromfile(ema_config) if isinstance(ema_config, str)
            else deepcopy(ema_config))
        model = teacher_config['model']
        if model['type'] != 'OrientedRCNN_CGA':
            raise ValueError('Oracle must wrap the original OrientedRCNN_CGA teacher')
        if model['roi_head']['bbox_head']['num_classes'] != len(evidence['classes']):
            raise ValueError('Oracle adapter and source teacher class counts differ')
        model['type'] = 'OrientedRCNNOracleCGA'
        for key in ('oracle_dataset', 'oracle_adapter', 'oracle_base_weights'):
            model[key] = cfg[key]
            setattr(self, key, cfg[key])
        self.oracle_evidence = evidence
        super().__init__(
            *args, cfg=cfg, ema_config=teacher_config, ema_ckpt=ema_ckpt, **kwargs)
        if self.num_classes != len(evidence['classes']):
            raise ValueError('Oracle adapter and student class counts differ')

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True,
                            return_feat=False, return_cga_meta=False):
        # return_feat is a legacy accepted no-op in SemiTwoStageDetector.
        # Keep its return shapes, but never enter its broad RAW fallback.
        teacher = getattr(self.ema_model, 'module', self.ema_model)
        return teacher.simple_test(
            img, img_metas, with_cga=True, rescale=rescale,
            return_cga_meta=return_cga_meta)


@DETECTORS.register_module()
class OracleCGAStudent(_OracleStudent, UnbiasedTeacher):
    """Original LoRA-CGA: image-only detector, target-trained VLM."""


@DETECTORS.register_module()
class OracleCGAVLSTStudent(_OracleStudent, UnbiasedTeacherVLST):
    """Original LoRA-CGA+VLST with two independent frozen oracle encoders."""

    def __init__(self, *args, cfg, **kwargs):
        if (cfg.get('vlst_enabled') is not True
                or cfg.get('vlst_strict') is not True
                or cfg.get('vlst_lora_path') is not None):
            raise ValueError(
                'Oracle VLST requires vlst_enabled=True, vlst_strict=True '
                'and vlst_lora_path=None')
        pretrained = cfg.get('vlst_pretrained')
        base = cfg.get('oracle_base_weights')
        if (not pretrained or not base
                or Path(pretrained).expanduser().resolve()
                != Path(base).expanduser().resolve()):
            raise ValueError('vlst_pretrained must match oracle_base_weights')
        if os.environ.get('VLST_BACKEND', 'sarclip').strip().lower() != 'sarclip':
            raise ValueError('Oracle VLST requires the SARCLIP backend')
        super().__init__(*args, cfg=cfg, **kwargs)

    def _init_vlst_text_prototypes(self):
        if self.vlst_teacher.text_prototypes is not None:
            return
        _require_oracle_base(self.oracle_base_weights)
        # Visual LoRA leaves the frozen text encoder unchanged. Build the
        # existing inference classifier first, then load the explicit adapter.
        vlm = CGA(
            class_names=list(self.oracle_evidence['classes']),
            backend='sarclip', model='ViT-B-32',
            pretrained=self.oracle_base_weights,
            cache_dir=self.vlst_cache_dir or str(
                Path(self.oracle_base_weights).expanduser().parent),
            precision=os.environ.get('SARCLIP_PRECISION', 'fp32'),
            templates=(os.environ.get('CGA_TEMPLATES') or
                       'A SAR image of a {};This SAR patch shows a {}').split(';'),
            strict=True,
        )
        vlm = attach_oracle_adapter(
            vlm, self.oracle_adapter, self.oracle_dataset,
            self.oracle_base_weights)
        self._vlst_vlm = vlm
        self.vlst_teacher.set_text_prototypes(
            torch.as_tensor(vlm.text_prototype_matrix()).float())
