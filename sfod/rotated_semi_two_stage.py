# Copyright (c) Hangzhou Hikvision Digital Technology Co., Ltd. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
"""
semi-supervised two stage detector
"""
import os

import torch

from mmrotate.core import rbbox2result
from mmrotate.models.detectors import RotatedTwoStageDetector
from .rotated_semi_base import SemiBaseDetector


def _log_cga_info(message):
    print(message, flush=True)
    try:
        from mmrotate.utils import get_root_logger

        logger = get_root_logger()
        logger.warning(message)
    except Exception:
        pass


class SemiTwoStageDetector(SemiBaseDetector, RotatedTwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 # ema model
                 ema_config=None,
                 ema_ckpt=None,
                 classes=None
                 ):
        SemiBaseDetector.__init__(self, ema_config=ema_config, ema_ckpt=ema_ckpt, classes=classes)
        RotatedTwoStageDetector.__init__(self, backbone=backbone, rpn_head=rpn_head, roi_head=roi_head,
                                  train_cfg=train_cfg, test_cfg=test_cfg, neck=neck, pretrained=pretrained)

    @torch.no_grad()
    def inference_unlabeled(self, img, img_metas, rescale=True,
                            return_feat=False, return_cga_meta=False):
        """Run the EMA teacher on unlabeled images.

        When ``return_cga_meta`` is True (only meaningful with a SARCLIP CGA
        scorer, e.g. ``CGA_FILTER_MODE=semantic_reweight``) this returns
        ``(bbox_results, cga_meta)`` via an explicit return value rather than
        any hidden global state.  ``cga_meta`` is ``None`` when CGA is disabled
        or fell back to a raw pass, and callers must tolerate that.
        """
        ema_model = getattr(self.ema_model, 'module', self.ema_model)
        cga_scorer = os.environ.get("CGA_SCORER", "").strip().lower()
        if not hasattr(self, "_cga_entry_logged"):
            self._cga_entry_logged = True
            _log_cga_info(
                "[CGA] inference_unlabeled "
                f"scorer={cga_scorer or '<unset>'}, "
                f"backend={os.environ.get('CGA_BACKEND', '<unset>')}, "
                f"lora={os.environ.get('SARCLIP_LORA', '<unset>')}, "
                f"ema_model={ema_model.__class__.__module__}.{ema_model.__class__.__name__}"
            )

        cga_meta = None
        if cga_scorer in ("", "none", "false", "0", "raw"):
            bbox_results = ema_model.simple_test(img, img_metas, rescale=rescale)
        elif cga_scorer in ("sarclip", "clip", "openai", "optical", "optical_clip"):
            if not hasattr(self, "_cga_with_cga_logged"):
                self._cga_with_cga_logged = True
                _log_cga_info("[CGA] calling ema_model.simple_test(with_cga=True)")
            try:
                if return_cga_meta:
                    bbox_results, cga_meta = ema_model.simple_test(
                        img, img_metas, with_cga=True,
                        return_cga_meta=True, rescale=rescale
                    )
                else:
                    bbox_results = ema_model.simple_test(
                        img, img_metas, with_cga=True, rescale=rescale
                    )
            except Exception as e:
                if not hasattr(self, "_cga_fallback_count"):
                    self._cga_fallback_count = 0
                self._cga_fallback_count += 1

                if self._cga_fallback_count == 1 or self._cga_fallback_count % 50 == 0:
                    _log_cga_info(
                        f"[CGA][WARN] with_cga=True failed "
                        f"(count={self._cga_fallback_count}), "
                        f"fallback to raw simple_test. err={repr(e)}"
                    )

                cga_meta = None
                bbox_results = ema_model.simple_test(
                    img, img_metas, rescale=rescale
                )
        else:
            bbox_results = ema_model.simple_test(img, img_metas, rescale=rescale)

        if return_cga_meta:
            return bbox_results, cga_meta
        return bbox_results
            
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
