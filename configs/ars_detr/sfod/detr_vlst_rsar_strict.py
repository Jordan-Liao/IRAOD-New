"""DETR group, method E (VLST only) on RSAR, strict.

VLST settings = the R-CNN group's strict E except the student feature:
last decoder layer query feature (256-d) instead of the 1024-d RoI fc
feature. Frozen base SARCLIP from SARCLIP_PRETRAINED, no LoRA.
"""
_base_ = './detr_st_rsar_strict.py'

import os

vlst_pretrained = os.path.expanduser(
    os.environ.get('SARCLIP_PRETRAINED', '').strip())
model = dict(
    type='DETRVLSTTeacher',
    cfg=dict(
        vlst_loss_weight=0.1,
        vlst_temperature=0.07,
        vlst_prototype_momentum=0.9,
        vlst_text_visual_alpha=0.5,
        vlst_score_thr=None,
        vlst_projection_hidden=256,
        vlst_vlm_dim=512,
        vlst_pretrained=vlst_pretrained,
        vlst_cache_dir=os.environ.get('SARCLIP_CACHE_DIR') or None))
work_dir = './work_dirs/detr_vlst_strict_${corrupt}'
