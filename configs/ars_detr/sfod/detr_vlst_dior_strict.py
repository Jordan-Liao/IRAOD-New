"""DETR group, method E on DIOR-R (method part = detr_vlst_rsar_strict)."""
_base_ = './detr_st_dior_strict.py'

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
work_dir = './work_dirs/detr_vlst_dior_strict_${corrupt}'
