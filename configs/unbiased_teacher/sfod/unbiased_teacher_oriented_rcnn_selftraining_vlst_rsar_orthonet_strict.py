"""Strict source-free VLST (method E).

Label-level CGA OFF, independent frozen base SARCLIP VLST encoder ON.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

import os

sarclip_pretrained = os.environ.get('SARCLIP_PRETRAINED', '').strip()
if not sarclip_pretrained:
    raise RuntimeError(
        'Strict VLST requires SARCLIP_PRETRAINED to point to the verified '
        'base checkpoint')
sarclip_pretrained = os.path.expanduser(sarclip_pretrained)
if not os.path.isfile(sarclip_pretrained):
    raise FileNotFoundError(
        f'Strict VLST base checkpoint does not exist: {sarclip_pretrained}')
if os.environ.get('SARCLIP_LORA', '').strip():
    raise RuntimeError(
        'Strict VLST forbids SARCLIP_LORA; use the verified base checkpoint')

os.environ['CGA_SCORER'] = 'none'
os.environ['CGA_BACKEND'] = 'none'
os.environ['CGA_FILTER_MODE'] = 'none'
os.environ['CGA_STRICT'] = '1'
os.environ['VLST_BACKEND'] = 'sarclip'

ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthonet.py'

model = dict(
    type='UnbiasedTeacherVLST',
    ema_config=ema_config,
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,
        vlst_enabled=True,
        vlst_loss_weight=0.1,
        vlst_temperature=0.07,
        vlst_prototype_momentum=0.9,
        vlst_text_visual_alpha=0.5,
        vlst_score_thr=None,
        vlst_projection_hidden=256,
        vlst_strict=True,
        vlst_pretrained=sarclip_pretrained,
        vlst_cache_dir=os.environ.get(
            'SARCLIP_CACHE_DIR', os.path.dirname(sarclip_pretrained)),
        vlst_lora_path=None,
        vlst_detector_dim=1024,
        vlst_vlm_dim=512,
    ),
)

work_dir = './work_dirs/vlst_strict_${corrupt}'
