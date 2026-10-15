"""Strict source-free SARCLIP-CGA (method D).

Label-level CGA ON, VLST OFF, frozen base SARCLIP encoder.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

import os

sarclip_pretrained = os.environ.get('SARCLIP_PRETRAINED', '').strip()
if not sarclip_pretrained:
    raise RuntimeError(
        'Strict CGA requires SARCLIP_PRETRAINED to point to the verified '
        'base checkpoint')
sarclip_pretrained = os.path.expanduser(sarclip_pretrained)
if not os.path.isfile(sarclip_pretrained):
    raise FileNotFoundError(
        f'Strict CGA base checkpoint does not exist: {sarclip_pretrained}')
if os.environ.get('SARCLIP_LORA', '').strip():
    raise RuntimeError(
        'Strict CGA forbids SARCLIP_LORA; use the verified base checkpoint')

os.environ.pop('VLST_BACKEND', None)
os.environ['CGA_SCORER'] = 'sarclip'
os.environ['CGA_BACKEND'] = 'sarclip'
os.environ['CGA_STRICT'] = '1'
os.environ['CGA_FILTER_MODE'] = 'veto_soft'
os.environ['CGA_DROP_SCORE'] = '0.0'
os.environ['CGA_FILTER_LOG_EVERY'] = '500'
os.environ['CGA_VETO_PRED_THR'] = '0.7'
os.environ['CGA_VETO_LABEL_THR'] = '0.1'
os.environ['CGA_PROTECT_DET_SCORE'] = '0.9'
os.environ['CGA_BLEND_DET_WEIGHT'] = '0.7'

work_dir = './work_dirs/st_cga_strict_${corrupt}'
