"""Strict source-free Original CLIP-CGA (method C).

Uses the repository CLIP backend and RN50x64 aerial-template settings.
Does not construct SARCLIP, LoRA, or VLST.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

import os

for key in (
        'SARCLIP_LORA',
        'SARCLIP_PRETRAINED',
        'SARCLIP_DIR',
        'SARCLIP_MODEL',
        'SARCLIP_CACHE_DIR',
        'SARCLIP_PRECISION',
        'VLST_BACKEND'):
    os.environ.pop(key, None)

os.environ['CGA_SCORER'] = 'clip'
os.environ['CGA_BACKEND'] = 'clip'
os.environ['CGA_STRICT'] = '1'
os.environ['CGA_CLIP_MODEL'] = 'RN50x64'
os.environ['CGA_TEMPLATES'] = 'an aerial image of a {}'
os.environ['CGA_FILTER_MODE'] = 'legacy'

work_dir = './work_dirs/clip_cga_strict_${corrupt}'
