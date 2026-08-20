"""Arm B (ST + CGA): label-level CGA ON, feature-level prototype OFF.

The existing CGA method on top of ST, with the project's canonical veto_soft
stack. No VLST module is constructed (type stays UnbiasedTeacher).
"""
import os

# Label-level CGA branch ON (project canonical stack, see run_slrp_ablation.sh)
os.environ['CGA_SCORER'] = 'sarclip'
os.environ['CGA_BACKEND'] = 'sarclip'
os.environ['CGA_FILTER_MODE'] = 'veto_soft'
os.environ['CGA_DROP_SCORE'] = '0.0'
os.environ['CGA_FILTER_LOG_EVERY'] = '500'
os.environ['CGA_VETO_PRED_THR'] = '0.7'
os.environ['CGA_VETO_LABEL_THR'] = '0.1'
os.environ['CGA_PROTECT_DET_SCORE'] = '0.9'
os.environ['CGA_BLEND_DET_WEIGHT'] = '0.7'
os.environ['SARCLIP_LORA'] = (
    '/myfile/mycode/IRAOD-New/work_dirs/'
    'sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth')

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# Work directory
work_dir = './work_dirs/st_cga_noise_suppression'
