"""Target-supervised appendix: explicit TRAIN LoRA-CGA, never a strict A-F alias."""

_base_ = '../unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet_arm_b_strict.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension', 'sfod.extensions.oracle'],
    allow_failed_imports=False)
model = dict(
    type='OracleCGAStudent',
    cfg=dict(oracle_dataset='RSAR', oracle_adapter=None, oracle_base_weights=None))
