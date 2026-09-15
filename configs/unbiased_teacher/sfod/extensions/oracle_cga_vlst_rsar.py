"""Target-supervised appendix: two independent frozen TRAIN-LoRA encoders."""

_base_ = '../unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_rsar_orthonet_strict.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension', 'sfod.extensions.oracle'],
    allow_failed_imports=False)
model = dict(
    type='OracleCGAVLSTStudent',
    cfg=dict(oracle_dataset='RSAR', oracle_adapter=None, oracle_base_weights=None))
