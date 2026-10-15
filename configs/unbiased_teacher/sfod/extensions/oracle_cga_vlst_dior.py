"""DIOR Target-supervised LoRA-CGA+VLST; source detector identity stays fixed."""

_base_ = '../../../../experiments/comparison/dior_recovery/configs/dior_F_strict.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension', 'sfod.extensions.oracle'],
    allow_failed_imports=False)
model = dict(
    type='OracleCGAVLSTStudent',
    cfg=dict(oracle_dataset='DIOR', oracle_adapter=None, oracle_base_weights=None))
