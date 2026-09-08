"""DIOR Target-supervised appendix, requiring a DIOR-specific completed adapter."""

_base_ = '../../../../experiments/comparison/dior_recovery/configs/dior_D_strict.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension', 'sfod.extensions.oracle'],
    allow_failed_imports=False)
model = dict(
    type='OracleCGAStudent',
    cfg=dict(oracle_dataset='DIOR', oracle_adapter=None, oracle_base_weights=None))
