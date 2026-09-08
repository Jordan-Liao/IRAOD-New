"""DIOR TAM+MT+SSM, requiring a distinct fitted TAM for this domain and seed."""

_base_ = './sfut_dior.py'

model = dict(
    type='SFYOLOOBB',
    cfg=dict(tam_checkpoint=None, tam_dataset='DIOR', tam_domain=None, tam_seed=None))
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='StudentStabilizationHook', priority='HIGH'),
    dict(type='AfterOptimizerTeacherHook', priority=45),
]
