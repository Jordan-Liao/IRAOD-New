"""DIOR TAM+MT+SSM; separate two-epoch budget, domain TAM fit seed42."""

_base_ = './sfut_dior.py'

model = dict(
    type='SFYOLOOBB',
    cfg=dict(tam_checkpoint=None, tam_dataset='DIOR', tam_domain=None, tam_seed=42))
runner = dict(max_epochs=2)
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='StudentStabilizationHook', priority='HIGH'),
    dict(type='AfterOptimizerTeacherHook', priority=45),
]
