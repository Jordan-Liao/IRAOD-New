"""TAM+MT+SSM OBB; one-epoch common budget leaves SSM inactive in final Teacher."""

_base_ = './sfut_rsar.py'

model = dict(
    type='SFYOLOOBB',
    cfg=dict(tam_checkpoint=None, tam_dataset='RSAR', tam_domain=None, tam_seed=None))
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='StudentStabilizationHook', priority='HIGH'),
    dict(type='AfterOptimizerTeacherHook', priority=45),
]
