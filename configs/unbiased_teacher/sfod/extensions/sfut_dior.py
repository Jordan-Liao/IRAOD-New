"""Paper-defined SF-UT with the unchanged DIOR source architecture."""

_base_ = './lpld_dior.py'

model = dict(type='SFUTOBB')
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='AfterOptimizerTeacherHook', priority=45),
]
