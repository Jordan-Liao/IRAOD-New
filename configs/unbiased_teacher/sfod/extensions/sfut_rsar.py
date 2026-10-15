"""Paper-defined SF-UT, shared HPL>=.7 rather than original>.8; not method B."""

_base_ = './lpld_rsar.py'

model = dict(type='SFUTOBB')
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    # Standard optimizer is40 and checkpoint is50.
    dict(type='AfterOptimizerTeacherHook', priority=45),
]
