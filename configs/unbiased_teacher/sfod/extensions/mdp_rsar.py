"""Preprint-guided MDP-OBB; no target GT and no source-detector replacement."""

_base_ = './lpld_rsar.py'

model = dict(type='MDPOBB')

# Sec. IV.B specifies retention0.9 once after each epoch, before checkpointing.
custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='EpochFinalTeacherHook', priority='HIGH'),
]
