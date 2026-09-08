"""LPLD official-core OBB port under the shared one-epoch detector budget."""

_base_ = '../unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

model = dict(
    type='LPLDOBB',
    cfg=dict(strict_source_free=True, weight_l=0.0, weight_u=1.0,
             use_bbox_reg=True, score_thr=0.7))

data = dict(samples_per_gpu=32)
optimizer = dict(lr=0.02)

custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='EpochFinalTeacherHook', priority='HIGH'),
]
