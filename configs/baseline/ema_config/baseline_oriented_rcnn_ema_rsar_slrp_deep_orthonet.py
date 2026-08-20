"""EMA teacher for the C4/C5 SLRP placement ablation.

Exists only to test the depth-profile prediction that a module placed where the
interference signature has already decayed cannot act on it. See
``configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_slrp_deep_rsar_orthonet.py``.
"""

_base_ = './baseline_oriented_rcnn_ema_rsar.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

model = dict(
    type='OrientedRCNN',
    backbone=dict(
        _delete_=True,
        type='OrthoNet',
        depth=50,
        reduction=16,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=True,
        style='pytorch',
        init_cfg=None,
        slrp_cfg=dict(
            enabled=True,
            stages=(2, 3),
            window=7,
            hidden_channels=16,
            gamma=0.5,
            sparse_multiplier=1.0)))
