"""EMA teacher structure for no-CGA SOGC-OrthoNet adaptation."""

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
        sogc_cfg=dict(
            enabled=True,
            stages=(2, 3),
            block_selector='last',
            mode='calibrate',
            gamma=0.10,
            reduction=16,
            eps=1e-5,
            z_clip=5.0,
            strict_stats=True)))
