"""EMA teacher structure for the StripNet-S RSAR experiment."""

_base_ = './baseline_oriented_rcnn_ema_rsar.py'

model = dict(
    type='OrientedRCNN',
    backbone=dict(
        _delete_=True,
        type='StripNet',
        embed_dims=[64, 128, 320, 512],
        k1s=[1, 1, 1, 1],
        k2s=[19, 19, 19, 19],
        depths=[2, 2, 4, 2],
        drop_rate=0.1,
        drop_path_rate=0.15,
        init_cfg=None),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
