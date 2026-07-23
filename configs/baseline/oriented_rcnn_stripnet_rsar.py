"""Fully supervised RSAR source training with a StripNet-S backbone.

Everything except the backbone and FPN input channels follows the existing
OrthoNet-Oriented R-CNN RSAR baseline.  Per user request, StripNet is trained
from random initialization (no ImageNet checkpoint).
"""

_base_ = './oriented_rcnn_orthonet_rsar.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

load_from = None

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

work_dir = 'work_dirs/oriented_rcnn_stripnet_rsar'
