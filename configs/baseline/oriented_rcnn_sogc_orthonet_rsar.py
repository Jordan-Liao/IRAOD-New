"""RSAR OrthoNet baseline with phase-one SOGC enabled.

Before training or inference, set ``load_from`` to a checkpoint produced by
``tools/analysis/collect_sogc_source_stats.py``. A plain OrthoNet checkpoint
does not contain finalized source statistics and cannot be used directly with
``strict_stats=True``.
"""

_base_ = './oriented_rcnn_orthonet_rsar.py'

load_from = None

model = dict(
    backbone=dict(
        sogc_cfg=dict(
            enabled=True,
            stages=(2, 3),  # zero-based: C4 and C5
            block_selector='last',
            mode='calibrate',
            gamma=0.10,
            reduction=16,
            eps=1e-5,
            z_clip=5.0,
            strict_stats=True)))
