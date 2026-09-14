"""SLRP at C4/C5, for the placement ablation only.

Not the recommended configuration. See
``oriented_rcnn_slrp_orthonet_rsar.py`` for the measured placement and the depth
profile that chose it.
"""

_base_ = './oriented_rcnn_slrp_orthonet_rsar.py'

model = dict(
    backbone=dict(
        slrp_cfg=dict(
            enabled=True,
            stages=(2, 3),
            window=7,
            hidden_channels=16,
            gamma=0.5,
            sparse_multiplier=1.0)))
