"""EMA teacher architecture matching ``oriented_rcnn_orthonet_dior.py``.

The CGA detector type is retained because VLST uses its box-flattening helper
to crop frozen-VLM instance features.  ``CGA_SCORER=none`` keeps label-level
CGA inactive, so this remains the ST + feature-level VLST experiment.
"""

_base_ = './baseline_oriented_rcnn_ema_dior_cga.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

model = dict(
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
        init_cfg=None))
