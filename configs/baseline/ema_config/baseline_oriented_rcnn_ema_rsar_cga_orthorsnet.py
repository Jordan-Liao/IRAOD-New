"""EMA teacher architecture matching ``oriented_rcnn_orthorsnet_rsar.py``.

The backbone block must stay byte-for-byte identical to the student's, or the
EMA update will refuse the state dict.
"""

_base_ = './baseline_oriented_rcnn_ema_rsar_cga.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

model = dict(
    backbone=dict(
        _delete_=True,
        type='OrthoRSNet',
        depth=50,
        reduction=16,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # Spelled out in full, not relying on the class defaults, so that a
        # literal comparison against the student config stays exact.
        radiometric_cfg=dict(
            enabled=True,
            window=5,
            eps=1e-4,
            gamma=0.5,
            hidden_channels=16,
            learnable_affine=True,
            log_input=False,
            use_variance=True),
        stem_cfg=dict(
            enabled=True,
            orientations=(0, 45, 90, 135),
            dir_kernel_size=7,
            dir_channels=8,
            renormalize=True),
        attn_cfg=dict(
            enabled=True,
            basis_rank=32,
            strip_pool=True,
            strip_kernel=7,
            identity_init=True,
            basis_ridge=1e-3),
        frozen_stages=1,
        norm_eval=True,
        style='pytorch',
        init_cfg=None))
