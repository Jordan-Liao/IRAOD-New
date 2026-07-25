"""Phase-one source-free adaptation with SOGC-OrthoNet and no CGA.

The launch environment must set ``CGA_SCORER=none``, ``CGA_BACKEND=none`` and
``CGA_FILTER_MODE=none``. Both student and EMA teacher start from the same
checkpoint containing finalized source statistics.
"""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

source_checkpoint = (
    'work_dirs/oriented_rcnn_sogc_orthonet_rsar/'
    'source_with_sogc_stats.pth')
ema_config = (
    './configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_rsar_sogc_orthonet.py')

load_from = source_checkpoint

model = dict(
    ema_config=ema_config,
    ema_ckpt=source_checkpoint,
    cfg=dict(
        weight_l=0,
        weight_u=1,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,
        semantic_reweight=False,
        dynamic_threshold=False),
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

work_dir = 'work_dirs/unbiased_teacher_oriented_rcnn_sogc_rsar_orthonet'
