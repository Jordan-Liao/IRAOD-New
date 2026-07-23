"""Source-free target adaptation using only StripNet + the existing UT flow.

CGA is disabled by the launcher environment.  SOGC, robust projection and
Strip Head are absent from this configuration.  Both student and EMA teacher
start from the same fully supervised StripNet RSAR source checkpoint.
"""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

source_checkpoint = 'work_dirs/oriented_rcnn_stripnet_rsar/stripnet_rsar_source.pth'
ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_stripnet.py'

load_from = source_checkpoint

# One adaptation epoch is exactly one deterministic pass over corrupted-val.
# Without this override SemiDOTADataset inherits the 78,837-image clean-source
# length and repeats the 8,467-image target set about nine times.
data = dict(
    train=dict(
        unlabeled_epoch_size=8467,
        unlabeled_subset_seed=42))

model = dict(
    ema_config=ema_config,
    ema_ckpt=source_checkpoint,
    cfg=dict(
        weight_l=0.0,
        weight_u=1.0,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,
        semantic_reweight=False,
        dynamic_threshold=False),
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
        num_outs=5),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_head=dict(type='RotatedShared2FCBBoxHead')))

work_dir = 'work_dirs/unbiased_teacher_oriented_rcnn_stripnet_rsar'
