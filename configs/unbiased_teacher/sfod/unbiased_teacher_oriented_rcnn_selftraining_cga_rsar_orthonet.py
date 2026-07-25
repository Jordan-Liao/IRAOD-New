_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# Clean-source OrthoNet RSAR checkpoint supplied for this workspace.
load_from = '/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth'

ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthonet.py'

model = dict(
    ema_config=ema_config,
    ema_ckpt=load_from,
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
