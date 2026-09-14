# DIOR-R overlay on frozen RSAR strict method. Image-only val. No target GT/LoRA.
_base_ = '/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af/configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet_strict.py'
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
ema_config = '/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/reproducibility/resolved_config.py'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        unlabeled_epoch_size=5863,
        unlabeled_subset_seed=42))
evaluation = None
model = dict(
    ema_config=ema_config,
    roi_head=dict(bbox_head=dict(num_classes=20)),
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False))
load_from = None
