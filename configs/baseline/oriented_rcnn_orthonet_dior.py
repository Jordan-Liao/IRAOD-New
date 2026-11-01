_base_ = './ema_config/baseline_oriented_rcnn_ema_dior.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# Unified OrthoNet-50 + Oriented R-CNN for DIOR-R. Not ResNet-50.
classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')

data_root = '/mnt/shared/zechuan/iraod_data/DIOR/'
angle_version = 'le90'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='DIORDataset',
        ann_file=data_root + 'ImageSets/train.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_prefix=data_root + 'JPEGImages/',
        img_subdir='',
        classes=classes,
        version=angle_version,
        xmltype='obb',
        pipeline=train_pipeline),
    val=dict(
        type='DIORDataset',
        ann_file=data_root + 'ImageSets/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_prefix=data_root + 'JPEGImages/',
        img_subdir='',
        classes=classes,
        version=angle_version,
        xmltype='obb',
        pipeline=test_pipeline),
    test=dict(
        type='DIORDataset',
        ann_file=data_root + 'ImageSets/test.txt',
        ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
        img_prefix=data_root + 'JPEGImages/',
        img_subdir='',
        classes=classes,
        version=angle_version,
        xmltype='obb',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='mAP')

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

optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[67, 92])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1, max_keep_ckpts=2, save_last=True)
workflow = [('train', 1)]
load_from = None
resume_from = None
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
