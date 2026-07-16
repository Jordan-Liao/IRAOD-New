_base_ = './ema_config/baseline_oriented_rcnn_ema_rsar_cga.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

data_root = '/home/storageSDA1/liaojr/dataset/RSAR/'

train_img = data_root + 'train/images/'
train_ann = data_root + 'train/annfiles/'

val_img = data_root + 'val/images/'
val_ann = data_root + 'val/annfiles/'

test_img = data_root + 'test/images/'
test_ann = data_root + 'test/annfiles/'

angle_version = 'le90'

samples_per_gpu = 4
workers_per_gpu = 4
total_epoch = 100

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

image_size = (800, 800)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor',
            'flip', 'flip_direction', 'img_norm_cfg'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=workers_per_gpu,
    train=dict(
        type='DOTADataset',
        ann_file=train_ann,
        img_prefix=train_img,
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type='DOTADataset',
        ann_file=val_ann,
        img_prefix=val_img,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='DOTADataset',
        ann_file=test_ann,
        img_prefix=test_img,
        classes=classes,
        pipeline=test_pipeline)
)

evaluation = dict(interval=1, metric='mAP')

optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[67, 92])

runner = dict(type='EpochBasedRunner', max_epochs=total_epoch)

checkpoint_config = dict(interval=1)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])

dist_params = dict(backend='nccl')
log_level = 'INFO'

# Optionally warm-start shared layers from the existing ResNet RSAR baseline.
# Set this to None for training OrthoNet completely from scratch.
load_from = 'baseline/rsar_oriented_rcnn_epoch_12_mmcv_compat.pth'

resume_from = None
workflow = [('train', 1)]

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
