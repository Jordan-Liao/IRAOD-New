"""MMRotate 0.3.x Oriented R-CNN R50-FPN supervised DIOR-R baseline.

This follows the Tang et al. reproduction protocol used in this repository:
the complete trainval split is used for training, test.txt for evaluation,
images stay at 800x800, and the 12-epoch SGD schedule decays at epochs 8/11.
The rotation settings are the standard MMRotate 0.x ``PolyRandomRotate``
defaults commonly used for DIOR-R; the paper does not publish these two
augmentation probabilities/ranges explicitly.
"""

_base_ = './oriented_rcnn_orthonet_dior.py'

train_ann = '/myfile/dataset/DIOR/ImageSets/Main/trainval.txt'

image_size = (800, 800)
angle_version = 'le90'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=2,
    train=dict(
        ann_file=train_ann,
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(samples_per_gpu=1),
    test=dict(samples_per_gpu=1),
)

model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')))

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3.0,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, metric='mAP')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/oriented_rcnn_r50_dior_trainval'
