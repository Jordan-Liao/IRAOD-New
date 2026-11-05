"""Strict source-free ST on DIOR-R. Image-only target. No target GT / LoRA."""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

import torchvision.transforms as transforms

data_root = '/mnt/shared/zechuan/iraod_data/DIOR/'
classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')

ema_config = './configs/baseline/oriented_rcnn_orthonet_dior.py'

strict_pipeline_share = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version='le90'),
]

strict_pipeline_weak = [
    dict(type='RResize', img_scale=(800, 800)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')),
]

strict_pipeline_strong = [
    dict(type='DTToPILImage'),
    dict(
        type='DTRandomApply',
        operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
        p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(
        type='DTRandomApply',
        operations=[dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])]),
    dict(type='DTToNumpy'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape', 'img_shape',
            'pad_shape', 'scale_factor', 'flip', 'flip_direction',
            'img_norm_cfg')),
]

data = dict(
    _delete_=True,
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='StrictSourceFreeDOTADataset',
        img_prefix=data_root + 'Corruption/${corrupt}/',
        pipeline_share=strict_pipeline_share,
        pipeline_weak=strict_pipeline_weak,
        pipeline_strong=strict_pipeline_strong,
        classes=classes,
        unlabeled_epoch_size=None,
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

load_from = None  # set at launch to DIOR OrthoNet source epoch_last
