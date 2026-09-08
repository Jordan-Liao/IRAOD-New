"""Strict source-free ST using only a source checkpoint and target images."""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

import torchvision.transforms as transforms

target_image_root = (
    '/myfile/dataset/RSAR/corruptions/${corrupt}/val/images/')

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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='StrictSourceFreeDOTADataset',
        img_prefix=target_image_root,
        pipeline_share=strict_pipeline_share,
        pipeline_weak=strict_pipeline_weak,
        pipeline_strong=strict_pipeline_strong,
        classes=('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'),
        unlabeled_epoch_size=8467,
        unlabeled_subset_seed=42))

evaluation = None

model = dict(
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0))

work_dir = './work_dirs/st_baseline_strict_${corrupt}'
