"""DETR group, method B (ST / Mean Teacher) on RSAR, strict source-free.

Shared DETR-group protocol (fixed before any target run, see
experiments/comparison/detr_portability_audit.md §10): source = public ARS-DETR RSAR
epoch_36, target = image-only corruption val, the R-CNN group's weak/strong
views, 1 GPU x 8 images (global batch 8), 1 epoch, the source's AdamW 1e-4,
score_thr 0.7, classification-only loss on pseudo labels, final EMA.
Every other method config inherits from this file.
"""
_base_ = '../arcsl_detr_r50_rsar.py'

import os

import torchvision.transforms as transforms

custom_imports = dict(
    imports=['sfod', 'ars_detr', 'sfod.detr_sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# mmcv executes a child config before its _base_, so a base must never
# overwrite the CGA env a method config sets; the cell script clears CGA_*.
os.environ.setdefault('CGA_SCORER', 'none')

classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')
source_ckpt = ('/mnt/shared/zechuan/iraod_artifacts/ars_detr/official_rsar/'
               'arcsl_detr_r50_rsar_epoch_36.pth')
target_image_root = (
    '/mnt/shared/zechuan/iraod_data/RSAR/corruptions/${corrupt}/val/images/')
ema_config = './configs/ars_detr/sfod/ema_rsar.py'

# Same views as the R-CNN group's strict configs.
strict_meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg')
strict_pipeline_share = [
    dict(type='LoadImageFromFile'),
    dict(type='RRandomFlip', flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'], version='le90'),
]
strict_pipeline_weak = [
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=strict_meta_keys),
]
strict_pipeline_strong = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply',
         operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply',
         operations=[dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])]),
    dict(type='DTToNumpy'),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=strict_meta_keys),
]

data = dict(
    _delete_=True,
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='StrictSourceFreeDOTADataset',
        img_prefix=target_image_root,
        pipeline_share=strict_pipeline_share,
        pipeline_weak=strict_pipeline_weak,
        pipeline_strong=strict_pipeline_strong,
        classes=classes,
        unlabeled_epoch_size=8467,
        unlabeled_subset_seed=42))

model = dict(
    type='DETRMeanTeacher',
    ema_config=ema_config,
    ema_ckpt=source_ckpt,
    # load_from initialises every weight; skip the torchvision download.
    backbone=dict(init_cfg=None),
    cfg=dict(weight_l=0.0, weight_u=1.0, score_thr=0.7, momentum=0.998))
load_from = source_ckpt

# Optimizer and grad clip = the source's (AdamW 1e-4, inherited); lr policy =
# the R-CNN group's adaptation policy (100-iter linear warm-up, no decay
# inside the single epoch).
lr_config = dict(_delete_=True, policy='step', warmup='linear',
                 warmup_iters=100, warmup_ratio=0.001, step=[1])
runner = dict(_delete_=True, type='SemiEpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1)
evaluation = None
custom_hooks = []
work_dir = './work_dirs/detr_st_strict_${corrupt}'
