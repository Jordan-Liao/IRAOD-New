"""Fully supervised OrthoNet source detector on clean DIOR OBB data.

This is the source model for DIOR source-free adaptation.  Its detector
structure is deliberately shared with the DIOR VLST configuration so a source
checkpoint can initialize both the student and the EMA teacher without any
missing backbone keys.
"""

_base_ = './baseline_oriented_rcnn_dior.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

# The inherited historical DIOR config leaves an unused
# ``JPEGImages-${corrupt}`` variable at top level.  ``patch_config`` resolves
# every placeholder before the data override below, so provide a harmless
# value even though this source-only configuration never reads corruption data.
corrupt = 'clean'

classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)

data_root = '/myfile/dataset/DIOR/'
ann_subdir = data_root + 'Annotations/Oriented Bounding Boxes'
# In this project's SFOD split, train.txt is the labeled source domain,
# val.txt is reserved as the unlabeled target-domain split, and test.txt is
# the held-out evaluation set.  Do not merge val.txt into source training.
train_ann = data_root + 'ImageSets/Main/train.txt'
test_ann = data_root + 'ImageSets/Main/test.txt'
train_img = data_root + 'JPEGImages-trainval'
test_img = data_root + 'JPEGImages-test'

angle_version = 'le90'
image_size = (800, 800)
samples_per_gpu = 8

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
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='DIORDataset',
        ann_file=train_ann,
        ann_subdir=ann_subdir,
        img_prefix=train_img,
        classes=classes,
        version=angle_version,
        pipeline=train_pipeline),
    val=dict(
        _delete_=True,
        type='DIORDataset',
        ann_file=test_ann,
        ann_subdir=ann_subdir,
        img_prefix=test_img,
        classes=classes,
        version=angle_version,
        pipeline=test_pipeline),
    test=dict(
        _delete_=True,
        type='DIORDataset',
        ann_file=test_ann,
        ann_subdir=ann_subdir,
        img_prefix=test_img,
        classes=classes,
        version=angle_version,
        pipeline=test_pipeline),
)

# OrthoNet's non-attention parameters are key-for-key identical to ResNet-50
# (23.508 M of 26.038 M parameters; the only extra keys are ``*.attn.*``), so
# the torchvision ResNet-50 weights inflate into it unchanged. Verified in
# tools/analysis/audit_orthonet_attention.py: loading them reports 48 missing
# keys, all of them attention parameters, and zero unexpected keys. Together
# with ``attn_init='identity'`` (which makes a fresh attention an exact
# pass-through) this makes the run a fair ResNet-50 baseline instead of a
# from-scratch one. The 100-epoch schedule is kept so the only variable
# against the previous source run is the initialization.
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    # ``ortho_vector`` only ever enters the forward pass through
    # ``F.normalize``, so its magnitude carries no function and weight decay
    # on it is pure noise on the parameterization. Without this the deeper
    # stages collapse to ||v|| ~ 0.05 (see the SLRP baseline config).
    paramwise_cfg=dict(
        custom_keys={'ortho_vector': dict(decay_mult=0.0)}))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[67, 92])
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='mAP')
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Train the DIOR source model from scratch; do not inherit the obsolete paths
# and ResNet initialization assumptions of the historical baseline config.
load_from = None
resume_from = None
work_dir = './work_dirs/orthonet_dior_source_seed42'

model = dict(
    backbone=dict(
        _delete_=True,
        type='OrthoNet',
        depth=50,
        reduction=16,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        # ``identity`` zero-initializes the attention's output layer and scales
        # its gate by two, so a freshly built backbone computes exactly the
        # ResNet-50 function it was initialized from. ``legacy`` would multiply
        # every residual branch by a random value near 0.5 and destroy the
        # pretrained features at step zero.
        attn_init='identity',
        frozen_stages=1,
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50')))
