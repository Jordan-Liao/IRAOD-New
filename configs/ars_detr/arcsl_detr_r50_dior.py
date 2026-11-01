# DETR-group DIOR-R source: ARS-DETR (AR-CSL, no DN), R-50 ImageNet init,
# DIOR `ImageSets/train.txt` only (val.txt corruptions are adaptation images),
# input scale and augmentation of the Oriented R-CNN DIOR source, official
# optimizer (AdamW 1e-4, total batch 8), 72 epochs, final checkpoint = last
# epoch.
_base_ = './arcsl_detr_r50_rsar.py'

classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
           'chimney', 'expressway-service-area', 'expressway-toll-station',
           'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass',
           'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation',
           'vehicle', 'windmill')
angle_version = 'le90'
data_root = '/mnt/shared/zechuan/iraod_data/DIOR/'

model = dict(bbox_head=dict(num_classes=20))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
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
            dict(type='Collect', keys=['img'])
        ])
]
dior_common = dict(
    type='DIORDataset',
    ann_subdir=data_root + 'Annotations/Oriented Bounding Boxes/',
    img_prefix=data_root + 'JPEGImages/',
    img_subdir='JPEGImages',
    classes=classes,
    version=angle_version,
    xmltype='obb')
data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'ImageSets/train.txt',
        pipeline=train_pipeline,
        **dior_common),
    # val.txt is reserved; the only in-training eval is test.txt at the last
    # epoch (interval=72), so it reports the final score and selects nothing.
    val=dict(
        ann_file=data_root + 'ImageSets/test.txt',
        pipeline=test_pipeline,
        **dior_common),
    test=dict(
        ann_file=data_root + 'ImageSets/test.txt',
        pipeline=test_pipeline,
        **dior_common))

# 72 epochs; the official 36e schedule drops lr at 32/36, kept proportional.
evaluation = dict(_delete_=True, interval=72, metric='mAP')
lr_config = dict(step=[64])
runner = dict(max_epochs=72)
checkpoint_config = dict(interval=12)
