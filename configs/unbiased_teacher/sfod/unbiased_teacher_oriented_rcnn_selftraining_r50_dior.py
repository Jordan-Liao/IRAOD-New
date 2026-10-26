"""MMRotate 0.3.x plain SFOD/ST run for DIOR-R.

The student and EMA teacher are initialized from the native 0.3.x clean
Oriented R-CNN-R50 checkpoint produced by
``configs/baseline/oriented_rcnn_r50_dior_source.py``.  This file keeps the
repository's existing UnbiasedTeacher implementation, but removes its old
relative paths and the obsolete ``baseline/baseline.pth`` default.
"""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining.py'

custom_imports = dict(imports=['sfod'], allow_failed_imports=False)

classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)

data_root = '/myfile/dataset/DIOR/'
ann_subdir = data_root + 'Annotations/Oriented Bounding Boxes'
corrupt = 'gaussian_noise'
source_ckpt = (
    '/myfile/mycode/IRAOD-New/work_dirs/'
    'oriented_rcnn_r50_dior_source_seed42/epoch_12.pth')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        ann_file=data_root + 'ImageSets/Main/train.txt',
        ann_file_u=data_root + 'ImageSets/Main/val.txt',
        ann_subdir=ann_subdir,
        img_prefix=data_root + 'JPEGImages-trainval',
        img_prefix_u=data_root + 'Corruption/JPEGImages-${corrupt}',
        classes=classes,
        version='le90'),
    val=dict(
        samples_per_gpu=1,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_subdir=ann_subdir,
        img_prefix=data_root + 'Corruption/JPEGImages-${corrupt}',
        classes=classes,
        version='le90'),
    test=dict(
        samples_per_gpu=1,
        ann_file=data_root + 'ImageSets/Main/test.txt',
        ann_subdir=ann_subdir,
        img_prefix=data_root + 'Corruption/JPEGImages-${corrupt}',
        classes=classes,
        version='le90'))

ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_dior.py'
model = dict(
    ema_config=ema_config,
    ema_ckpt=source_ckpt,
    cfg=dict(
        weight_l=0,
        weight_u=1,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False))

optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[1])
runner = dict(type='SemiEpochBasedRunner', max_epochs=1)
evaluation = dict(interval=1, metric='mAP', only_ema=True)
checkpoint_config = dict(interval=1)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
load_from = source_ckpt
resume_from = None
work_dir = './work_dirs/st_dior_gaussian_noise_r50_034'
