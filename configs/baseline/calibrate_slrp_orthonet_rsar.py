"""Calibrate SLRP against the real detection loss, everything else frozen.

Why the detection loss and not a feature-space objective
-------------------------------------------------------
A feature-invariance objective was tried first (``tools/analysis/calibrate_slrp.py``)
and it worked on its own terms -- held-out invariance down 11.5%, the gate 83x
more active on corrupt input than clean -- yet it *cost* recall: clean 0.8526 ->
0.7904 and chaff 0.7744 -> 0.7123, while only the two AM stripe cases improved.
A control run confirmed the pipeline was sound (zero-init SLRP reproduces the
baseline exactly), so the loss itself was the problem. Two reasons:

1. Cosine distance to the clean reference averages over every position, and
   background dominates, whereas detection only cares about the few positions
   holding targets.
2. Interference destroys information; a multiplicative gate can suppress an
   interference-dominated position but cannot restore it. Driving corrupt
   features towards clean ones asks for something unreachable, and the optimizer
   settles on a small global perturbation, which a frozen detection head is
   extremely sensitive to.

Training the same 1200 parameters against the real detection loss removes the
objective mismatch: the gradient is the metric. It is still source-free -- source
images, source labels, synthetic interference whose parameters are held out from
the seven fixed evaluation specs (``tools/dataset/random_interference.py``). No
target-domain data is touched.

``p=0.5`` keeps clean images in every batch; without them the gate learns to fire
unconditionally and the clean-input cost reappears.
"""

_base_ = './oriented_rcnn_orthonet_rsar.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

angle_version = 'le90'
image_size = (800, 800)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# Interference is injected right after loading, so it acts in the 8-bit
# amplitude domain before resize and normalization -- where it physically occurs.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomSARInterference', p=0.5, seed=1234),
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

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline))

model = dict(
    backbone=dict(
        slrp_cfg=dict(
            enabled=True,
            stages=(-1,),
            window=7,
            hidden_channels=16,
            gamma=0.5,
            sparse_multiplier=1.0)))

# Only SLRP trains, so a much higher LR than the 0.005 used for the full network
# is appropriate; AdamW because 1200 parameters with a bounded output do not need
# SGD's implicit regularization.
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-3,
    weight_decay=0.0)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=5.0, norm_type=2))

lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.01,
    min_lr_ratio=0.05)

total_epoch = 1
runner = dict(type='EpochBasedRunner', max_epochs=total_epoch)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric='mAP')

custom_hooks = [
    dict(type='FreezeExceptSLRPHook', trainable_markers=('slrp',), strict=True)
]

load_from = '/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth'
work_dir = 'work_dirs/calibrate_slrp_orthonet_rsar'
