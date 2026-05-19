# --- 必须项：确保注册 sfod 里的项目侧数据集增强 ---
custom_imports = dict(imports=['sfod','mmrotate.datasets.pipelines'], allow_failed_imports=False)#

import torchvision.transforms as transforms

# ------------------ 基本训练超参 ------------------
gpu = 1
score = 0.7
samples_per_gpu = 2
total_epoch = 12
test_interval = 1
save_interval = 1

# RSAR 六类
classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

# ------------------ 数据路径（DOTA/RSAR 目录结构） ------------------
# 修改为你的真实目录，例如：'/home/storageSDA1/liaojr/dataset/RSAR/'
data_root = '/home/storageSDA1/liaojr/dataset/RSAR/'

train_img = data_root + 'train/images/'
train_ann = data_root + 'train/annfiles/'
val_img   = data_root + 'val/images/'
val_ann   = data_root + 'val/annfiles/'
test_img  = data_root + 'test/images/'
test_ann  = data_root + 'test/annfiles/'

# 角度表示
angle_version = 'le90'

# ------------------ 预处理/增强流水线 ------------------
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)

image_size = (800, 800)

# 监督样本（有标注）
sup_pipeline = [
    dict(type='LoadImageFromFile'),
    # 用 RLoadAnnotations（老版本 mmrotate 提供）直接读取旋转标注
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal', 'vertical', 'diagonal'],
         version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename','ori_filename','ori_shape','img_shape','pad_shape',
                    'scale_factor','flip','flip_direction','img_norm_cfg')),
]

# 无监督样本，共享几何变换分支（先读取 + 轻量几何）
unsup_pipeline_share = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RRandomFlip',
         flip_ratio=[0.25, 0.25, 0.25],
         direction=['horizontal','vertical','diagonal'],
         version=angle_version),
]

# 无监督样本的弱增强（送 teacher）
unsup_pipeline_weak = [
    dict(type='RResize', img_scale=image_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

# 无监督样本的强增强（送 student）
unsup_pipeline_strong = [
    dict(type='DTToPILImage'),
    dict(type='DTRandomApply',
         operations=[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    dict(type='DTRandomGrayscale', p=0.2),
    dict(type='DTRandomApply',
         operations=[dict(type='DTGaussianBlur', rad_range=[0.1, 2.0])]),
    dict(type='DTToNumpy'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_labels'],
         meta_keys=('filename','ori_filename','ori_shape','img_shape','pad_shape',
                    'scale_factor','flip','flip_direction','img_norm_cfg')),
]

# 测试/验证
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
        ])
]

# ------------------ 数据字典 ------------------
data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=2,
    train=dict(
        type='SemiDOTADataset',
        ann_file=train_ann,             # 直接给 annfiles 目录
        ann_file_u=val_ann,             # 这里用 val 作为"无标注"源（可按需换）
        pipeline=sup_pipeline,
        pipeline_u_share=unsup_pipeline_share,
        pipeline_u=unsup_pipeline_weak,
        pipeline_u_1=unsup_pipeline_strong,
        img_prefix=train_img,
        img_prefix_u=val_img,
        classes=classes,
    ),
    val=dict(
        type='DOTADataset',
        ann_file=test_ann,
        img_prefix=test_img,
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type='DOTADataset',
        ann_file=test_ann,
        img_prefix=test_img,
        classes=classes,
        pipeline=test_pipeline),
)

# ------------------ 评测 ------------------
evaluation = dict(interval=test_interval, metric='mAP')

# ------------------ 优化器 / 训练策略 ------------------
learning_rate = 0.02 * samples_per_gpu * gpu / 32
optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.001,
    step=[total_epoch]
)
runner = dict(type='SemiEpochBasedRunner', max_epochs=total_epoch)

checkpoint_config = dict(interval=save_interval)
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook')]
)

custom_hooks = [dict(type='SetEpochInfoHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None

# ------------------ EMA / teacher（如暂无合适权重可设为 None） ------------------
load_from = f'/home/storageSDA1/liaojr/RSAR/work_dirs/oriented-rcnn-le90_r50_fpn_1x_rsar/epoch_12.pth'
ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga.py'
workflow = [('train', 1)]

# ------------------ 模型 ------------------
model = dict(
    type='UnbiasedTeacher',
    ema_config=ema_config,
    ema_ckpt=load_from,
    cfg=dict(
        weight_l=0,           # SFOD: 仅用无监督损失
        weight_u=1,
        debug=False,
        score_thr=score,
        use_bbox_reg=False,
    ),
    backbone=dict(
        type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True, style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048],
              out_channels=256, num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256, feat_channels=256, version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder', angle_range=angle_version,
            target_means=[0., 0., 0., 0., 0., 0.],
            target_stds=[1., 1., 1., 1., 0.5, 0.5]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)
    ),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
            out_channels=256, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256, fc_out_channels=1024, roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder', angle_range=angle_version,
                norm_factor=None, edge_swap=True, proj_xy=True,
                target_means=(0., 0., 0., 0., 0.),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                          min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                         neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=0, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=2000, max_per_img=2000,
                          nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                          min_pos_iou=0.5, match_low_quality=False,
                          iou_calculator=dict(type='RBboxOverlaps2D'),
                          ignore_iof_thr=-1),
            sampler=dict(type='RRandomSampler', num=512, pos_fraction=0.25,
                         neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1, debug=False)
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=2000, max_per_img=2000,
                 nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0),
        rcnn=dict(nms_pre=2000, min_bbox_size=0, score_thr=0.05,
                  nms=dict(iou_thr=0.1), max_per_img=2000)
    )
)
