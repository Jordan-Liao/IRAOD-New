"""LODS (our reimplementation) on DIOR-R. Strict source-free, image-only target.

Same LODS settings as the RSAR config; DIOR data/source come from the strict
ST DIOR config and the launch overrides in train_d17_seed_1gpu.sh.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_dior_orthonet_strict.py'

data = dict(
    train=dict(
        pipeline_strong=[dict(type='Collect', keys=[], meta_keys=['filename'])]))

model = dict(
    type='LODSTeacher',
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False,
        momentum=0.999,
        thresh=0.8,
        gw_ins_weight=0.1,
        gw_global_weight=0.1,
        label_weight=1.0,
        gw_add_rate=0.5,
        style_add_alpha=0.4,
        gw_global_level=2,
        # official teacher proposals: cfg.TEST RPN_POST_NMS_TOP_N=300, RPN_NMS_THRESH=0.7
        # (config.py:191-196); min-size filter is commented out (proposal_layer.py:113)
        teacher_rpn=dict(nms_pre=2000, max_per_img=300,
                         nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        sem_ckpt=None,
    ),
)

optimizer = dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys={
            'roi_head.bbox_head.fc_cls.weight': dict(lr_mult=0.1),
            'roi_head.bbox_head.fc_cls.bias': dict(lr_mult=0.1, decay_mult=0.0),
        }))
