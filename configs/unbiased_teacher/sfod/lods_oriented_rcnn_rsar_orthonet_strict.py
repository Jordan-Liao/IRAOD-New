"""LODS (our reimplementation) on RSAR. Strict source-free, image-only target.

Shared OrthoNet-50 Oriented R-CNN source for student and EMA teacher; shared
budget 1 GPU x 32, lr 0.02, 1 epoch. LODS hyperparameters are the official
defaults (Overlook/train_target.py:103-154); see
experiments/comparison/lods_obb_port.md. Launch must set
model.cfg.sem_ckpt to the SEM trained on the same target images.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

data = dict(
    train=dict(
        # LODS feeds the student SEM(original); the second dataset view is unused.
        pipeline_strong=[dict(type='Collect', keys=[], meta_keys=['filename'])]))

model = dict(
    type='LODSTeacher',
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False,
        momentum=0.999,        # --alpha
        thresh=0.8,            # --thresh
        gw_ins_weight=0.1,     # --gw_ins_weight
        gw_global_weight=0.1,  # --gw_global_weight
        label_weight=1.0,      # --label_weight
        gw_add_rate=0.5,       # --gw_add_rate
        style_add_alpha=0.4,   # official city2foggy value (README:164); see lods_obb_port.md
        gw_global_level=2,     # backbone C4 (official res101 RCNN_base4 = layer3)
        # official teacher proposals: cfg.TEST RPN_POST_NMS_TOP_N=300, RPN_NMS_THRESH=0.7
        # (config.py:191-196); min-size filter is commented out (proposal_layer.py:113)
        teacher_rpn=dict(nms_pre=2000, max_per_img=300,
                         nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        sem_ckpt=None,
    ),
)

# Official param groups (train_target.py:496-512 with cfgs/res101.yml):
# RCNN_cls_score lr x0.1; BIAS_DECAY=False -> no weight decay on biases.
optimizer = dict(
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        custom_keys={
            'roi_head.bbox_head.fc_cls.weight': dict(lr_mult=0.1),
            'roi_head.bbox_head.fc_cls.bias': dict(lr_mult=0.1, decay_mult=0.0),
        }))
