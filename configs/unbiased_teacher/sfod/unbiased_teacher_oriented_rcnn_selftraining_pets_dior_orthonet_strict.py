"""PETS-OBB on DIOR-R (our reimplementation). Not method B. Shared OrthoNet-50 source, image-only target.

Same PETS hyperparameters as the RSAR config; see
experiments/comparison/pets_obb_port.md.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_dior_orthonet_strict.py'

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='PETSScheduleHook')]

model = dict(
    type='PETSUnbiasedTeacher',
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False,
        score_thr=0.5,
        pets_pair_iou_thr=0.5,
        pets_wbf_iou_thr=0.55,
        pets_skip_box_thr=0.01,
        pets_warmup_keep_rate=0.9996,
        pets_keep_rate=0.999,
        pets_epochs=10,
        pets_warmup_epochs=2,
        pets_period=1,
    ),
)
