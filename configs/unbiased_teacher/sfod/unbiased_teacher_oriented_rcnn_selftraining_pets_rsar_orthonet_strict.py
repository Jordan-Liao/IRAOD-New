"""PETS-OBB on RSAR (our reimplementation). Not method B. Shared OrthoNet-50 source, image-only target.

Hyperparameters = official PETS (RainbowNebula/PETS @ 1e87e45); sources and the
1-epoch mapping are in experiments/comparison/pets_obb_port.md.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

custom_hooks = [dict(type='SetEpochInfoHook'), dict(type='PETSScheduleHook')]

model = dict(
    type='PETSUnbiasedTeacher',
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False,
        score_thr=0.5,                 # SFDA.BBOX_THRESHOLD
        pets_pair_iou_thr=0.5,         # SFDA.PSEUDO_THRESHOLD
        pets_wbf_iou_thr=0.55,         # weighted_boxes_fusion iou_thr default
        pets_skip_box_thr=0.01,        # filter_pseudo_label skip_box_thr
        pets_warmup_keep_rate=0.9996,  # SFDA.EMA_KEEP_RATE_dynamic
        pets_keep_rate=0.999,          # set at the first exchange
        pets_epochs=10,                # SOLVER.EPOCHS, as virtual epochs of the 1-epoch run
        pets_warmup_epochs=2,          # SOLVER.WARM_UP_EPOCH
        pets_period=1,                 # SOLVER.Period
    ),
)
