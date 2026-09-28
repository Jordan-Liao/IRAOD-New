"""DRU-OBB on RSAR. Not method B. Shared OrthoNet-50 source, image-only target."""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

model = dict(
    type='DRUUnbiasedTeacher',
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False,
        dru_hist_weight=0.5,
        dru_retrain_ratio=1.25,
        dru_momentum_step=0.02,
    ),
)
