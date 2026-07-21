_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1_research.py'

# Prototype-CGA v2 calibration overlay.
#
# Both screening arms use this same config.  The launcher changes only
# CGA_FILTER_MODE (legacy vs prototype_legacy_v2).  In particular, target
# adaptation and evaluation both use corrupted-val; corrupted-test is never
# selected by the training workflow.
data = dict(
    train=dict(
        ann_file_u='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles/',
        img_prefix_u=(
            '/home/storageSDA1/liaojr/dataset/RSAR/'
            'corruptions/${corrupt}/val/images/'),
        unlabeled_epoch_size=8467,
        unlabeled_subset_seed=20260714),
    val=dict(
        ann_file='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles/',
        img_prefix=(
            '/home/storageSDA1/liaojr/dataset/RSAR/'
            'corruptions/${corrupt}/val/images/')),
    # Defensive override: even an accidental test workflow remains on val.
    # No dataset entry in this experiment resolves to corrupted-test.
    test=dict(
        ann_file='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles/',
        img_prefix=(
            '/home/storageSDA1/liaojr/dataset/RSAR/'
            'corruptions/${corrupt}/val/images/')))

model = dict(
    cfg=dict(
        score_thr=0.9,
        # Repository-standard source-free setting: source supervised loss off.
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False,
        semantic_reweight=False,
    ),
)
