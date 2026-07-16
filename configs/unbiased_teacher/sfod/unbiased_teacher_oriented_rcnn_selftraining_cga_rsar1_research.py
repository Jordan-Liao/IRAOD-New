_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1.py'

data = dict(
    train=dict(
        ann_file_u='/home/storageSDA1/liaojr/dataset/RSAR/train/annfiles/',
        img_prefix_u=(
            '/home/storageSDA1/liaojr/dataset/RSAR/'
            'corruptions/${corrupt}/train/images/'),
        unlabeled_epoch_size=8467,
        unlabeled_subset_seed=20260714),
    val=dict(
        ann_file='/home/storageSDA1/liaojr/dataset/RSAR/val/annfiles/',
        img_prefix=(
            '/home/storageSDA1/liaojr/dataset/RSAR/'
            'corruptions/${corrupt}/val/images/')))
