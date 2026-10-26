"""MMRotate 0.3.x R50 source detector for DIOR source-free adaptation."""

_base_ = './oriented_rcnn_r50_dior_trainval.py'

data_root = '/myfile/dataset/DIOR/'
data = dict(
    train=dict(
        ann_file=data_root + 'ImageSets/Main/train.txt'))

work_dir = './work_dirs/oriented_rcnn_r50_dior_source_seed42'
