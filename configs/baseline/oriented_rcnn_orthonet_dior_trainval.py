"""Fully supervised OrthoNet DIOR-R comparison using trainval labels."""

_base_ = './oriented_rcnn_orthonet_dior.py'

train_ann = '/myfile/dataset/DIOR/ImageSets/Main/trainval.txt'
data = dict(train=dict(ann_file=train_ann))
work_dir = './work_dirs/oriented_rcnn_orthonet_dior_trainval'
