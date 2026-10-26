"""Fully supervised OrthoRSNet DIOR-R comparison using trainval labels.

Protocol-matched against ``oriented_rcnn_r50_dior_trainval.py`` (0.6693 mAP)
and ``oriented_rcnn_orthonet_dior_trainval.py`` (0.5771 mAP, from scratch).
This run differs from the OrthoNet one only in the three remote-sensing
modules, and both now start from the same ImageNet ResNet-50 weights.
"""

_base_ = './oriented_rcnn_orthorsnet_dior.py'

train_ann = '/myfile/dataset/DIOR/ImageSets/Main/trainval.txt'
data = dict(train=dict(ann_file=train_ann))
work_dir = './work_dirs/oriented_rcnn_orthorsnet_dior_trainval'
