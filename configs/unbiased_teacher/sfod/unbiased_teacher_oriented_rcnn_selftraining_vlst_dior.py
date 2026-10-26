"""VLST adaptation on corrupted DIOR with oriented annotations.

This reuses the DIOR source/target data protocol from
``unbiased_teacher_oriented_rcnn_selftraining_cga.py`` while replacing the
detector with the feature-level Vision-Language Semantic Teacher (VLST) arm.
The label-level CGA scorer is intentionally disabled so this configuration
isolates ST + semantic-prototype supervision, matching the RSAR VLST arm.

The data paths follow the SFOD-RS DIOR layout: source images are clean,
unlabeled and evaluation images come from ``Corruption/JPEGImages-${corrupt}``,
and OBB XML files come from ``Annotations/Oriented Bounding Boxes``.
"""
import os


os.environ['CGA_SCORER'] = 'none'
os.environ['VLST_BACKEND'] = 'clip'
os.environ.setdefault('CLIP_MODEL', 'RN50x64')
os.environ.setdefault('VLST_CLIP_MODEL', os.environ['CLIP_MODEL'])

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

# Must match DIORDataset.cat2label and the clean DIOR baseline checkpoint.
classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)

data_root = '/myfile/dataset/DIOR/'
# ``JPEGImages`` contains non-decodable Windows link stubs in this workspace;
# all ``ImageSets/Main/train`` source images are in this real image directory.
data_root_l = data_root + 'JPEGImages-trainval'
data_root_u = data_root + 'Corruption/JPEGImages-${corrupt}'
ann_file_l = data_root + 'ImageSets/Main/train.txt'
ann_file_u = data_root + 'ImageSets/Main/val.txt'
ann_file_test = data_root + 'ImageSets/Main/test.txt'
ann_subdir = data_root + 'Annotations/Oriented Bounding Boxes'

# Override the inherited class tuples so the dataset label IDs and the VLST
# text-prototype columns have the same order.
data = dict(
    train=dict(
        ann_file=ann_file_l,
        ann_file_u=ann_file_u,
        img_prefix=data_root_l,
        img_prefix_u=data_root_u,
        classes=classes,
        ann_subdir=ann_subdir,
        version='le90'),
    val=dict(
        ann_file=ann_file_test,
        img_prefix=data_root_u,
        classes=classes,
        ann_subdir=ann_subdir,
        version='le90'),
    test=dict(
        ann_file=ann_file_test,
        img_prefix=data_root_u,
        classes=classes,
        ann_subdir=ann_subdir,
        version='le90'),
)

# This must match the fully supervised DIOR source detector.  The inherited
# configuration uses a ResNet, so overriding it here is essential: otherwise
# the source checkpoint and the VLST Student/EMA teacher have different
# backbones and the experiment is not "VLST + OrthoNet".
ema_config = (
    './configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_dior_orthonet.py')
load_from = None

model = dict(
    type='UnbiasedTeacherVLST',
    ema_config=ema_config,
    # The launcher sets both this and ``load_from`` to the clean DIOR source
    # checkpoint, so Student and EMA teacher start from identical weights.
    ema_ckpt=None,
    backbone=dict(
        _delete_=True,
        type='OrthoNet',
        depth=50,
        reduction=16,
        in_channels=3,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    cfg=dict(
        weight_l=0,
        weight_u=1,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,

        # Vision-Language Semantic Teacher.
        vlst_enabled=True,
        vlst_loss_weight=0.1,
        vlst_temperature=0.07,
        vlst_prototype_momentum=0.9,
        vlst_text_visual_alpha=0.5,
        vlst_score_thr=None,
        vlst_projection_hidden=256,
        # The available adapter was trained on six RSAR classes; do not apply
        # it to DIOR's 20 optical classes.
        vlst_lora_path=None,
        vlst_detector_dim=1024,
        # OpenAI CLIP RN50x64 image/text embedding dimension.
        vlst_vlm_dim=1024,
    ),
)

work_dir = './work_dirs/vlst_dior'
