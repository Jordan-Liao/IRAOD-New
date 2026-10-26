"""CGA + VLST optical-CLIP adaptation on corrupted DIOR with OrthoRSNet.

This is the OrthoRSNet counterpart of
``unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_dior_orthonet.py``.
Everything outside the backbone -- the DIOR class order, the split roles, the
CGA veto stack, the VLST prototype teacher, and the ``weight_l=1/weight_u=1``
protocol -- is identical, so the two runs differ only in the backbone.

Prerequisite: train the OrthoRSNet DIOR source detector first, because both the
Student and the EMA teacher are initialized from its checkpoint.

    CUDA_VISIBLE_DEVICES=0 python train.py \
      configs/baseline/oriented_rcnn_orthorsnet_dior.py \
      --work-dir work_dirs/orthorsnet_dior_source_seed42 \
      --seed 42 --deterministic \
      --cfg-options data.samples_per_gpu=8 optimizer.lr=0.01 \
                    runner.max_epochs=100 evaluation.interval=1 \
                    checkpoint_config.interval=1

If the source run used the trainval split, use
``configs/baseline/oriented_rcnn_orthorsnet_dior_trainval.py`` instead and keep
``source_ckpt`` below consistent with it.

The backbone block must stay byte-for-byte identical to the one in
``configs/baseline/ema_config/baseline_oriented_rcnn_ema_dior_orthorsnet.py``.
"""
import os


# DIOR is optical imagery: use OpenAI CLIP for both label-level CGA and the
# feature-level VLST prototype teacher.  SARCLIP is reserved for RSAR.
os.environ.setdefault('CGA_SCORER', 'clip')
os.environ.setdefault('CGA_BACKEND', 'clip')
os.environ.setdefault('CLIP_MODEL', 'RN50x64')
os.environ.setdefault('VLST_CLIP_MODEL', os.environ['CLIP_MODEL'])
os.environ.setdefault('CGA_STRICT', '1')
os.environ.setdefault('CGA_FILTER_MODE', 'veto_soft')
os.environ.setdefault('CGA_DROP_SCORE', '0.0')
os.environ.setdefault('CGA_FILTER_LOG_EVERY', '500')
os.environ.setdefault('CGA_VETO_PRED_THR', '0.7')
os.environ.setdefault('CGA_VETO_LABEL_THR', '0.1')
os.environ.setdefault('CGA_PROTECT_DET_SCORE', '0.9')
os.environ.setdefault('CGA_BLEND_DET_WEIGHT', '0.7')
os.environ.setdefault('VLST_BACKEND', 'clip')

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)

# Produced by configs/baseline/oriented_rcnn_orthorsnet_dior_trainval.py.
# Change the epoch if a different one is selected on the source val set.
source_ckpt = (
    '/myfile/mycode/IRAOD-New/work_dirs/'
    'oriented_rcnn_orthorsnet_dior_trainval/epoch_100.pth')

# Keep the absolute DIOR paths from the existing CGA config and only normalize
# class metadata in all splits.
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes),
)

ema_config = (
    './configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_dior_orthorsnet.py')

orthorsnet_backbone = dict(
    _delete_=True,
    type='OrthoRSNet',
    depth=50,
    reduction=16,
    in_channels=3,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    radiometric_cfg=dict(
        enabled=True,
        window=5,
        eps=1e-4,
        gamma=0.5,
        hidden_channels=16,
        learnable_affine=True,
        log_input=False,
        use_variance=True),
    stem_cfg=dict(
        enabled=True,
        orientations=(0, 45, 90, 135),
        dir_kernel_size=7,
        dir_channels=8,
        renormalize=True),
    attn_cfg=dict(
        enabled=True,
        basis_rank=32,
        strip_pool=True,
        strip_kernel=7,
        identity_init=True,
        basis_ridge=1e-3),
    frozen_stages=1,
    norm_eval=True,
    style='pytorch',
    init_cfg=None)

model = dict(
    type='UnbiasedTeacherVLST',
    ema_config=ema_config,
    ema_ckpt=source_ckpt,
    backbone=orthorsnet_backbone,
    cfg=dict(
        weight_l=1,
        weight_u=1,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,
        vlst_enabled=True,
        vlst_loss_weight=0.1,
        vlst_temperature=0.07,
        vlst_prototype_momentum=0.9,
        vlst_text_visual_alpha=0.5,
        vlst_score_thr=None,
        vlst_projection_hidden=256,
        vlst_lora_path=None,
        vlst_detector_dim=1024,
        # OpenAI CLIP RN50x64 image/text embedding dimension.
        vlst_vlm_dim=1024,
    ),
)

load_from = source_ckpt
resume_from = None
work_dir = './work_dirs/cga_vlst_dior_gaussian_noise_orthorsnet'
