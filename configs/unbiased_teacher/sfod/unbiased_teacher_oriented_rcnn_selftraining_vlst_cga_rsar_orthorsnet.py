"""VLST arm D (ST + CGA + Proto) on RSAR with the OrthoRSNet backbone.

This is the OrthoRSNet counterpart of
``unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_rsar_orthonet.py``.
The CGA stack, the SARCLIP-LoRA adapters, the feature-level prototype teacher
and the source-free (``weight_l=0``, ``weight_u=1``) protocol are identical, so
the two runs differ only in the backbone.

Prerequisite: train the OrthoRSNet RSAR source detector first, because the
Student and the EMA teacher both start from its checkpoint.

    CUDA_VISIBLE_DEVICES=0 python train.py \
      configs/baseline/oriented_rcnn_orthorsnet_rsar.py \
      --work-dir work_dirs/oriented_rcnn_orthorsnet_rsar \
      --seed 42 --deterministic \
      --cfg-options data.samples_per_gpu=4 optimizer.lr=0.005 \
                    runner.max_epochs=100 evaluation.interval=1 \
                    checkpoint_config.interval=1

Then copy or symlink ``epoch_100.pth`` to ``source_ckpt`` below, or override it
at launch with
``--cfg-options load_from=<path> model.ema_ckpt=<path>``.

The backbone block must stay byte-for-byte identical to the one in
``configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthorsnet.py``.
"""
import os

# Label-level CGA branch ON (project canonical stack, see run_slrp_ablation.sh)
os.environ['CGA_SCORER'] = 'sarclip'
os.environ['CGA_BACKEND'] = 'sarclip'
os.environ['CGA_STRICT'] = '1'
os.environ['SARCLIP_PRETRAINED'] = (
    '/myfile/pretrain/SARCLIP/ViT-B-32/vit_b_32_model.safetensors')
os.environ['SARCLIP_CACHE_DIR'] = '/myfile/pretrain/SARCLIP/ViT-B-32'
os.environ['CGA_FILTER_MODE'] = 'veto_soft'
os.environ['CGA_DROP_SCORE'] = '0.0'
os.environ['CGA_FILTER_LOG_EVERY'] = '500'
os.environ['CGA_VETO_PRED_THR'] = '0.7'
os.environ['CGA_VETO_LABEL_THR'] = '0.1'
os.environ['CGA_PROTECT_DET_SCORE'] = '0.9'
os.environ['CGA_BLEND_DET_WEIGHT'] = '0.7'
os.environ['SARCLIP_LORA'] = (
    '/myfile/mycode/IRAOD-New/work_dirs/'
    'sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth')

# The prototype teacher gets its own LoRA-SARCLIP instance. It is separate
# from CGA but intentionally uses the same adapter proved useful on RSAR.
os.environ['VLST_BACKEND'] = 'sarclip'

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# Produced by configs/baseline/oriented_rcnn_orthorsnet_rsar.py.
source_ckpt = '/myfile/pretrain/oriented_rcnn_orthorsnet_rsar_epoch_100.pth'
load_from = source_ckpt

ema_config = (
    './configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_rsar_cga_orthorsnet.py')

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
        weight_l=0,  # SFOD: no labeled data
        weight_u=1,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,

        # Vision-Language Semantic Teacher (VLST)
        vlst_enabled=True,
        vlst_loss_weight=0.1,
        vlst_temperature=0.07,
        vlst_prototype_momentum=0.9,
        vlst_text_visual_alpha=0.5,
        vlst_score_thr=None,  # Inherit from score_thr
        vlst_projection_hidden=256,
        vlst_lora_path=(
            '/myfile/mycode/IRAOD-New/work_dirs/'
            'sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth'),
        vlst_detector_dim=1024,  # RotatedShared2FCBBoxHead fc_out_channels
        vlst_vlm_dim=512,  # ViT-B-32 SARCLIP embedding dimension
    ),
)

# Work directory
work_dir = './work_dirs/vlst_cga_rsar_orthorsnet'
