"""VLST arm C (ST + Proto): feature-level semantic guidance only.

Label-level CGA is OFF, the SARCLIP prototype teacher is ON, so this arm
isolates the feature-level contribution against the ST baseline.
"""
import os

# Arm C disables the label-level CGA branch. The prototype teacher builds its
# own SARCLIP text encoder (VLST_BACKEND), so it is unaffected by this.
os.environ['CGA_SCORER'] = 'none'
os.environ['VLST_BACKEND'] = 'sarclip'

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# Source checkpoint (clean RSAR training)
load_from = '/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth'

ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthonet.py'

model = dict(
    type='UnbiasedTeacherVLST',
    ema_config=ema_config,
    ema_ckpt=load_from,
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
        vlst_detector_dim=1024,  # RotatedShared2FCBBoxHead fc_out_channels
        vlst_vlm_dim=512,  # ViT-B-32 SARCLIP embedding dimension
    ),
)

# Work directory
work_dir = './work_dirs/vlst_baseline_noise_suppression'
