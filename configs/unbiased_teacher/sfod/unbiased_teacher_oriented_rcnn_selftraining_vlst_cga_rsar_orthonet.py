"""VLST arm D (ST + CGA + Proto): label-level CGA ON, feature-level prototype ON.

The proposed configuration. Identical to arm C except the label-level CGA
branch is enabled with the project's canonical veto_soft stack, so arms C and
D differ ONLY in CGA on/off.
"""
import os

# Label-level CGA branch ON (project canonical stack, see run_slrp_ablation.sh)
os.environ['CGA_SCORER'] = 'sarclip'
os.environ['CGA_BACKEND'] = 'sarclip'
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
        vlst_lora_path=(
            '/myfile/mycode/IRAOD-New/work_dirs/'
            'sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth'),
        vlst_detector_dim=1024,  # RotatedShared2FCBBoxHead fc_out_channels
        vlst_vlm_dim=512,  # ViT-B-32 SARCLIP embedding dimension
    ),
)

# Work directory
work_dir = './work_dirs/vlst_cga_baseline_noise_suppression'
