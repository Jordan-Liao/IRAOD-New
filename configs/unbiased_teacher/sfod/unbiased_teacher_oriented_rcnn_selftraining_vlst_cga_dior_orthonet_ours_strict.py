"""OURS (本文方法 strict): optical-CLIP CGA + VLST on DIOR-C, strict source-free.

Method knobs are those of origin/sogc-orthonet-phase1 @ 8b42687
``unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_dior_orthonet.py``
(CLIP RN50x64 for CGA and VLST, veto_soft, veto thr 0.7, VLST settings).
Protocol is method F's (dior_bf/configs/dior_F_strict.py on the RSAR strict
chain): train-only DIOR source, image-only val, weight_l=0, no LoRA, no target
GT. Versus F only the VLM differs (CLIP RN50x64 instead of SARCLIP ViT-B-32).
Runs in IRAOD-New-strict-af @ 0f98a48 + experiments/comparison/
ours_strict_vlst_model.patch (adds ``vlst_model``); see protocol.md.
"""
_base_ = './unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py'

import os

# mmcv executes this file before its _base_, and the base sets no env, so
# these values are what CGA/VLST read at model build.
os.environ['CGA_SCORER'] = 'clip'
os.environ['CGA_BACKEND'] = 'clip'
os.environ['CLIP_MODEL'] = 'RN50x64'
os.environ['VLST_CLIP_MODEL'] = 'RN50x64'
os.environ['CGA_STRICT'] = '1'
os.environ['CGA_FILTER_MODE'] = 'veto_soft'
os.environ['CGA_DROP_SCORE'] = '0.0'
os.environ['CGA_FILTER_LOG_EVERY'] = '500'
os.environ['CGA_VETO_PRED_THR'] = '0.7'
os.environ['CGA_VETO_LABEL_THR'] = '0.1'
os.environ['CGA_PROTECT_DET_SCORE'] = '0.9'
os.environ['CGA_BLEND_DET_WEIGHT'] = '0.7'
os.environ['VLST_BACKEND'] = 'clip'

# vlst_strict requires an explicit checkpoint file. clip.load('RN50x64') reads
# this same file through ~/.cache/clip (the launcher symlinks it there).
clip_rn50x64 = os.path.join(os.environ['CLIP_DOWNLOAD_ROOT'], 'RN50x64.pt')
if not os.path.isfile(clip_rn50x64):
    raise FileNotFoundError(f'CLIP RN50x64 checkpoint missing: {clip_rn50x64}')

classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
ema_config = (
    '/home/zechuan/iraod_artifacts/dior_source/'
    'seed42-orthonet-ddp4-gpu4567-spg16-gbs64/reproducibility/'
    'resolved_config.py')
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        classes=classes,
        unlabeled_epoch_size=5863,
        unlabeled_subset_seed=42))
evaluation = None

model = dict(
    type='UnbiasedTeacherVLST',
    ema_config=ema_config,
    roi_head=dict(bbox_head=dict(num_classes=20)),
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
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
        vlst_strict=True,
        vlst_model='RN50x64',
        vlst_pretrained=clip_rn50x64,
        vlst_cache_dir=os.path.dirname(clip_rn50x64),
        vlst_lora_path=None,
        vlst_detector_dim=1024,
        # OpenAI CLIP RN50x64 embedding dim (SARCLIP ViT-B-32 is 512).
        vlst_vlm_dim=1024,
    ),
)
load_from = None
work_dir = './work_dirs/ours_strict_${corrupt}'
