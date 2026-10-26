"""CGA + VLST source-free adaptation on corrupted DIOR.

The detector stays ResNet-50 so it matches the native MMRotate 0.3.x clean
``epoch_29.pth`` checkpoint.  CGA provides label-level pseudo-label filtering;
VLST adds feature-level semantic prototype supervision.
"""
import os


# DIOR is optical imagery, so both CGA and VLST use the optical OpenAI CLIP
# backend.  SARCLIP is reserved for the RSAR configurations.
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
# Feature-level VLST uses the same optical CLIP family.  No RSAR LoRA is
# applied to DIOR's 20 optical classes.
os.environ.setdefault('VLST_BACKEND', 'clip')

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)

# Match the class-index order stored in the clean R50 checkpoint and DIORDataset.
classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill',
)

# The base config stores class tuples separately in each dataset split, so
# override all three nested values as well as the top-level tuple.
data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes),
)

model = dict(
    type='UnbiasedTeacherVLST',
    ema_ckpt=(
        '/myfile/mycode/IRAOD-New/work_dirs/'
        'oriented_rcnn_r50_dior_tang034_seed42_36ep/epoch_29.pth'),
    cfg=dict(
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

# Initialize both student and EMA teacher from the clean-domain checkpoint.
load_from=(
    '/myfile/mycode/IRAOD-New/work_dirs/'
    'oriented_rcnn_r50_dior_tang034_seed42_36ep/epoch_29.pth')
work_dir = './work_dirs/cga_vlst_dior'
