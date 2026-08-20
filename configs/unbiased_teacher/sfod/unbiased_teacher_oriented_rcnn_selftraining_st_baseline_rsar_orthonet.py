"""ST baseline: Self-Training without CGA or VLST.

For comparison with VLST to isolate feature-level semantic guidance effect.
"""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension'],
    allow_failed_imports=False)

# Source checkpoint (clean RSAR training)
load_from = '/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth'

ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthonet.py'

model = dict(
    type='UnbiasedTeacher',
    ema_config=ema_config,
    ema_ckpt=load_from,
    cfg=dict(
        weight_l=0,  # SFOD: no labeled data
        weight_u=1,
        debug=False,
        score_thr=0.7,
        use_bbox_reg=False,
    ),
)

# Disable CGA via environment
import os
os.environ['CGA_SCORER'] = 'none'

# Work directory
work_dir = './work_dirs/st_baseline_noise_suppression'
