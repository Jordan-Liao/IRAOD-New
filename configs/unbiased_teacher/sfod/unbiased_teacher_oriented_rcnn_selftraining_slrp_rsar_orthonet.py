"""Source-free adaptation with SLRP-OrthoNet at the post-maxpool tap.

``load_from`` must point at a checkpoint produced by
``tools/analysis/calibrate_slrp.py``: SLRP has no usable gradient inside this
loop on its own, because the pseudo-labels come from a teacher looking at the
same corrupted image. The calibration stage supplies that signal beforehand,
using source images only.

Student and EMA teacher start from the same calibrated checkpoint.
"""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

# Detection-loss calibration (tools/analysis/calibrate_slrp.py's feature-space
# objective was measured to be worse: it cost 6.2 clean recall points while
# gaining only 1.4 on the high-severity classes).
source_checkpoint = 'work_dirs/calibrate_slrp_orthonet_rsar/epoch_1.pth'
ema_config = (
    './configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_rsar_slrp_orthonet.py')

load_from = source_checkpoint

slrp_cfg = dict(
    enabled=True,
    stages=(-1,),
    window=7,
    hidden_channels=16,
    gamma=0.5,
    sparse_multiplier=1.0)

model = dict(
    ema_config=ema_config,
    ema_ckpt=source_checkpoint,
    backbone=dict(slrp_cfg=slrp_cfg))

# ortho_vector is L2-normalized in the forward pass, so decay on it only shrinks
# its norm without changing the function.
optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={'ortho_vector': dict(decay_mult=0.0)}))

work_dir = 'work_dirs/unbiased_teacher_slrp_rsar_orthonet'
