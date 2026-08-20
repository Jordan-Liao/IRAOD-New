"""Placement ablation: SLRP at C4/C5 instead of post-maxpool.

The depth profile (``work_dirs/depth_profile_400``) predicts this arm performs
close to the no-SLRP baseline, because the interference signature it needs has
decayed to ~2% of its input-domain strength by C4. Confirming that prediction is
the point: it turns "we put the module where the signal is" from an assertion
into a measurement.

``load_from`` must be a checkpoint calibrated with the *same* placement, i.e.
``tools/analysis/calibrate_slrp.py --config`` pointing at
``configs/baseline/oriented_rcnn_slrp_deep_orthonet_rsar.py``.
"""

_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

source_checkpoint = 'work_dirs/slrp_calibrated_deep_v1/slrp_calibrated.pth'
ema_config = (
    './configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_rsar_slrp_deep_orthonet.py')

load_from = source_checkpoint

model = dict(
    ema_config=ema_config,
    ema_ckpt=source_checkpoint,
    backbone=dict(
        slrp_cfg=dict(
            enabled=True,
            stages=(2, 3),
            window=7,
            hidden_channels=16,
            gamma=0.5,
            sparse_multiplier=1.0)))

optimizer = dict(
    paramwise_cfg=dict(
        custom_keys={'ortho_vector': dict(decay_mult=0.0)}))

work_dir = 'work_dirs/unbiased_teacher_slrp_deep_rsar_orthonet'
