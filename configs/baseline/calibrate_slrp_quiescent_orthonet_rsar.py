"""SLRP calibration with an explicit clean-input quiescence penalty.

The detection-loss-only variant (``calibrate_slrp_orthonet_rsar.py``) recovered
+14.5 mAP on ``noise_suppression`` and +10.1 on ``am_noise_horizontal`` but cost
2.5 points of clean recall, because clean source samples sit near a loss minimum
after 100 epochs of pretraining and contribute almost no gradient. This config
adds ``loss_slrp_quiescence``, which penalises the gate on the clean samples of
each batch directly.

Weight history -- the penalty trades interference suppression for clean fidelity
and the exchange rate is steep:

- ``w=20`` with ``p=0.3``: clean recall 0.8277 -> 0.8348 (+0.7) but
  noise_suppression 0.6057 -> 0.4813 and am_noise_horizontal 0.3517 -> 0.2771,
  i.e. *below* the no-SLRP baseline of 0.2895. Over-corrected, and it moved two
  variables at once so the cause was not attributable.
- ``w=3`` with ``p=0.5``: one variable changed from the detection-loss-only run.
"""

_base_ = './calibrate_slrp_orthonet_rsar.py'

angle_version = 'le90'
image_size = (800, 800)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# p stays at 0.5, matching the detection-loss-only run: the first attempt changed
# both p and the penalty weight at once and could not attribute the outcome.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomSARInterference', p=0.5, seed=1234),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=image_size),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        # sar_interference must reach the detector for the penalty to apply.
        meta_keys=(
            'filename', 'ori_filename', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor',
            'flip', 'flip_direction', 'img_norm_cfg', 'sar_interference'))
]

data = dict(train=dict(pipeline=train_pipeline))

model = dict(
    type='SLRPCalibrationRCNN',
    quiescence_weight=3.0,
    clean_tag='none',
    require_tag=True)

work_dir = 'work_dirs/calibrate_slrp_quiescent_orthonet_rsar'
