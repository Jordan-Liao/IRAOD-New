_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1.py'

# Teacher uses standard OrientedRCNN, without CGA/CLIP post-processing.
ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar.py'
