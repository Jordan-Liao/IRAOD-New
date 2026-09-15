"""Unchanged20-class DIOR detector/source; only AASFOD adaptation is added."""

_base_ = './aasfod_rsar.py'

classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')
ema_config = './configs/baseline/ema_config/baseline_oriented_rcnn_ema_dior_cga_orthonet.py'
model = dict(ema_config=ema_config, roi_head=dict(bbox_head=dict(num_classes=20)))
data = dict(train=dict(classes=classes, unlabeled_epoch_size=5863))
runner = dict(max_iters=110)
checkpoint_config = dict(interval=110)
