"""Strict source-free ST on DIOR-R. Image-only target. No target GT / LoRA."""
_base_ = '../unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py'

data_root = '/mnt/shared/zechuan/iraod_data/DIOR/'
classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
    'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
    'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill')

ema_config = './configs/baseline/oriented_rcnn_orthonet_dior.py'

data = dict(
    _delete_=True,
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='StrictSourceFreeDOTADataset',
        img_prefix=data_root + 'Corruption/${corrupt}/',
        classes=classes,
        unlabeled_epoch_size=None,
        unlabeled_subset_seed=42))

evaluation = None

model = dict(
    ema_config=ema_config,
    cfg=dict(
        strict_source_free=True,
        weight_l=0.0,
        weight_u=1.0,
        use_bbox_reg=False))

load_from = None  # set at launch to DIOR OrthoNet source epoch_last
