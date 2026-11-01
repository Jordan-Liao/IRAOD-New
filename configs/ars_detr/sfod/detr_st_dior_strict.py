"""DETR group, method B on DIOR-R: the RSAR protocol with DIOR data/source.

Target = image-only corruption val split (5,863 images, as the R-CNN group),
source = the DETR-group DIOR-R source (train.txt only, final epoch 72).
"""
_base_ = './detr_st_rsar_strict.py'

classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
           'chimney', 'expressway-service-area', 'expressway-toll-station',
           'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass',
           'ship', 'stadium', 'storagetank', 'tenniscourt', 'trainstation',
           'vehicle', 'windmill')
source_ckpt = ('/mnt/shared/zechuan/iraod_artifacts/ars_detr/dior_source/'
               'seed42-r50-1gpu-spg8/epoch_72.pth')
ema_config = './configs/ars_detr/sfod/ema_dior.py'

data = dict(
    train=dict(
        img_prefix='/home/zechuan/iraod_jobs/dior_splits/${corrupt}/val/',
        classes=classes,
        unlabeled_epoch_size=5863))
model = dict(
    ema_config=ema_config,
    ema_ckpt=source_ckpt,
    bbox_head=dict(num_classes=20))
load_from = source_ckpt
