"""DETR group, SF-YOLO (TAM + MT + SSM) on RSAR, strict; separate 2-epoch
budget, as the R-CNN port SFYOLOOBB (configs/.../extensions/sfyolo_rsar.py).

The TAM is the R-CNN group's completed domain fit (seed 42, reused for every
detector seed); the cell script passes model.cfg.tam_checkpoint.
"""
_base_ = './detr_sfut_rsar_strict.py'

model = dict(
    type='DETRSFYOLOTeacher',
    cfg=dict(tam_checkpoint=None,
             tam_encoder_weights='/mnt/shared/zechuan/iraod_artifacts/third_party/'
                                 'vgg16_oxford/encoder_conv1_1_to_conv4_1.pt',
             tam_dataset='RSAR', tam_domain='${corrupt}', tam_seed=42))
runner = dict(max_epochs=2)
custom_hooks = [
    dict(type='StudentStabilizationHook', priority='HIGH'),
    dict(type='DETRAfterOptimizerTeacherHook', priority=45),
]
work_dir = './work_dirs/detr_sfyolo_strict_${corrupt}'
