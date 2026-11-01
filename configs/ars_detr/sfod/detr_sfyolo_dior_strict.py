"""DETR group, SF-YOLO on DIOR-R (method part = detr_sfyolo_rsar_strict)."""
_base_ = './detr_sfut_dior_strict.py'

model = dict(
    type='DETRSFYOLOTeacher',
    cfg=dict(tam_checkpoint=None,
             tam_encoder_weights='/mnt/shared/zechuan/iraod_artifacts/third_party/'
                                 'vgg16_oxford/encoder_conv1_1_to_conv4_1.pt',
             tam_dataset='DIOR', tam_domain='${corrupt}', tam_seed=42))
runner = dict(max_epochs=2)
custom_hooks = [
    dict(type='StudentStabilizationHook', priority='HIGH'),
    dict(type='DETRAfterOptimizerTeacherHook', priority=45),
]
work_dir = './work_dirs/detr_sfyolo_dior_strict_${corrupt}'
