"""DETR group, PETS on DIOR-R (method part = detr_pets_rsar_strict)."""
_base_ = './detr_st_dior_strict.py'

custom_hooks = [dict(type='PETSScheduleHook')]
model = dict(
    type='DETRPETSTeacher',
    cfg=dict(score_thr=0.5, pets_pair_iou_thr=0.5, pets_wbf_iou_thr=0.55,
             pets_skip_box_thr=0.01, pets_warmup_keep_rate=0.9996,
             pets_keep_rate=0.999, pets_epochs=10, pets_warmup_epochs=2,
             pets_period=1))
work_dir = './work_dirs/detr_pets_dior_strict_${corrupt}'
