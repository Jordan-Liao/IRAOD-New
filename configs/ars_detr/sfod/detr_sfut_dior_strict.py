"""DETR group, SF-UT on DIOR-R (method part = detr_sfut_rsar_strict)."""
_base_ = './detr_st_dior_strict.py'

model = dict(type='DETRSFUTTeacher', cfg=dict(score_thr=0.7))
custom_hooks = [dict(type='DETRAfterOptimizerTeacherHook', priority=45)]
work_dir = './work_dirs/detr_sfut_dior_strict_${corrupt}'
