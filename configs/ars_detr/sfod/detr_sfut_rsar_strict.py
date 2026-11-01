"""DETR group, SF-UT (arXiv 2407.07586) on RSAR, strict. Not method B.

Same mechanics as the R-CNN port SFUTOBB (configs/.../extensions/sfut_rsar.py):
shared pseudo threshold 0.7, every pseudo loss kept including box/angle
regression, teacher EMA 0.9996 after each optimizer step.
"""
_base_ = './detr_st_rsar_strict.py'

model = dict(type='DETRSFUTTeacher', cfg=dict(score_thr=0.7))
# Standard optimizer hook is priority 40 and checkpoint 50.
custom_hooks = [dict(type='DETRAfterOptimizerTeacherHook', priority=45)]
work_dir = './work_dirs/detr_sfut_strict_${corrupt}'
