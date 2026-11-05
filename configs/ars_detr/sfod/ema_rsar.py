# EMA / static teacher of every DETR-group arm on RSAR: the source ARS-DETR
# plus CGA's refine_test (only called when CGA_SCORER is set).
_base_ = '../arcsl_detr_r50_rsar.py'
model = dict(type='ARSDETR_CGA')
