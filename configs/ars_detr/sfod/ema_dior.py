# EMA / static teacher of every DETR-group arm on DIOR-R (see ema_rsar.py).
_base_ = '../arcsl_detr_r50_dior.py'
model = dict(type='ARSDETR_CGA')
