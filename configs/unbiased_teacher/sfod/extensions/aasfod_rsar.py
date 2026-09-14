"""Approved code-defined AASFOD; stage runner binds TSD and compressed durations."""

_base_ = './lpld_rsar.py'

custom_imports = dict(
    imports=['sfod', 'mmdet_extension', 'sfod.extensions.aasfod'],
    allow_failed_imports=False)
model = dict(type='AASFODOBB', cfg=dict(aasfod_stage='alignment', aasfod_ema_interval=1))
data = dict(train=dict(type='AASFODTargetDataset', stage='alignment', tsd_split=None))
runner = dict(_delete_=True, type='SemiIterBasedRunner', max_iters=159)
lr_config = dict(_delete_=True, policy='Fixed', by_epoch=False,
                 warmup='linear', warmup_iters=100, warmup_ratio=0.001)
checkpoint_config = dict(_delete_=True, by_epoch=False, interval=159, save_last=True,
                         max_keep_ckpts=2)
custom_hooks = [dict(type='AASFODTeacherHook', priority=45)]
