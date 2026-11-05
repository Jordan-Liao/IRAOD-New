"""DETR group, PETS on RSAR, strict. Official PETS constants as in the R-CNN
port (configs/.../pets_rsar_orthonet_strict.py on feat/pets-obb); exchanges
at the proportional virtual-epoch boundaries of the 1-epoch run.
"""
_base_ = './detr_st_rsar_strict.py'

custom_hooks = [dict(type='PETSScheduleHook')]
model = dict(
    type='DETRPETSTeacher',
    cfg=dict(
        score_thr=0.5,                 # SFDA.BBOX_THRESHOLD
        pets_pair_iou_thr=0.5,         # SFDA.PSEUDO_THRESHOLD
        pets_wbf_iou_thr=0.55,         # weighted_boxes_fusion iou_thr default
        pets_skip_box_thr=0.01,        # filter_pseudo_label skip_box_thr
        pets_warmup_keep_rate=0.9996,  # SFDA.EMA_KEEP_RATE_dynamic
        pets_keep_rate=0.999,          # set at the first exchange
        pets_epochs=10,                # SOLVER.EPOCHS -> virtual epochs
        pets_warmup_epochs=2,          # SOLVER.WARM_UP_EPOCH
        pets_period=1))                # SOLVER.Period
work_dir = './work_dirs/detr_pets_strict_${corrupt}'
