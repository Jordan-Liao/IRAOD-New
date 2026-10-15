_base_ = './baseline_oriented_rcnn_ema_rsar_cga_orthonet.py'

model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))
