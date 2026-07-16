_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1_research.py'

# SARCLIP Semantic Reliability Reweighting (SRW).
#
# Detector confidence alone decides pseudo-box admission (raw s_det >=
# model.cfg.score_thr).  SARCLIP never removes or rescales an admitted box; it
# only produces a per-pseudo-GT semantic reliability weight that scales the ROI
# POSITIVE classification loss.  Run with:
#
#   CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=semantic_reweight
#   SARCLIP_LORA=<...>/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth
#
# The reweighting ramp (low/high/lambda) is read by sfod/cga.py from
# CGA_SEM_LOW_THR / CGA_SEM_HIGH_THR / CGA_SEM_LAMBDA; the cfg values below seed
# those env vars when they are not already exported, and keep this config
# self-documenting.

model = dict(
    cfg=dict(
        # Admission uses the raw detector score; keep the stable recipe's 0.9.
        score_thr=0.9,
        weight_l=1.0,
        weight_u=0.3,
        use_bbox_reg=False,
        semantic_reweight=True,
        semantic_low_thr=0.90,
        semantic_high_thr=0.95,
        semantic_lambda=0.50,
    ),
    roi_head=dict(
        type='SemanticWeightedOrientedStandardRoIHead',
    ),
)
