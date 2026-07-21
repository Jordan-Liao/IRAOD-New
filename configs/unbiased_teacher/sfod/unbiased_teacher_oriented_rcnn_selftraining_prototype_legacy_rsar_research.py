_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1_research.py'

# Target-Adaptive Prototype-Guided CGA (prototype_legacy).
#
# Same final score handling as plain legacy
#   score_new = 0.7*raw_det_score + 0.3*fused_prob[det_label]   (on disagreement)
# but the class probability is the fusion of the fixed SARCLIP TEXT prototypes
# with online per-class target-domain VISUAL prototypes built (EMA) from
# high-reliability teacher proposals in the frozen SARCLIP feature space.
#
# Score-only method: standard OrientedStandardRoIHead (semantic_reweight off),
# no loss / RPN changes. Run with:
#   CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=prototype_legacy
#   CGA_BLEND_DET_WEIGHT=0.70 CGA_PROTO_BETA=0.50 CGA_PROTO_MOMENTUM=0.95
#   CGA_PROTO_MIN_COUNT=20 CGA_PROTO_CONTEXT_RATIO=0.15 CGA_PROTO_ROTATED_CROP=1
#   CGA_STRICT=1
model = dict(
    cfg=dict(
        score_thr=0.9,
        weight_l=1.0,
        weight_u=0.3,
        use_bbox_reg=False,
        semantic_reweight=False,
    ),
)
