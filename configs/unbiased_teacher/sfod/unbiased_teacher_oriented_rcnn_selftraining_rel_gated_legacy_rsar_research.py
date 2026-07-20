_base_ = './unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1_research.py'

# Reliability-Gated Legacy.
#
# Runs the FULL legacy score-blend (s = 0.7*s_det + 0.3*p_clip[det_label]) but
# ONLY on a reliable disagreement below the high-confidence cap, so it keeps
# legacy's box-removal power (a blended score can drop under the 0.9 admission
# threshold, removing the box from RPN + ROI cls + bbox reg) while avoiding
# legacy's two failure modes: over-pruning the high-confidence [0.95,1.0) band
# (box audit: 47 FP vs 57 TP removed there) and firing on hesitant SARCLIP
# opposition. Non-firing boxes are left as raw detector output (== no-CGA).
#
# Score-only method: NO ROI head / loss / RPN changes, so it uses the standard
# OrientedStandardRoIHead from the CGA research base (semantic_reweight stays
# off). Run with:
#
#   CGA_SCORER=sarclip CGA_BACKEND=sarclip
#   CGA_FILTER_MODE=reliability_gated_legacy
#   CGA_BLEND_DET_WEIGHT=0.7  CGA_SEM_HIGH_THR=0.95  CGA_REL_LEGACY_TAU=0.10
#
# CGA_REL_LEGACY_TAU=0.0 gives the cap-only ablation (blend every disagreement
# with s_det < 0.95, no reliability filter).
model = dict(
    cfg=dict(
        score_thr=0.9,
        weight_l=1.0,
        weight_u=0.3,
        use_bbox_reg=False,
    ),
)
