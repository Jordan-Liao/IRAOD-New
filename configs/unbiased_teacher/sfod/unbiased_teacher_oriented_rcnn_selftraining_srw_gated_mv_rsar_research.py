_base_ = './unbiased_teacher_oriented_rcnn_selftraining_srw_rsar_research.py'

# Reliability-Gated SRW (multi-view).
#
# Same safe SRW contract, but reliability is built on cross-scale consistency
# of SARCLIP over several crops of each box (tight / 1.5x / 2.0x context):
#   r = 1[all view top1 agree] * mean(p_top1) * (1 - mean H_norm).
# A single unstable AABB crop can no longer trigger a strong downweight -- only
# multi-scale-consistent, confident opposition does. Motivated by RSAR classes
# that are ambiguous in a single AABB crop (ship/aircraft/car/tank look alike;
# bridge/harbor need scene context). Run with:
#
#   CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=reliability_gated_mv
#   CGA_SEM_GATED_LAMBDA=0.80  CGA_SEM_VIEW_RATIOS=0.0,0.25,0.5
#
# Note: ~3x SARCLIP forward passes per box (one per view), so teacher inference
# is slower than single-view SRW.
model = dict(
    cfg=dict(
        semantic_reweight=True,
        semantic_low_thr=0.90,
        semantic_high_thr=0.95,
    ),
)
