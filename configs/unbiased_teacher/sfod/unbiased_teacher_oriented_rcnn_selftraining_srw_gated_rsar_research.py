_base_ = './unbiased_teacher_oriented_rcnn_selftraining_srw_rsar_research.py'

# Reliability-Gated SRW (single view).
#
# Same safe SRW contract (raw detector score admits pseudo boxes; SARCLIP never
# drops or rescales), but the per-box downweight is gated by SARCLIP's
# reliability r = p_top1 * margin * (1 - H_norm), so a larger lambda only bites
# on stable, confident, low-entropy opposition. Run with:
#
#   CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=reliability_gated
#   CGA_SEM_GATED_LAMBDA=0.80  (reuses CGA_SEM_LOW_THR/HIGH_THR for g(s))
#
# The ROI head is inherited from the SRW config
# (SemanticWeightedOrientedStandardRoIHead); semantic_reweight stays True so the
# per-GT weights are applied to the ROI positive classification loss.
model = dict(
    cfg=dict(
        semantic_reweight=True,
        semantic_low_thr=0.90,
        semantic_high_thr=0.95,
    ),
)
