# Failures / stalls

## missed_source_to_AF_continuation
Source ended 2026-09-05T20:36:10+08; A–F launch 23:59 (~3.3h idle). Agent-only `tail --pid` did not continue. Fix: remote tmux wait-for.

## E_single_gpu_spg32_oom
2026-09-06T00:09:58 exit 1 CUDA OOM iter2. Formal E is DDP2 spg16 same global 32 / LR 0.02.

## F_preflight_stopper
Intended 8-step confirm ran full 265. Accepted as formal F after identity+eval; do not retrain.

## F_logvarpad / epoch35 overlay
NON_RESULT. Forbidden for formal.

## DIOR-R
Resolved: DIOR staging and source + 18 A–F evaluations completed on `.67`. The previous blocker was missing data and lack of direct `.67` → `.183` SSH access. Source clean val/test = 0.410/0.295. This historical staging blocker is no longer current.

## B/C/D/F class AP
eval_ema.sh did not tee; no predictions.pkl. Recover by re-eval of existing EMA, not retrain.

## IRG/LPLD/etc
Not ported. N/A until faithful OBB reimplementation.

## DIOR qualitative export limitations
288 same-image detection visualizations and 33 RoI directories are complete. The RoI exporter saves pre-NMS features and post-NMS predictions without a row mapping; do not assign prediction labels directly to feature rows. Original `run_roi.sh` skips on any existing index.json, so a partial output needs explicit inspection and isolation before recovery. No export was rerun during local integration.
