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
No DIOR tree on `.67` (`/mnt/shared/zechuan/iraod_data` is RSAR only). 21G oriented DIOR lives on gpu-183. `.67` cannot ssh `.183` (publickey). Staging must relay from the coordinator host.

## B/C/D/F class AP
eval_ema.sh did not tee; no predictions.pkl. Recover by re-eval of existing EMA, not retrain.

## IRG/LPLD/etc
Not ported. N/A until faithful OBB reimplementation.
