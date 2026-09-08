#!/usr/bin/env bash
# Command reference only. Does not launch or alter the live controller.
set -euo pipefail
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_s424344
RUNNER=/mnt/SSD2_8TB/zechuan/IRAOD-New-finite-c91bc35
INPUTS=$Q/finite_inputs/full-20260907T185024Z

# Explicit single-cell recipes, for an approved reproduction with free GPUs:
# bash "$Q/run_train_1gpu.sh" 4 RSAR chaff 42 B
# bash "$Q/run_train_2gpu.sh" 6,7 29806 DIOR brightness 43 F
# bash "$Q/run_eval_full.sh" 4 RSAR chaff 42 B
# A is eval-only: bash "$Q/run_eval_full.sh" 4 RSAR clean 42 A
# These use existing artifacts; never overwrite completed evidence to force a rerun.
# Record new reproduction outputs separately before deliberately reproducing a completed cell.

# Full matrix recovery uses the original 130 train / 192 eval lists. Only the
# compute owner may invoke it after confirming the old producer has exited;
# existing canonical GPU workers are adopted, not stopped. Use a NEW run directory.
# PYTHONPATH="$RUNNER" "$PY" -m experiments.comparison.finite_resumer launch \
#   --queue "$Q" --run-dir "$Q/finite_runs/NEW_APPROVED_RECOVERY" \
#   --train-list "$INPUTS/train.list" --eval-list "$INPUTS/eval.list" \
#   --gpus 4,5,6,7 --handoff-confirmed

printf '%s\n' \
  "Command reference only; no jobs launched." \
  "Training: /mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af @0f98a48" \
  "Evaluation: /mnt/SSD2_8TB/zechuan/IRAOD-New-rc331d213 @331d213" \
  "Runner: $RUNNER; Python: $PY; frozen lists: $INPUTS" \
  "Source identities/checkpoints: run_manifest.yaml and checkpoint manifests." \
  "CPU final collection/publication: experiments/comparison/REPORT_CONSUMER.md"
