#!/usr/bin/env bash
# Two-arm SFOD baseline alignment on one corruption.
#
#   A  no-slrp        baseline SFOD + OrthoNet
#   B  slrp-maxpool   SLRP at the post-maxpool tap, detection-loss calibrated
#
# CGA is disabled (CGA_SCORER=none): the SARCLIP LoRA weights referenced by
# run.txt are absent from this workspace, so the veto_soft configuration cannot
# be reproduced. Disabling CGA also removes it as a confound, leaving the
# backbone module as the only difference between arms.
#
# Usage:  bash tools/run_sfod_arms.sh <corruption> [gpu]
set -uo pipefail

CORRUPT="${1:?usage: run_sfod_arms.sh <corruption> [gpu]}"
GPU="${2:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export CGA_SCORER=none
export CGA_BACKEND=none
export CGA_FILTER_MODE=none

CFG_DIR=configs/unbiased_teacher/sfod
LOG_DIR="work_dirs/arms_${CORRUPT}"
mkdir -p "$LOG_DIR"

run_arm () {
  local name="$1" cfg="$2"
  local wd="work_dirs/arm_${name}_${CORRUPT}"
  local log="${LOG_DIR}/${name}.log"
  echo "=== arm ${name} -> ${wd}"
  timeout 28800 python train.py "$cfg" \
    --cfg-options corrupt="$CORRUPT" work_dir="$wd" > "$log" 2>&1
  local status=$?
  echo "    exit=${status}"
  grep -ao "mAP: [0-9.]*" "$log" | tail -1 || echo "    (no mAP in log)"
}

run_arm noslrp "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py"
run_arm slrp   "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_slrp_rsar_orthonet.py"

echo "ALLDONE"
