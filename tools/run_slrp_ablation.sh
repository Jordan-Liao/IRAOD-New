#!/usr/bin/env bash
# Three-arm SLRP placement ablation on one RSAR corruption.
#
#   A  no SLRP            (existing OrthoNet+CGA baseline)
#   B  SLRP @ post-maxpool (the measured placement)
#   C  SLRP @ C4/C5        (the placement the depth profile predicts is dead)
#
# Arm C is the point of the experiment: work_dirs/depth_profile_400 shows the
# interference signature decaying to ~2% of its input-domain strength by C4, so C
# should land near A. If it does, the placement claim is a measurement rather
# than an assertion.
#
# Usage:  bash tools/run_slrp_ablation.sh <corruption> [gpu]
set -euo pipefail

CORRUPT="${1:?usage: run_slrp_ablation.sh <corruption> [gpu]}"
GPU="${2:-0}"
SCORE_THR="${SCORE_THR:-0.82}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/opt/conda/envs/iraod}/lib:${LD_LIBRARY_PATH:-}"

# CGA on, matching the veto_soft setting used for the OrthoNet+CGA baseline.
export CGA_SCORER=sarclip
export CGA_BACKEND=sarclip
export CGA_FILTER_MODE=veto_soft
export CGA_DROP_SCORE=0.0
export CGA_FILTER_LOG_EVERY=500
export CGA_VETO_PRED_THR=0.7
export CGA_VETO_LABEL_THR=0.1
export CGA_PROTECT_DET_SCORE=0.9
export CGA_BLEND_DET_WEIGHT=0.7
export SARCLIP_LORA="${SARCLIP_LORA:-work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth}"

CFG_DIR=configs/unbiased_teacher/sfod
SUFFIX="${CORRUPT}_thr${SCORE_THR//./}"

run_arm () {
  local name="$1" cfg="$2"
  local wd="work_dirs/ablation_${name}_${SUFFIX}"
  echo "=== arm ${name} -> ${wd}"
  CUDA_VISIBLE_DEVICES="$GPU" python train.py "$cfg" \
    --cfg-options corrupt="$CORRUPT" model.cfg.score_thr="$SCORE_THR" \
    work_dir="$wd"
}

run_arm noslrp  "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py"
run_arm maxpool "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_slrp_rsar_orthonet.py"
run_arm deep    "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_slrp_deep_rsar_orthonet.py"

echo
echo "=== final mAP per arm ==="
for name in noslrp maxpool deep; do
  wd="work_dirs/ablation_${name}_${SUFFIX}"
  printf '%-10s ' "$name"
  grep -ho "mAP: [0-9.]*" "$wd"/*.log 2>/dev/null | tail -1 || echo "(no log)"
done
