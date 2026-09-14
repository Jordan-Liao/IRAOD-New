#!/usr/bin/env bash
# Three-arm SFOD ablation with CGA enabled, using the locally trained LoRA.
#
#   A  no-slrp        baseline SFOD + OrthoNet + CGA
#   B  slrp-maxpool   SLRP calibrated, post-maxpool tap + CGA
#
# Run after tools/build_patches_and_train_lora.sh has finished and
# work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth exists.
#
# Usage:  bash tools/run_sfod_arms_cga.sh <corruption> [score_thr] [gpu]
set -uo pipefail

CORRUPT="${1:?usage: run_sfod_arms_cga.sh <corruption> [score_thr] [gpu]}"
SCORE_THR="${2:-0.82}"
GPU="${3:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LORA="$REPO/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth"
if [ ! -f "$LORA" ]; then
  echo "FATAL: LoRA not found at $LORA -- run build_patches_and_train_lora.sh first"
  exit 1
fi

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="$GPU"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/opt/conda/envs/iraod}/lib:${LD_LIBRARY_PATH:-}"

# CGA enabled, veto_soft -- matching the run.txt configuration.
export CGA_SCORER=sarclip
export CGA_BACKEND=sarclip
export CGA_FILTER_MODE=veto_soft
export CGA_DROP_SCORE=0.0
export CGA_FILTER_LOG_EVERY=500
export CGA_VETO_PRED_THR=0.7
export CGA_VETO_LABEL_THR=0.1
export CGA_PROTECT_DET_SCORE=0.9
export CGA_BLEND_DET_WEIGHT=0.7
export SARCLIP_LORA="$LORA"
# Point at the local SARCLIP weights (cga.py defaults to old machine path).
export SARCLIP_PRETRAINED="/myfile/dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors"
export SARCLIP_CACHE_DIR="/myfile/dataset/SARCLIP/ViT-B-32"

CFG_DIR=configs/unbiased_teacher/sfod
LOG_DIR="work_dirs/cga_arms_${CORRUPT}_thr${SCORE_THR//./}"
mkdir -p "$LOG_DIR"

run_arm () {
  local name="$1" cfg="$2"
  local wd="work_dirs/cga_arm_${name}_${CORRUPT}_thr${SCORE_THR//./}"
  local log="$LOG_DIR/${name}.log"
  echo "=== arm ${name} -> ${wd}"
  timeout 28800 python train.py "$cfg" \
    --cfg-options corrupt="$CORRUPT" \
      model.cfg.score_thr="${SCORE_THR}" \
      work_dir="$wd" > "$log" 2>&1
  local status=$?
  echo "    exit=${status}"
  grep -ao "mAP: [0-9.]*" "$log" | tail -1 || echo "    (no mAP)"
  grep -a "pseudo_num:" "$log" | tail -1 | grep -o "pseudo_num: [0-9.]*.*pseudo_num(acc): [0-9.]*" || true
}

run_arm noslrp "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet.py"
run_arm slrp   "$CFG_DIR/unbiased_teacher_oriented_rcnn_selftraining_slrp_rsar_orthonet.py"

echo
echo "=== summary (CGA on, score_thr=${SCORE_THR}, corrupt=${CORRUPT}) ==="
for name in noslrp slrp; do
  log="$LOG_DIR/${name}.log"
  printf '%-10s ' "$name"
  grep -ao "mAP: [0-9.]*" "$log" 2>/dev/null | tail -1 || echo "(no log)"
done
echo "ALLDONE"
