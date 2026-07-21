#!/usr/bin/env bash
# Prototype-CGA single-seed experiment (seed 42), 3 arms on ONE physical GPU:
#   A. no_cga            B. legacy            C. prototype_legacy
# All re-run from scratch, same source ckpt / corruption / val / subset order.
# Do NOT touch corrupted test.
set -uo pipefail
cd /home/storageSDA1/liaojr/IRAOD-New
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export CONDA_PREFIX=/home/liaojr/anaconda3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
PY=/home/liaojr/anaconda3/envs/iraod/bin/python3.10

GPU_INDEX=${1:-5}
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_prototype_legacy_rsar_research.py
ROOT=work_dirs/prototype_cga_research
LORA=/home/storageSDA1/liaojr/IRAOD-New/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth
SEED=42
CORRUPT=chaff

common_cfgopts=(
  corrupt=$CORRUPT
  optimizer.lr=0.000125
  model.cfg.weight_l=1.0
  model.cfg.weight_u=0.3
  model.cfg.score_thr=0.9
)

run_one () {
  local name="$1"; shift
  local wd="$ROOT/runs/${name}_seed${SEED}"
  local log="$ROOT/logs/${name}_seed${SEED}.log"
  mkdir -p "$wd" "$(dirname "$log")"
  echo "=================================================================="
  echo "[launcher] START $name  seed=$SEED  gpu_index=$GPU_INDEX  $(date)"
  echo "[launcher] CGA_SCORER=${CGA_SCORER:-<unset>} CGA_FILTER_MODE=${CGA_FILTER_MODE:-<unset>}"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES=$GPU_INDEX $PY train.py "$CFG" \
      --work-dir "$wd" --seed $SEED --deterministic \
      --cfg-options "${common_cfgopts[@]}" > "$log" 2>&1
  echo "[launcher] DONE  $name  exit=$?  $(date)"
}

# --- Arm A: no_cga ------------------------------------------------------------
CGA_SCORER=none \
run_one no_cga

# --- Arm B: legacy ------------------------------------------------------------
CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=legacy \
CGA_EXPAND_RATIO=0.4 CGA_BLEND_DET_WEIGHT=0.7 SARCLIP_LORA=$LORA \
run_one legacy

# --- Arm C: prototype_legacy --------------------------------------------------
CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=prototype_legacy \
CGA_BLEND_DET_WEIGHT=0.70 CGA_PROTO_BETA=0.50 CGA_PROTO_MOMENTUM=0.95 \
CGA_PROTO_MIN_COUNT=20 CGA_PROTO_CONTEXT_RATIO=0.15 CGA_PROTO_ROTATED_CROP=1 \
CGA_STRICT=1 SARCLIP_LORA=$LORA \
CGA_PROTO_DIAG_CSV=$ROOT/diag/prototype_legacy_seed${SEED}.csv \
run_one prototype_legacy

echo "[launcher] ALL 3 ARMS COMPLETE $(date)"
