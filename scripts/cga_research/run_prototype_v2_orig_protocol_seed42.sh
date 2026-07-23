#!/usr/bin/env bash
# Prototype-CGA v2 under the ORIGINAL 0.6929 protocol (paired control).
#
# Protocol = exactly the run that produced legacy mAP=0.6929:
#   config prototype_legacy_rsar_research.py -> weight_l=1.0, weight_u=0.3,
#   unlabeled = corrupted TRAIN (epoch_size 8467), score_thr=0.9, lr 1.25e-4,
#   seed 42, deterministic, source epoch_100, same GPU 5 (GPU-8ba88aa4).
#
# Only CGA_FILTER_MODE differs between the two arms:
#   A. legacy               (text-only SARCLIP prototypes; should reproduce ~0.6929)
#   B. prototype_legacy_v2  (text+visual fusion; AABB crop = same as legacy,
#                            degrades to legacy exactly when no prototype active)
# Question: does adding target-domain visual prototypes lift SARCLIP's help?
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

# EXACT config + cfg-options that produced legacy=0.6929.
CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_prototype_legacy_rsar_research.py
ROOT=work_dirs/prototype_v2_orig_protocol
LORA=/home/storageSDA1/liaojr/IRAOD-New/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth
SEED=42
CFGOPTS=(corrupt=chaff optimizer.lr=0.000125 model.cfg.weight_l=1.0 model.cfg.weight_u=0.3 model.cfg.score_thr=0.9)

run_one () {
  local name="$1"
  local wd="$ROOT/runs/${name}_seed${SEED}"
  local log="$ROOT/logs/${name}_seed${SEED}.log"
  mkdir -p "$wd"
  echo "=================================================================="
  echo "[launcher] START $name seed=$SEED gpu=$GPU_INDEX MODE=$CGA_FILTER_MODE $(date)"
  echo "=================================================================="
  CUDA_VISIBLE_DEVICES=$GPU_INDEX $PY train.py "$CFG" \
    --work-dir "$wd" --seed $SEED --deterministic \
    --cfg-options "${CFGOPTS[@]}" > "$log" 2>&1
  echo "[launcher] DONE $name exit=$? $(date)"
}

# --- Arm A: legacy (paired control, reproduces 0.6929 protocol) ---------------
CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=legacy \
CGA_EXPAND_RATIO=0.4 CGA_BLEND_DET_WEIGHT=0.7 SARCLIP_LORA=$LORA \
run_one legacy

# --- Arm B: prototype_legacy_v2 (text+visual fusion) --------------------------
CGA_SCORER=sarclip CGA_BACKEND=sarclip CGA_FILTER_MODE=prototype_legacy_v2 \
CGA_EXPAND_RATIO=0.4 CGA_BLEND_DET_WEIGHT=0.70 \
CGA_PROTO_BETA=0.50 CGA_PROTO_MOMENTUM=0.95 CGA_PROTO_MIN_COUNT=20 \
CGA_STRICT=1 SARCLIP_LORA=$LORA \
CGA_PROTO_DIAG_CSV=$ROOT/diag/prototype_legacy_v2_seed${SEED}.csv \
run_one prototype_legacy_v2

echo "[launcher] ALL ARMS COMPLETE $(date)"
