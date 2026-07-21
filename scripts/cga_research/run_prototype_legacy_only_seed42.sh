#!/usr/bin/env bash
# Rerun ONLY the prototype_legacy arm (seed 42) on the SAME physical GPU 5 as the
# already-completed no_cga (0.6917) / legacy (0.6929) arms. Fixes the no_grad bug
# in the strong-view pass. Does NOT touch corrupted test.
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

export CGA_SCORER=sarclip
export CGA_BACKEND=sarclip
export CGA_FILTER_MODE=prototype_legacy
export CGA_BLEND_DET_WEIGHT=0.70
export CGA_PROTO_BETA=0.50
export CGA_PROTO_MOMENTUM=0.95
export CGA_PROTO_MIN_COUNT=20
export CGA_PROTO_CONTEXT_RATIO=0.15
export CGA_PROTO_ROTATED_CROP=1
export CGA_STRICT=1
export SARCLIP_LORA=$LORA
export CGA_PROTO_DIAG_CSV=$ROOT/diag/prototype_legacy_seed42.csv

WD=$ROOT/runs/prototype_legacy_seed42
LOG=$ROOT/logs/prototype_legacy_seed42.log
mkdir -p "$WD" "$(dirname "$LOG")"

echo "[launcher] START prototype_legacy seed=$SEED gpu_index=$GPU_INDEX $(date)"
CUDA_VISIBLE_DEVICES=$GPU_INDEX $PY train.py "$CFG" \
  --work-dir "$WD" --seed $SEED --deterministic \
  --cfg-options \
  corrupt=chaff \
  optimizer.lr=0.000125 \
  model.cfg.weight_l=1.0 \
  model.cfg.weight_u=0.3 \
  model.cfg.score_thr=0.9 > "$LOG" 2>&1
echo "[launcher] DONE prototype_legacy exit=$? $(date)"
