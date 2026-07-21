#!/usr/bin/env bash
# Prototype-CGA smoke: <=8 iterations on a tiny corrupted-val unlabeled subset.
# Exercises the FULL prototype_legacy path end-to-end with real SARCLIP+LoRA:
# rotated crop, single encode per proposal, weak/strong matching, EMA bank,
# strict mode. NOT a real experiment. Arg1 = GPU index.
set -euo pipefail
cd /home/storageSDA1/liaojr/IRAOD-New
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export CONDA_PREFIX=/home/liaojr/anaconda3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
PY=/home/liaojr/anaconda3/envs/iraod/bin/python3.10

export CGA_SCORER=sarclip
export CGA_BACKEND=sarclip
export CGA_FILTER_MODE=prototype_legacy
export CGA_BLEND_DET_WEIGHT=0.70
export CGA_PROTO_BETA=0.50
export CGA_PROTO_MOMENTUM=0.95
# smoke: lower min_count so a prototype can activate within a few iters
export CGA_PROTO_MIN_COUNT=3
export CGA_PROTO_CONTEXT_RATIO=0.15
export CGA_PROTO_ROTATED_CROP=1
export CGA_STRICT=1
export CGA_FILTER_LOG_EVERY=1
export CGA_PROTO_DIAG_CSV=work_dirs/prototype_cga_research/diag/smoke_diag.csv
export SARCLIP_LORA=/home/storageSDA1/liaojr/IRAOD-New/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_prototype_legacy_rsar_research.py
GPU=${1:-2}
WD=work_dirs/prototype_cga_research/smoke_gpu${GPU}
rm -f "$CGA_PROTO_DIAG_CSV"

CUDA_VISIBLE_DEVICES=$GPU $PY train.py "$CFG" \
  --work-dir "$WD" \
  --seed 42 --deterministic \
  --cfg-options \
  corrupt=chaff \
  optimizer.lr=0.000125 \
  model.cfg.weight_l=1.0 \
  model.cfg.weight_u=0.3 \
  model.cfg.score_thr=0.9 \
  data.train.unlabeled_epoch_size=16 \
  data.workers_per_gpu=0 \
  evaluation.interval=999 \
  checkpoint_config.interval=999 \
  log_config.interval=1
