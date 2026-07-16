#!/usr/bin/env bash
# SRW smoke test: <=8 iterations, semantic_reweight mode, batch size 2.
# Confirms wiring end-to-end; NOT a real experiment.
set -euo pipefail

cd /home/storageSDA1/liaojr/IRAOD-New
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export CONDA_PREFIX=/home/liaojr/anaconda3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
PY=/home/liaojr/anaconda3/envs/iraod/bin/python3.10

# SRW / SARCLIP CGA environment.
export CGA_SCORER=sarclip
export CGA_BACKEND=sarclip
export CGA_FILTER_MODE=semantic_reweight
export CGA_SEM_LOW_THR=0.90
export CGA_SEM_HIGH_THR=0.95
export CGA_SEM_LAMBDA=0.50
export CGA_FILTER_LOG_EVERY=1
export SARCLIP_LORA=/home/storageSDA1/liaojr/IRAOD-New/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth

CFG=configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_srw_rsar_research.py
WD=work_dirs/srw_research/smoke_srw_gpu${1:-4}
GPU=${1:-4}

CUDA_VISIBLE_DEVICES=$GPU $PY train.py "$CFG" \
  --work-dir "$WD" \
  --seed 41 --deterministic \
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
