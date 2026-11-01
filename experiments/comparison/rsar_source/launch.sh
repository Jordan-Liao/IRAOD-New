#!/usr/bin/env bash
set -euo pipefail
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-rsar-source
RUN=/mnt/shared/zechuan/iraod_artifacts/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64
cd "$CODE"
export RSAR_ROOT=/mnt/shared/zechuan/iraod_data/RSAR
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29574
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export IRAOD_RUNTIME_READY=1
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export IRAOD_CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
printf '%s\n' "$PY" -m torch.distributed.launch --nproc_per_node=4 --master_port=29574 train.py "$RUN/reproducibility/resolved_config.py" --launcher pytorch --work-dir "$RUN/train" --seed 42 --deterministic > "$RUN/reproducibility/command.txt"
echo "[source] START $(date --iso-8601=seconds) world=4 spg=16 gbs=64 lr=0.08 seed=42 gpus=4,5,6,7" | tee -a "$RUN/logs/source_train.log"
set +e
"$PY" -m torch.distributed.launch --nproc_per_node=4 --master_port=29574 \
  train.py "$RUN/reproducibility/resolved_config.py" \
  --launcher pytorch --work-dir "$RUN/train" --seed 42 --deterministic \
  >> "$RUN/logs/source_train.log" 2>&1
TRAIN_EXIT=$?
set -e
CKPT="$RUN/train/epoch_100.pth"
if [[ "$TRAIN_EXIT" -eq 0 && -f "$CKPT" ]]; then STATUS=succeeded; else STATUS=failed; fi
{
  echo "status=${STATUS}"
  echo "exit_code=${TRAIN_EXIT}"
  echo "checkpoint=${CKPT}"
  date --iso-8601=seconds
} > "$RUN/terminal_status"
echo "[source] TERMINAL ${STATUS} exit=${TRAIN_EXIT}" | tee -a "$RUN/logs/source_train.log"
exit "$TRAIN_EXIT"
