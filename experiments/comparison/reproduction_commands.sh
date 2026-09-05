#!/usr/bin/env bash
# Single-cell and matrix recovery. Run on gpu-67 after ssh.
set -euo pipefail
CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
SRC=/mnt/shared/zechuan/iraod_artifacts/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/rsar/queue_0f98a48
RSAR=/mnt/shared/zechuan/iraod_data/RSAR

# A Source-Only one corruption
# CUDA_VISIBLE_DEVICES=4 $Q/run_A_eval.sh 4 chaff

# B/C/D train one corruption (example chaff already done — do not duplicate)
# CUDA_VISIBLE_DEVICES=4 $Q/run_train_bcd.sh 4 B chaff

# E/F DDP2
# $Q/run_train_ef.sh 4,5 E chaff 29582

# Eval final EMA
# $Q/run_eval_ema_corrupt.sh 4 B chaff

# Full remaining matrix (already launched; skip-if-done):
# bash $Q/queue_gpu567.sh
# bash $Q/queue_rest.sh

echo "See $Q/*.sh and comparison/*/seed_42/*/methods/*/command.txt"
echo "Source $SRC"
echo "Code $(git -C $CODE rev-parse HEAD)"
