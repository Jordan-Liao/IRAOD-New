#!/usr/bin/env bash
# VLST 4-arm ablation queue: runs arm B (ST+CGA) then arm D (ST+CGA+Proto)
# after arm C (currently training, pid passed as $1) finishes.
set -u

ARM_C_PID="${1:?usage: queue_arms_bd.sh <arm_c_pid>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/opt/conda/envs/iraod/lib:${LD_LIBRARY_PATH:-}"

while kill -0 "$ARM_C_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date +%T)] arm C finished, launching arm B"

/opt/conda/envs/iraod/bin/python train.py \
    configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga_rsar_orthonet_arm_b.py \
    --cfg-options corrupt=noise_suppression work_dir=work_dirs/st_cga_noise_suppression \
    > /tmp/vlst_full_arm_b.log 2>&1
echo "[$(date +%T)] arm B finished, launching arm D"

/opt/conda/envs/iraod/bin/python train.py \
    configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_rsar_orthonet.py \
    --cfg-options corrupt=noise_suppression work_dir=work_dirs/vlst_cga_baseline_noise_suppression \
    > /tmp/vlst_full_arm_d.log 2>&1
echo "[$(date +%T)] arm D finished"
