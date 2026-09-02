#!/usr/bin/env bash
# Portable clean-source OrthoNet-50-FPN baseline on one physical GPU.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${RSAR_ROOT:?Set RSAR_ROOT to the clean RSAR dataset root.}"

PYTHON_BIN="${IRAOD_PYTHON:-python}"
GPU_INDEX="${1:-0}"
RUN_ROOT="${RUN_ROOT:-work_dirs/orthonet_rsar_source_seed42}"
SOURCE_WORK="${RUN_ROOT}/train"
REPRO_DIR="${RUN_ROOT}/reproducibility"
SOURCE_CFG="configs/baseline/oriented_rcnn_orthonet_rsar.py"
DATASET_MANIFEST="${REPRO_DIR}/dataset_manifest.json"
RESOLVED_CONFIG="${REPRO_DIR}/resolved_config.py"

# Trigger: a reused output tree can mix artifacts from different attempts.
# Git alone cannot distinguish those outputs.  The decision is one fresh tree
# per run; this existence check is the smallest mechanism, consumed by the
# experiment supervisor through the reproducibility artifacts below.
if [[ -e "${RUN_ROOT}" ]]; then
  echo "[source-baseline] refusing to reuse run directory: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${SOURCE_WORK}" "${REPRO_DIR}" "${RUN_ROOT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONNOUSERSITE="1"
export PYTHONUNBUFFERED="1"

"${PYTHON_BIN}" tools/rsar_source_baseline.py prepare \
  --rsar-root "${RSAR_ROOT}" \
  --python "${PYTHON_BIN}" \
  --config "${SOURCE_CFG}" \
  --work-dir "${SOURCE_WORK}" \
  --artifact-dir "${REPRO_DIR}" \
  --gpu-index "${GPU_INDEX}" \
  --git-commit "$(git rev-parse HEAD)"

"${PYTHON_BIN}" - "${SOURCE_CFG}" "${RESOLVED_CONFIG}" <<'PY'
import sys

from mmcv import Config

Config.fromfile(sys.argv[1]).dump(sys.argv[2])
PY

"${PYTHON_BIN}" tools/cga_research/build_data_manifest.py build \
  --ann-root "${RSAR_ROOT}/train/annfiles" \
  --image-root "${RSAR_ROOT}/train/images" \
  --split train \
  --corruption clean \
  --class-order "ship,aircraft,car,tank,bridge,harbor" \
  --output "${DATASET_MANIFEST}"

set +e
"${PYTHON_BIN}" tools/rsar_source_baseline.py execute \
  --artifact-dir "${REPRO_DIR}" \
  --project-root "${REPO_ROOT}" \
  > "${RUN_ROOT}/logs/source_train.log" 2>&1
TRAIN_EXIT_CODE=$?
set -e

"${PYTHON_BIN}" tools/rsar_source_baseline.py finalize \
  --artifact-dir "${REPRO_DIR}" \
  --work-dir "${SOURCE_WORK}" \
  --dataset-manifest "${DATASET_MANIFEST}" \
  --exit-code "${TRAIN_EXIT_CODE}"
FINALIZE_EXIT_CODE=$?

if [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
  exit "${TRAIN_EXIT_CODE}"
fi
exit "${FINALIZE_EXIT_CODE}"
