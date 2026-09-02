#!/usr/bin/env bash
# Portable clean-source OrthoNet-50-FPN baseline on one physical GPU.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: run_orthonet_rsar_source_seed42.sh --gpu GPU [--port PORT] [--smoke-iters N]

--gpu is a physical GPU index and is required. --smoke-iters runs exactly N
finite-loss optimizer steps in a separate smoke directory; it never selects a
checkpoint. Without --smoke-iters, this is the full 100-epoch source run and
requires epoch_100.pth. Existing run directories are always refused: retry or
restart with a new RUN_ROOT; in-place resume is intentionally unsupported.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "[source-baseline] ${1} requires a value" >&2
    exit 2
  fi
}

GPU_INDEX=""
MASTER_PORT="${MASTER_PORT:-20067}"
SMOKE_ITERS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      require_value "$@"
      GPU_INDEX="$2"
      shift 2
      ;;
    --port)
      require_value "$@"
      MASTER_PORT="$2"
      shift 2
      ;;
    --smoke-iters)
      require_value "$@"
      SMOKE_ITERS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[source-baseline] unknown argument: $1 (use --gpu GPU)" >&2
      exit 2
      ;;
  esac
done

if [[ ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[source-baseline] --gpu must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
  echo "[source-baseline] --port must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${SMOKE_ITERS}" && ! "${SMOKE_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[source-baseline] --smoke-iters must be a positive integer" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${RSAR_ROOT:?Set RSAR_ROOT to the clean RSAR dataset root.}"

PYTHON_BIN="${IRAOD_PYTHON:-/opt/conda/envs/iraod/bin/python}"
FULL_RUN_ROOT="work_dirs/orthonet_rsar_source_seed42"
MODE="full"
if [[ -n "${SMOKE_ITERS}" ]]; then
  MODE="smoke"
  RUN_ROOT="${RUN_ROOT:-${FULL_RUN_ROOT}_smoke_iters${SMOKE_ITERS}}"
  if [[ "${RUN_ROOT}" == "${FULL_RUN_ROOT}" ]]; then
    echo "[source-baseline] smoke RUN_ROOT must differ from the full run" >&2
    exit 2
  fi
else
  RUN_ROOT="${RUN_ROOT:-${FULL_RUN_ROOT}}"
fi
SOURCE_WORK="${RUN_ROOT}/train"
REPRO_DIR="${RUN_ROOT}/reproducibility"
SOURCE_CFG="configs/baseline/oriented_rcnn_orthonet_rsar.py"
DATASET_MANIFEST="${REPRO_DIR}/dataset_manifest.json"
RESOLVED_CONFIG="${REPRO_DIR}/resolved_config.py"

# A reused output tree can mix artifacts from distinct attempts, so every
# retry is a new RUN_ROOT and in-place resume is deliberately unsupported.
if [[ -e "${RUN_ROOT}" ]]; then
  echo "[source-baseline] refusing to reuse run directory: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${SOURCE_WORK}" "${REPRO_DIR}" "${RUN_ROOT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export MASTER_PORT
export PYTHONNOUSERSITE="1"
export PYTHONUNBUFFERED="1"

PREPARE_ARGS=(
  --rsar-root "${RSAR_ROOT}"
  --python "${PYTHON_BIN}"
  --config "${SOURCE_CFG}"
  --work-dir "${SOURCE_WORK}"
  --artifact-dir "${REPRO_DIR}"
  --gpu-index "${GPU_INDEX}"
  --git-commit "$(git rev-parse HEAD)"
)
if [[ "${MODE}" == "smoke" ]]; then
  PREPARE_ARGS+=(--smoke-iters "${SMOKE_ITERS}")
fi
"${PYTHON_BIN}" tools/rsar_source_baseline.py prepare "${PREPARE_ARGS[@]}"

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

echo "[source-baseline] START mode=${MODE} seed=42 gpu=${GPU_INDEX} port=${MASTER_PORT}"
set +e
"${PYTHON_BIN}" tools/rsar_source_baseline.py execute \
  --artifact-dir "${REPRO_DIR}" \
  --project-root "${REPO_ROOT}" \
  > "${RUN_ROOT}/logs/source_train.log" 2>&1
TRAIN_EXIT_CODE=$?
set -e

set +e
if [[ "${MODE}" == "smoke" ]]; then
  "${PYTHON_BIN}" tools/rsar_source_baseline.py finalize-smoke \
    --artifact-dir "${REPRO_DIR}" \
    --work-dir "${SOURCE_WORK}" \
    --dataset-manifest "${DATASET_MANIFEST}" \
    --exit-code "${TRAIN_EXIT_CODE}" \
    --expected-iterations "${SMOKE_ITERS}"
else
  "${PYTHON_BIN}" tools/rsar_source_baseline.py finalize \
    --artifact-dir "${REPRO_DIR}" \
    --work-dir "${SOURCE_WORK}" \
    --dataset-manifest "${DATASET_MANIFEST}" \
    --exit-code "${TRAIN_EXIT_CODE}"
fi
FINALIZE_EXIT_CODE=$?
set -e

if [[ "${TRAIN_EXIT_CODE}" -ne 0 ]]; then
  echo "[source-baseline] TERMINAL mode=${MODE} status=failed" >&2
  exit "${TRAIN_EXIT_CODE}"
fi
if [[ "${FINALIZE_EXIT_CODE}" -ne 0 ]]; then
  echo "[source-baseline] TERMINAL mode=${MODE} status=failed" >&2
  exit "${FINALIZE_EXIT_CODE}"
fi
echo "[source-baseline] TERMINAL mode=${MODE} status=succeeded"
