#!/usr/bin/env bash
# Portable clean-source OrthoNet-50-FPN baseline on one or four physical GPUs.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: run_orthonet_rsar_source_seed42.sh (--gpu GPU | --gpus GPU,GPU,GPU,GPU) [--port PORT] [--smoke-iters N]

--gpu preserves the one-GPU path. --gpus requires physical GPUs 4,5,6,7 in
that exact logical-rank order and runs distributed training with
samples_per_gpu=1, preserving the one-GPU global batch of 4 and LR 0.005.
--smoke-iters is one-GPU-only and runs exactly N finite-loss optimizer steps in
a separate smoke directory; it never selects a checkpoint. Without
--smoke-iters, this is the full 100-epoch source run and requires epoch_100.pth.
The default four-GPU RUN_ROOT is work_dirs/orthonet_rsar_source_seed42_4gpu.
Existing run directories are always refused: retry or restart with a new
RUN_ROOT; in-place resume is intentionally unsupported. This launcher never
stops or modifies an in-progress run.
EOF
}

require_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "[source-baseline] ${1} requires a value" >&2
    exit 2
  fi
}

GPU_INDEX=""
GPU_INDICES=""
MASTER_PORT="${MASTER_PORT:-20067}"
SMOKE_ITERS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      require_value "$@"
      GPU_INDEX="$2"
      shift 2
      ;;
    --gpus)
      require_value "$@"
      GPU_INDICES="$2"
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
      echo "[source-baseline] unknown argument: $1 (use --gpu GPU or --gpus GPU,GPU,GPU,GPU)" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${GPU_INDEX}" && -n "${GPU_INDICES}" ]]; then
  echo "[source-baseline] use exactly one of --gpu or --gpus" >&2
  exit 2
fi
if [[ -z "${GPU_INDEX}" && -z "${GPU_INDICES}" ]]; then
  echo "[source-baseline] one of --gpu or --gpus is required" >&2
  exit 2
fi
if [[ -n "${GPU_INDEX}" && ! "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[source-baseline] --gpu must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${GPU_INDICES}" ]]; then
  if [[ ! "${GPU_INDICES}" =~ ^[0-9]+,[0-9]+,[0-9]+,[0-9]+$ ]]; then
    echo "[source-baseline] --gpus must contain exactly four comma-separated non-negative integers" >&2
    exit 2
  fi
  if [[ "${GPU_INDICES}" != "4,5,6,7" ]]; then
    echo "[source-baseline] --gpus must be exactly 4,5,6,7 in logical-rank order" >&2
    exit 2
  fi
fi
if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]]; then
  echo "[source-baseline] --port must be a non-negative integer" >&2
  exit 2
fi
if [[ -n "${SMOKE_ITERS}" && ! "${SMOKE_ITERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[source-baseline] --smoke-iters must be a positive integer" >&2
  exit 2
fi
if [[ -n "${GPU_INDICES}" && -n "${SMOKE_ITERS}" ]]; then
  echo "[source-baseline] --smoke-iters supports only the one-GPU --gpu path" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${IRAOD_PYTHON:?Set IRAOD_PYTHON to the exact iraod interpreter.}"
: "${RSAR_ROOT:?Set RSAR_ROOT to the clean RSAR dataset root.}"

PYTHON_BIN="${IRAOD_PYTHON}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[source-baseline] IRAOD_PYTHON is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
SINGLE_FULL_RUN_ROOT="work_dirs/orthonet_rsar_source_seed42"
MODE="full"
if [[ -n "${SMOKE_ITERS}" ]]; then
  MODE="smoke"
  RUN_ROOT="${RUN_ROOT:-${SINGLE_FULL_RUN_ROOT}_smoke_iters${SMOKE_ITERS}}"
  if [[ "${RUN_ROOT}" == "${SINGLE_FULL_RUN_ROOT}" ]]; then
    echo "[source-baseline] smoke RUN_ROOT must differ from the full run" >&2
    exit 2
  fi
elif [[ -n "${GPU_INDICES}" ]]; then
  MODE="distributed"
  RUN_ROOT="${RUN_ROOT:-${SINGLE_FULL_RUN_ROOT}_4gpu}"
  if [[ "${RUN_ROOT}" == "${SINGLE_FULL_RUN_ROOT}" ]]; then
    echo "[source-baseline] distributed RUN_ROOT must differ from the one-GPU full run" >&2
    exit 2
  fi
else
  RUN_ROOT="${RUN_ROOT:-${SINGLE_FULL_RUN_ROOT}}"
fi
SOURCE_WORK="${RUN_ROOT}/train"
REPRO_DIR="${RUN_ROOT}/reproducibility"
SOURCE_CFG="configs/baseline/oriented_rcnn_orthonet_rsar.py"
DATASET_MANIFEST="${REPRO_DIR}/dataset_manifest.json"
RESOLVED_CONFIG="${REPRO_DIR}/resolved_config.py"

# Claim the root atomically so concurrent launchers cannot share one attempt.
mkdir -p "$(dirname "${RUN_ROOT}")"
if ! mkdir "${RUN_ROOT}"; then
  echo "[source-baseline] refusing to reuse run directory: ${RUN_ROOT}" >&2
  exit 2
fi

mkdir -p "${SOURCE_WORK}" "${REPRO_DIR}" "${RUN_ROOT}/logs"
if [[ "${MODE}" == "distributed" ]]; then
  GPU_BINDING="${GPU_INDICES}"
  WORLD_SIZE=4
else
  GPU_BINDING="${GPU_INDEX}"
  WORLD_SIZE=1
fi
export CUDA_VISIBLE_DEVICES="${GPU_BINDING}"
export MASTER_PORT
export PYTHONNOUSERSITE="1"
export PYTHONUNBUFFERED="1"

PREPARE_ARGS=(
  --rsar-root "${RSAR_ROOT}"
  --python "${PYTHON_BIN}"
  --config "${SOURCE_CFG}"
  --work-dir "${SOURCE_WORK}"
  --artifact-dir "${REPRO_DIR}"
  --master-port "${MASTER_PORT}"
  --git-commit "$(git rev-parse HEAD)"
)
if [[ "${MODE}" == "distributed" ]]; then
  PREPARE_ARGS+=(--gpu-indices "${GPU_INDICES}")
else
  PREPARE_ARGS+=(--gpu-index "${GPU_INDEX}")
fi
if [[ "${MODE}" == "smoke" ]]; then
  PREPARE_ARGS+=(--smoke-iters "${SMOKE_ITERS}")
fi
"${PYTHON_BIN}" tools/rsar_source_baseline.py prepare "${PREPARE_ARGS[@]}"

"${PYTHON_BIN}" - "${SOURCE_CFG}" "${RESOLVED_CONFIG}" "${WORLD_SIZE}" <<'PY'
import sys

from mmcv import Config

config = Config.fromfile(sys.argv[1])
if int(sys.argv[3]) == 4:
    config.data.samples_per_gpu = 1
config.dump(sys.argv[2])
PY

# Preparation and manifest creation occur in this parent process exactly once,
# before torch.distributed.launch creates any ranks.
"${PYTHON_BIN}" tools/cga_research/build_data_manifest.py build \
  --ann-root "${RSAR_ROOT}/train/annfiles" \
  --image-root "${RSAR_ROOT}/train/images" \
  --split train \
  --corruption clean \
  --class-order "ship,aircraft,car,tank,bridge,harbor" \
  --output "${DATASET_MANIFEST}"

echo "[source-baseline] START mode=${MODE} seed=42 physical_gpus=${GPU_BINDING} world_size=${WORLD_SIZE} port=${MASTER_PORT}"
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
