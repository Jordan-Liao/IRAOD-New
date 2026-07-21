#!/usr/bin/env bash
# Prototype-CGA v2 calibrated single-seed screening on one physical GPU.
# Sequential arms: legacy, then prototype_legacy_v2. Corrupted-test is unused.
set -euo pipefail

cd "/home/storageSDA1/liaojr/IRAOD-New"
unset PYTHONPATH
unset CGA_PROTO_CONTEXT_RATIO
unset CGA_PROTO_ROTATED_CROP
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export CONDA_PREFIX="/home/liaojr/anaconda3/envs/iraod"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

PY="/home/liaojr/anaconda3/envs/iraod/bin/python3.10"
GPU_INDEX="${1:-1}"
EXPECTED_GPU_UUID="GPU-4a145182-f129-af04-2cc5-d275f6c25802"
ACTUAL_GPU_UUID="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ "${ACTUAL_GPU_UUID}" != "${EXPECTED_GPU_UUID}" ]]; then
  echo "[launcher] GPU UUID mismatch: ${ACTUAL_GPU_UUID} != ${EXPECTED_GPU_UUID}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

CFG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_prototype_legacy_v2_rsar_research.py"
ROOT="work_dirs/prototype_cga_v2_research"
LORA="/home/storageSDA1/liaojr/IRAOD-New/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth"
SEED="42"

export CGA_SCORER="sarclip"
export CGA_BACKEND="sarclip"
export CGA_TAU="100.0"
export CGA_EXPAND_RATIO="0.4"
export CGA_BLEND_DET_WEIGHT="0.70"
export CGA_TEMPLATES="A SAR image of a {};This SAR patch shows a {}"
export CGA_FORCE_GRAYSCALE="0"
export SARCLIP_MODEL="ViT-B-32"
export SARCLIP_PRETRAINED="/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors"
export SARCLIP_CACHE_DIR="/home/storageSDA1/Dataset/SARCLIP/ViT-B-32"
export SARCLIP_PRECISION="fp32"
export SARCLIP_LORA="${LORA}"
export CGA_PROTO_BETA="0.50"
export CGA_PROTO_MOMENTUM="0.95"
export CGA_PROTO_MIN_COUNT="20"
export CGA_PROTO_SCORE_THR="0.97"
export CGA_PROTO_IOU_THR="0.70"
export CGA_STRICT="1"

COMMON_CFG_OPTIONS=(
  "corrupt=chaff"
  "optimizer.lr=0.000125"
  "model.cfg.weight_l=0.0"
  "model.cfg.weight_u=1.0"
  "model.cfg.score_thr=0.9"
)

mkdir -p "${ROOT}/logs" "${ROOT}/diag" "${ROOT}/runs"
if [[ ! -f "${ROOT}/run_manifest_seed42.json" ]]; then
  echo "[launcher] missing manifest: ${ROOT}/run_manifest_seed42.json" >&2
  exit 3
fi

run_one() {
  local mode="$1"
  local work_dir="${ROOT}/runs/${mode}_seed${SEED}"
  local log_path="${ROOT}/logs/${mode}_seed${SEED}.log"
  if [[ -e "${work_dir}" || -e "${log_path}" ]]; then
    echo "[launcher] refusing to overwrite ${work_dir} or ${log_path}" >&2
    exit 4
  fi
  mkdir -p "${work_dir}"
  export CGA_FILTER_MODE="${mode}"
  if [[ "${mode}" == "prototype_legacy_v2" ]]; then
    export CGA_PROTO_DIAG_CSV="${ROOT}/diag/prototype_legacy_v2_seed${SEED}.csv"
  else
    unset CGA_PROTO_DIAG_CSV
  fi

  echo "[launcher] START mode=${mode} seed=${SEED} gpu_index=${GPU_INDEX} gpu_uuid=${ACTUAL_GPU_UUID} $(date --iso-8601=seconds)"
  "${PY}" "train.py" "${CFG}" \
    --work-dir "${work_dir}" \
    --seed "${SEED}" \
    --deterministic \
    --cfg-options "${COMMON_CFG_OPTIONS[@]}" \
    > "${log_path}" 2>&1
  echo "[launcher] DONE mode=${mode} exit=0 $(date --iso-8601=seconds)"
}

run_one "legacy"
run_one "prototype_legacy_v2"
echo "[launcher] ALL ARMS COMPLETE $(date --iso-8601=seconds)"
