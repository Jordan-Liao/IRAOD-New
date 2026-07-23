#!/usr/bin/env bash
# StripNet-S (scratch) RSAR source training followed by CGA-free UT adaptation.
set -euo pipefail

cd "/home/storageSDA1/liaojr/IRAOD-New"

PY="/home/liaojr/anaconda3/envs/iraod/bin/python3.10"
GPU_INDEX="${1:-1}"
EXPECTED_GPU_UUID="GPU-4a145182-f129-af04-2cc5-d275f6c25802"
GPU_UUID="$(nvidia-smi -i "${GPU_INDEX}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"

if [[ "${GPU_UUID}" != "${EXPECTED_GPU_UUID}" ]]; then
  echo "[launcher] GPU UUID mismatch: ${GPU_UUID} != ${EXPECTED_GPU_UUID}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export PYTHONNOUSERSITE="1"
export PYTHONUNBUFFERED="1"
export CONDA_PREFIX="/home/liaojr/anaconda3/envs/iraod"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Explicitly disable all CGA/prototype paths.  SOGC, robust projection and
# Strip Head are not present in either resolved configuration.
export CGA_SCORER="none"
export CGA_BACKEND="none"
export CGA_FILTER_MODE="none"
export CGA_STRICT="0"
unset SARCLIP_LORA CGA_PROTO_DIAG_CSV CGA_PROTO_ROTATED_CROP

SEED="42"
CORRUPT="chaff"
SOURCE_CFG="configs/baseline/oriented_rcnn_stripnet_rsar.py"
SFOD_CFG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_stripnet_rsar.py"
SOURCE_WORK="work_dirs/oriented_rcnn_stripnet_rsar"
ADAPT_WORK="work_dirs/unbiased_teacher_oriented_rcnn_stripnet_rsar"
ROOT="work_dirs/stripnet_rsar_experiment"
LOG_DIR="${ROOT}/logs"
EVAL_DIR="${ROOT}/eval"
SOURCE_EPOCH="${SOURCE_WORK}/epoch_100.pth"
SOURCE_CKPT="${SOURCE_WORK}/stripnet_rsar_source.pth"
ADAPTED_EPOCH="${ADAPT_WORK}/iter_4235_ema.pth"
ADAPTED_CKPT="${ADAPT_WORK}/stripnet_rsar_adapted_ema.pth"

if [[ -e "${SOURCE_WORK}" || -e "${ADAPT_WORK}" || -e "${ROOT}/.started" ]]; then
  echo "[launcher] refusing to overwrite an existing formal run" >&2
  exit 3
fi

mkdir -p "${LOG_DIR}" "${EVAL_DIR}"
touch "${ROOT}/.started"

echo "[launcher] START seed=${SEED} corruption=${CORRUPT} gpu_index=${GPU_INDEX} gpu_uuid=${GPU_UUID} $(date --iso-8601=seconds)"
echo "[launcher] StripNet init_cfg=None; no pretrained checkpoint is used"

echo "[launcher] STAGE source_train START $(date --iso-8601=seconds)"
"${PY}" "train.py" "${SOURCE_CFG}" \
  --work-dir "${SOURCE_WORK}" \
  --seed "${SEED}" \
  --deterministic \
  > "${LOG_DIR}/source_train.log" 2>&1
test -f "${SOURCE_EPOCH}"
cp "${SOURCE_EPOCH}" "${SOURCE_CKPT}"
echo "[launcher] STAGE source_train DONE checkpoint=${SOURCE_CKPT} $(date --iso-8601=seconds)"

echo "[launcher] STAGE source_clean_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SOURCE_CFG}" "${SOURCE_CKPT}" \
  --eval mAP \
  --work-dir "${EVAL_DIR}/source_clean_test" \
  > "${LOG_DIR}/source_clean_test.log" 2>&1
echo "[launcher] STAGE source_clean_test DONE $(date --iso-8601=seconds)"

echo "[launcher] STAGE source_chaff_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SOURCE_CFG}" "${SOURCE_CKPT}" \
  --eval mAP \
  --work-dir "${EVAL_DIR}/source_chaff_test" \
  --cfg-options \
    data.test.ann_file="/home/storageSDA1/liaojr/dataset/RSAR/test/annfiles/" \
    data.test.img_prefix="/home/storageSDA1/liaojr/dataset/RSAR/corruptions/${CORRUPT}/test/images/" \
  > "${LOG_DIR}/source_chaff_test.log" 2>&1
echo "[launcher] STAGE source_chaff_test DONE $(date --iso-8601=seconds)"

echo "[launcher] STAGE target_adaptation START $(date --iso-8601=seconds)"
"${PY}" "train.py" "${SFOD_CFG}" \
  --work-dir "${ADAPT_WORK}" \
  --seed "${SEED}" \
  --deterministic \
  --no-validate \
  --cfg-options corrupt="${CORRUPT}" \
  > "${LOG_DIR}/target_adaptation.log" 2>&1
test -f "${ADAPTED_EPOCH}"
cp "${ADAPTED_EPOCH}" "${ADAPTED_CKPT}"
echo "[launcher] STAGE target_adaptation DONE checkpoint=${ADAPTED_CKPT} $(date --iso-8601=seconds)"

echo "[launcher] STAGE adapted_clean_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SOURCE_CFG}" "${ADAPTED_CKPT}" \
  --eval mAP \
  --work-dir "${EVAL_DIR}/adapted_clean_test" \
  > "${LOG_DIR}/adapted_clean_test.log" 2>&1
echo "[launcher] STAGE adapted_clean_test DONE $(date --iso-8601=seconds)"

echo "[launcher] STAGE adapted_chaff_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SFOD_CFG}" "${ADAPTED_CKPT}" \
  --eval mAP \
  --work-dir "${EVAL_DIR}/adapted_chaff_test" \
  --cfg-options corrupt="${CORRUPT}" \
  > "${LOG_DIR}/adapted_chaff_test.log" 2>&1
echo "[launcher] STAGE adapted_chaff_test DONE $(date --iso-8601=seconds)"

touch "${ROOT}/.complete"
echo "[launcher] ALL STAGES COMPLETE $(date --iso-8601=seconds)"
