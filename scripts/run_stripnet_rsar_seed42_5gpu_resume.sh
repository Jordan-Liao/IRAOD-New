#!/usr/bin/env bash
set -euo pipefail

cd "/home/storageSDA1/liaojr/IRAOD-New"

PY="/home/liaojr/anaconda3/envs/iraod/bin/python3.10"
GPU_LIST="1,2,3,4,5"
MASTER_PORT="29615"
EXPECTED_UUIDS=(
  "GPU-4a145182-f129-af04-2cc5-d275f6c25802"
  "GPU-7c7a3244-62a3-35a2-7a8f-2cb71688e026"
  "GPU-014832f9-e0c0-e872-b571-9eed9017da74"
  "GPU-b7cabf6c-1ad6-55c2-bb83-975c2f6a451f"
  "GPU-8ba88aa4-39a6-5876-9009-d49fbf44350f"
)

for i in 1 2 3 4 5; do
  actual="$(nvidia-smi -i "${i}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
  expected="${EXPECTED_UUIDS[$((i - 1))]}"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[5gpu] GPU UUID mismatch at physical GPU ${i}: ${actual} != ${expected}" >&2
    exit 2
  fi
done

export PYTHONNOUSERSITE="1"
export PYTHONUNBUFFERED="1"
export CONDA_PREFIX="/home/liaojr/anaconda3/envs/iraod"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CGA_SCORER="none"
export CGA_BACKEND="none"
export CGA_FILTER_MODE="none"
export CGA_STRICT="0"
unset SARCLIP_LORA CGA_PROTO_DIAG_CSV CGA_PROTO_ROTATED_CROP

SOURCE_CFG="configs/baseline/oriented_rcnn_stripnet_rsar.py"
SFOD_CFG="configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_stripnet_rsar.py"
SOURCE_WORK="work_dirs/oriented_rcnn_stripnet_rsar"
ADAPT_WORK="work_dirs/unbiased_teacher_oriented_rcnn_stripnet_rsar"
ROOT="work_dirs/stripnet_rsar_experiment_5gpu_resume"
LOG_DIR="${ROOT}/logs"
EVAL_DIR="${ROOT}/eval"
# Load the resume checkpoint from local temporary storage.  Five ranks reading
# the shared 244 MB checkpoint concurrently stalled in filesystem I/O.
RESUME_CKPT="/tmp/iraod_stripnet_rsar_epoch6_seed42.pth"
SOURCE_CKPT="${SOURCE_WORK}/stripnet_rsar_source.pth"
ADAPTED_EPOCH="${ADAPT_WORK}/iter_4235_ema.pth"
ADAPTED_CKPT="${ADAPT_WORK}/stripnet_rsar_adapted_ema.pth"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "[5gpu] missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 3
fi
if [[ -e "${ROOT}/.started" || -e "${ROOT}/.complete" ]]; then
  echo "[5gpu] refusing to overwrite an existing 5-GPU run" >&2
  exit 4
fi
if [[ -e "${ADAPT_WORK}" ]]; then
  echo "[5gpu] refusing to overwrite existing adaptation work dir" >&2
  exit 5
fi

mkdir -p "${LOG_DIR}" "${EVAL_DIR}"
touch "${ROOT}/.started"
echo "[5gpu] START seed=42 gpu_list=${GPU_LIST} resume=${RESUME_CKPT} $(date --iso-8601=seconds)"

export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
echo "[5gpu] STAGE source_train_resume START $(date --iso-8601=seconds)"
"${PY}" -m torch.distributed.launch \
  --nproc_per_node=5 \
  --master_port="${MASTER_PORT}" \
  "train.py" "${SOURCE_CFG}" \
  --launcher pytorch \
  --resume-from "${RESUME_CKPT}" \
  --seed 42 \
  --deterministic \
  > "${LOG_DIR}/source_train_5gpu.log" 2>&1
test -f "${SOURCE_WORK}/epoch_100.pth"
cp "${SOURCE_WORK}/epoch_100.pth" "${SOURCE_CKPT}"
echo "[5gpu] STAGE source_train_resume DONE $(date --iso-8601=seconds)"

export CUDA_VISIBLE_DEVICES="1"
echo "[5gpu] STAGE source_clean_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SOURCE_CFG}" "${SOURCE_CKPT}" --eval mAP \
  --work-dir "${EVAL_DIR}/source_clean_test" \
  > "${LOG_DIR}/source_clean_test.log" 2>&1

echo "[5gpu] STAGE source_chaff_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SOURCE_CFG}" "${SOURCE_CKPT}" --eval mAP \
  --work-dir "${EVAL_DIR}/source_chaff_test" \
  --cfg-options \
    data.test.ann_file="/home/storageSDA1/liaojr/dataset/RSAR/test/annfiles/" \
    data.test.img_prefix="/home/storageSDA1/liaojr/dataset/RSAR/corruptions/chaff/test/images/" \
  > "${LOG_DIR}/source_chaff_test.log" 2>&1

echo "[5gpu] STAGE target_adaptation START $(date --iso-8601=seconds)"
"${PY}" "train.py" "${SFOD_CFG}" --work-dir "${ADAPT_WORK}" \
  --seed 42 --deterministic --no-validate \
  --cfg-options corrupt=chaff \
  > "${LOG_DIR}/target_adaptation.log" 2>&1
test -f "${ADAPTED_EPOCH}"
cp "${ADAPTED_EPOCH}" "${ADAPTED_CKPT}"

echo "[5gpu] STAGE adapted_clean_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SOURCE_CFG}" "${ADAPTED_CKPT}" --eval mAP \
  --work-dir "${EVAL_DIR}/adapted_clean_test" \
  > "${LOG_DIR}/adapted_clean_test.log" 2>&1

echo "[5gpu] STAGE adapted_chaff_test START $(date --iso-8601=seconds)"
"${PY}" "test.py" "${SFOD_CFG}" "${ADAPTED_CKPT}" --eval mAP \
  --work-dir "${EVAL_DIR}/adapted_chaff_test" \
  --cfg-options corrupt=chaff \
  > "${LOG_DIR}/adapted_chaff_test.log" 2>&1

touch "${ROOT}/.complete"
echo "[5gpu] ALL STAGES COMPLETE $(date --iso-8601=seconds)"
