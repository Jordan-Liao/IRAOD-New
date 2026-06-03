#!/usr/bin/env bash
set -euo pipefail

cd /home/storageSDA1/liaojr/IRAOD-New

export RSAR_ROOT="${RSAR_ROOT:-/home/storageSDA1/liaojr/dataset/RSAR}"
export IRAOD_CONDA_PREFIX="${IRAOD_CONDA_PREFIX:-/home/liaojr/anaconda3/envs/cliptorch}"
export SARCLIP_DIR="${SARCLIP_DIR:-/home/storageSDA1/liaojr/SARCLIP}"
export SARCLIP_PRETRAINED="${SARCLIP_PRETRAINED:-/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors}"
export SARCLIP_CACHE_DIR="${SARCLIP_CACHE_DIR:-/home/storageSDA1/Dataset/SARCLIP/ViT-B-32}"
export SARCLIP_MODEL="${SARCLIP_MODEL:-ViT-B-32}"
export SARCLIP_PRECISION="${SARCLIP_PRECISION:-fp32}"

CORRUPTIONS=(
  chaff
  gaussian_white_noise
  point_target
  noise_suppression
  am_noise_horizontal
  smart_suppression
  am_noise_vertical
)
TEMPLATE="A SAR image of a {}"

if [[ ! -f "${SARCLIP_PRETRAINED}" ]]; then
  echo "SARCLIP_PRETRAINED not found: ${SARCLIP_PRETRAINED}" >&2
  echo "Set SARCLIP_PRETRAINED to a valid ViT-B-32 SARCLIP checkpoint before running." >&2
  exit 1
fi

python tools/build_rsar_sarclip_patches.py \
  --data-root "${RSAR_ROOT}" \
  --split train \
  --use-corruptions 1 \
  --corruptions "${CORRUPTIONS[@]}" \
  --out data/rsar_sarclip_patches/train_corrupt \
  --crop-modes aabb \
  --crop-expands 0.4 \
  --force-rgb 1

python tools/build_rsar_sarclip_patches.py \
  --data-root "${RSAR_ROOT}" \
  --split val \
  --use-corruptions 1 \
  --corruptions "${CORRUPTIONS[@]}" \
  --out data/rsar_sarclip_patches/val_corrupt \
  --crop-modes aabb \
  --crop-expands 0.4 \
  --force-rgb 1

python tools/build_rsar_sarclip_patches.py \
  --data-root "${RSAR_ROOT}" \
  --split test \
  --use-corruptions 1 \
  --corruptions "${CORRUPTIONS[@]}" \
  --out data/rsar_sarclip_patches/test_corrupt \
  --crop-modes aabb \
  --crop-expands 0.4 \
  --force-rgb 1

python tools/eval_sarclip_rsar_patches.py \
  --metadata data/rsar_sarclip_patches/val_corrupt/metadata.csv \
  --crop-mode aabb \
  --sarclip-dir "${SARCLIP_DIR}" \
  --sarclip-pretrained "${SARCLIP_PRETRAINED}" \
  --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" \
  --sarclip-model "${SARCLIP_MODEL}" \
  --precision "${SARCLIP_PRECISION}" \
  --templates "${TEMPLATE}" \
  --batch-size 64 \
  --out work_dirs/sarclip_rsar_diag/before_val_corrupt_aabb

python tools/train_sarclip_lora_rsar.py \
  --metadata data/rsar_sarclip_patches/train_corrupt/metadata.csv \
  --crop-mode aabb \
  --sarclip-dir "${SARCLIP_DIR}" \
  --sarclip-pretrained "${SARCLIP_PRETRAINED}" \
  --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" \
  --sarclip-model "${SARCLIP_MODEL}" \
  --output work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --precision "${SARCLIP_PRECISION}" \
  --templates "${TEMPLATE}" \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0

python tools/eval_sarclip_rsar_patches.py \
  --metadata data/rsar_sarclip_patches/val_corrupt/metadata.csv \
  --crop-mode aabb \
  --sarclip-dir "${SARCLIP_DIR}" \
  --sarclip-pretrained "${SARCLIP_PRETRAINED}" \
  --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" \
  --sarclip-model "${SARCLIP_MODEL}" \
  --precision "${SARCLIP_PRECISION}" \
  --templates "${TEMPLATE}" \
  --batch-size 64 \
  --lora work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth \
  --out work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_val_aabb

python tools/eval_sarclip_rsar_patches.py \
  --metadata data/rsar_sarclip_patches/test_corrupt/metadata.csv \
  --crop-mode aabb \
  --sarclip-dir "${SARCLIP_DIR}" \
  --sarclip-pretrained "${SARCLIP_PRETRAINED}" \
  --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" \
  --sarclip-model "${SARCLIP_MODEL}" \
  --precision "${SARCLIP_PRECISION}" \
  --templates "${TEMPLATE}" \
  --batch-size 64 \
  --lora work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth \
  --out work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_test_aabb
