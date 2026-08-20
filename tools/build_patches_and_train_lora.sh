#!/usr/bin/env bash
# Build SARCLIP patches from all 7 corrupt train splits, then train LoRA.
set -uo pipefail
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/opt/conda/envs/iraod}/lib:${LD_LIBRARY_PATH:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PATCH_OUT="work_dirs/sarclip_patches_train_corrupt_aabb"
LORA_OUT="work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1"
DATA_ROOT="/myfile/dataset/RSAR/"
SARCLIP_PRETRAINED="/myfile/dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors"
SARCLIP_CACHE="/myfile/dataset/SARCLIP/ViT-B-32"

mkdir -p "$PATCH_OUT" "$LORA_OUT"

echo "=== Stage 1: build patches (all 7 corruptions, train split) ==="
python tools/build_rsar_sarclip_patches.py \
  --data-root "$DATA_ROOT" \
  --split train \
  --out "$PATCH_OUT" \
  --corruptions am_noise_horizontal am_noise_vertical chaff gaussian_white_noise \
                noise_suppression point_target smart_suppression \
  --crop-modes aabb \
  2>&1
echo "Patch build exit=$?"

METADATA="$PATCH_OUT/metadata.csv"
if [ ! -f "$METADATA" ]; then
  echo "FATAL: metadata not found at $METADATA" && exit 1
fi
TOTAL=$(wc -l < "$METADATA")
echo "Total patches: $TOTAL"

echo "=== Stage 2: train LoRA (3 epochs, ViT-B-32) ==="
python tools/train_sarclip_lora_rsar.py \
  --metadata "$METADATA" \
  --sarclip-pretrained "$SARCLIP_PRETRAINED" \
  --sarclip-cache-dir "$SARCLIP_CACHE" \
  --sarclip-model ViT-B-32 \
  --output "$LORA_OUT/lora_rsar.pth" \
  --epochs 3 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --lora-r 8 \
  --lora-alpha 16.0 \
  --crop-mode aabb \
  2>&1
echo "LoRA training exit=$?"

if [ -f "$LORA_OUT/lora_rsar.pth" ]; then
  echo "LoRA saved: $(du -sh $LORA_OUT/lora_rsar.pth)"
else
  echo "FATAL: LoRA weights not found" && exit 1
fi
echo "ALLDONE"
