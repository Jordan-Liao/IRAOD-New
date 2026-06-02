# IRAOD-New Environment Setup

This document records the environment layout used by this project and the
recommended workflow for the SARCLIP-LoRA rescoring experiments.

## 1. Enter the Project

```bash
cd /home/storageSDA1/liaojr/IRAOD-New
```

## 2. Detector Environment: `iraod`

Use this environment for the IRAOD/MMRotate detector code.

```bash
conda create -n iraod python=3.10 -y
conda activate iraod
```

Install the CUDA 11.8 PyTorch build used by the detector:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r /home/storageSDA1/liaojr/IRAOD-New/requirements.txt
```

Raw self-training on RSAR chaff:

```bash
conda activate iraod
python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_rsar1.py --cfg-options corrupt="chaff"
```

## 3. SARCLIP Environment: `cliptorch`

Use this environment for SARCLIP patch extraction, SARCLIP zero-shot evaluation,
and SARCLIP-LoRA fine-tuning/evaluation.

The SARCLIP utilities in `tools/` default to:

```bash
IRAOD_CONDA_PREFIX=/home/liaojr/anaconda3/envs/cliptorch
```

Override it if your SARCLIP-compatible environment is somewhere else:

```bash
export IRAOD_CONDA_PREFIX=/path/to/your/cliptorch
```

The expected SARCLIP paths are:

```bash
export SARCLIP_DIR=/home/storageSDA1/liaojr/SARCLIP
export SARCLIP_PRETRAINED=/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors
export SARCLIP_CACHE_DIR=/home/storageSDA1/Dataset/SARCLIP/ViT-B-32
export SARCLIP_MODEL=ViT-B-32
export SARCLIP_PRECISION=fp32
```

The RSAR SARCLIP-LoRA workflow is split into direct Python commands:

```bash
python tools/build_rsar_sarclip_patches.py --split train --out data/rsar_sarclip_patches/train_corrupt --crop-modes aabb --crop-expands 0.2
python tools/build_rsar_sarclip_patches.py --split val --out data/rsar_sarclip_patches/val_corrupt --crop-modes aabb --crop-expands 0.2
python tools/build_rsar_sarclip_patches.py --split test --out data/rsar_sarclip_patches/test_corrupt --crop-modes aabb --crop-expands 0.2
python tools/train_sarclip_lora_rsar.py --metadata data/rsar_sarclip_patches/train_corrupt/metadata.csv --crop-mode aabb --sarclip-pretrained "${SARCLIP_PRETRAINED}" --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" --sarclip-model "${SARCLIP_MODEL}" --precision "${SARCLIP_PRECISION}" --output work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1
python tools/eval_sarclip_rsar_patches.py --metadata data/rsar_sarclip_patches/val_corrupt/metadata.csv --crop-mode aabb --sarclip-pretrained "${SARCLIP_PRETRAINED}" --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" --sarclip-model "${SARCLIP_MODEL}" --precision "${SARCLIP_PRECISION}" --lora work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth --out work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_val_aabb
python tools/eval_sarclip_rsar_patches.py --metadata data/rsar_sarclip_patches/test_corrupt/metadata.csv --crop-mode aabb --sarclip-pretrained "${SARCLIP_PRETRAINED}" --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" --sarclip-model "${SARCLIP_MODEL}" --precision "${SARCLIP_PRECISION}" --lora work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth --out work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_test_aabb
```

These commands build RSAR object patches, fine-tune the LoRA adapter, and
evaluate the adapter on validation/test patches.

Main outputs:

```text
work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth
work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_val_aabb/metrics.json
work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_test_aabb/metrics.json
```

## 4. Using the Fine-Tuned SARCLIP Adapter

The fine-tuned adapter is loaded through `SARCLIP_LORA`:

```bash
export SARCLIP_LORA=/path/to/lora_rsar.pth
```

For offline SARCLIP patch evaluation:

```bash
python tools/eval_sarclip_rsar_patches.py \
  --metadata data/rsar_sarclip_patches/test_corrupt/metadata.csv \
  --crop-mode aabb \
  --sarclip-dir "${SARCLIP_DIR}" \
  --sarclip-pretrained "${SARCLIP_PRETRAINED}" \
  --sarclip-cache-dir "${SARCLIP_CACHE_DIR}" \
  --sarclip-model "${SARCLIP_MODEL}" \
  --precision "${SARCLIP_PRECISION}" \
  --lora "${SARCLIP_LORA}" \
  --out work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/eval_test_aabb
```

For online CGA/SARCLIP detector rescoring, the detector process must be able to
import and run SARCLIP in the same Python runtime:

```bash
CGA_SCORER=sarclip \
SARCLIP_LORA=/path/to/lora_rsar.pth \
SARCLIP_DIR=/home/storageSDA1/liaojr/SARCLIP \
SARCLIP_PRETRAINED=/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/vit_b_32_model.safetensors \
SARCLIP_CACHE_DIR=/home/storageSDA1/Dataset/SARCLIP/ViT-B-32 \
python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_rsar1.py --cfg-options corrupt="chaff"
```

Do not mix packages from `cliptorch` directly into `iraod`. The two environments
use different core dependency versions, and mixing them can break either the
detector stack or SARCLIP. If online CGA/SARCLIP is required during detector
training, prefer creating a new clean environment that supports both the
detector and SARCLIP, then set `IRAOD_CONDA_PREFIX` or activate that environment.

## 5. Releasing Weights vs. Releasing Precomputed Scores

Publishing `lora_rsar.pth` is enough for users who can run SARCLIP. They still
need the SARCLIP code, the SARCLIP base checkpoint, and a SARCLIP-compatible
Python environment to compute rescoring results.

If users should avoid the SARCLIP environment entirely, release precomputed
rescoring artifacts instead. This is only valid for the exact dataset split,
corruption setting, teacher checkpoint, detector predictions, crop mode, and
class order used to generate the scores. A different detector checkpoint or
pseudo-label set requires recomputing the SARCLIP scores.

Current code supports loading the fine-tuned adapter for online SARCLIP rescoring
through `SARCLIP_LORA`. It does not yet include a detector-side cache reader for
fully precomputed CGA scores. To make detector training run only in `iraod`
without importing SARCLIP, add an offline score cache/export step and a matching
cache lookup path in `sfod/cga.py`.

## 6. Notes

- `iraod` is the stable detector environment.
- `cliptorch` is the stable SARCLIP/SARCLIP-LoRA environment.
- SARCLIP utility scripts automatically switch to `IRAOD_CONDA_PREFIX`.
- The LoRA adapter keeps releases small, but it is not a standalone model; it
  must be loaded together with the SARCLIP base checkpoint.
