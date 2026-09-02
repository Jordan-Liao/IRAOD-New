# IRAOD-New Environment Setup

This document records the unified `iraod` environment used by this project and
the recommended workflow for the SARCLIP-LoRA rescoring experiments.

## 1. Enter the Project

```bash
cd /home/storageSDA1/liaojr/IRAOD-New
```

## 2. Unified Environment: `iraod`

Use `iraod` for both the IRAOD/MMRotate detector code and SARCLIP/SARCLIP-LoRA
rescoring. The old separate `cliptorch` environment is only a historical backup
and is not required for the current workflow.

```bash
conda create -n iraod python=3.10 -y
conda activate iraod
```

Install the CUDA 11.8 PyTorch build and project dependencies:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r /home/storageSDA1/liaojr/IRAOD-New/requirements.txt
```

Install the SARCLIP-side packages in the same environment. Keep the detector
stack versions fixed; do not upgrade PyTorch/MMCV/MMRotate just for SARCLIP.

```bash
pip install "transformers<4.56" timm safetensors peft
```

Before running SARCLIP utilities, make the active `iraod` environment explicit:

```bash
conda activate iraod
export IRAOD_CONDA_PREFIX="$CONDA_PREFIX"
```

Raw self-training on RSAR chaff:

```bash
python train.py configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_rsar1.py --cfg-options corrupt="chaff"
```

## Clean RSAR OrthoNet Source Baseline

From the repository root, run the bounded source-only smoke on physical GPU 6:

```bash
RSAR_ROOT=/mnt/shared/zechuan/iraod_data/RSAR IRAOD_PYTHON=/home/zechuan/miniforge3/envs/iraod/bin/python scripts/run_orthonet_rsar_source_seed42.sh --gpu 6 --port 20067 --smoke-iters 4
```

It performs exactly four finite-loss forward/backward optimizer steps using
only the clean `train` and `val` bindings, writes a fresh
`work_dirs/orthonet_rsar_source_seed42_smoke_iters4/` tree, and does not
evaluate or select a checkpoint. `--gpu` is required and positional arguments
are rejected, so `4` cannot be mistaken for a GPU index.

The full 100-epoch run uses the same source config and physical GPU binding:

```bash
RSAR_ROOT=/mnt/shared/zechuan/iraod_data/RSAR IRAOD_PYTHON=/home/zechuan/miniforge3/envs/iraod/bin/python scripts/run_orthonet_rsar_source_seed42.sh --gpu 6 --port 20067
```

An optional four-GPU full run uses physical GPUs 4,5,6,7 in that exact
logical-rank order:

```bash
RSAR_ROOT=/mnt/shared/zechuan/iraod_data/RSAR IRAOD_PYTHON=/home/zechuan/miniforge3/envs/iraod/bin/python scripts/run_orthonet_rsar_source_seed42.sh --gpus 4,5,6,7 --port 20067
```

This path uses the repository's existing `torch.distributed.launch` interface
with world size 4 and `samples_per_gpu=1`. Its effective global batch remains
4, matching the one-GPU `samples_per_gpu=4` run, so the optimizer LR stays
0.005. It keeps seed 42, deterministic mode, 100 epochs, clean RSAR-only
bindings, and the same final `epoch_100.pth` selection policy. The ordered
physical mapping, distributed argv, world size, and passed invariant checks are
recorded under
`work_dirs/orthonet_rsar_source_seed42_4gpu/reproducibility/launch_plan.json`.
Preparation, resolved-config writing, and dataset-manifest creation happen once
in the parent launcher before distributed ranks start.

Only this full path accepts `epoch_100.pth` as the selected final checkpoint.
The launcher always rejects an existing `RUN_ROOT`, including smoke trees, to
prevent artifacts from separate attempts being mixed. It deliberately has no
in-place resume: retry a failed or interrupted run with a distinct directory,
for example:

```bash
RSAR_ROOT=/mnt/shared/zechuan/iraod_data/RSAR IRAOD_PYTHON=/home/zechuan/miniforge3/envs/iraod/bin/python RUN_ROOT=work_dirs/orthonet_rsar_source_seed42_retry1 scripts/run_orthonet_rsar_source_seed42.sh --gpu 6 --port 20067
```

The four-GPU default is the separate fresh
`work_dirs/orthonet_rsar_source_seed42_4gpu` tree and is forbidden from using
the one-GPU default `RUN_ROOT`. If a one-GPU run is still in progress, this
launcher never stops or modifies it. A safe owner cutover is:

1. Record the active one-GPU process, commit, `RUN_ROOT`, log, and any existing
   checkpoints, and preserve that run tree without reuse.
2. Have the run owner explicitly approve and perform termination of only that
   process; do not infer permission from starting the four-GPU launcher.
3. Confirm physical GPUs 4,5,6,7 are free and the preserved one-GPU artifacts
   remain intact.
4. Start the four-GPU command above with its fresh distinct `RUN_ROOT`.

## 3. RSAR Corruption Data and SARCLIP Patches

On a new server, first prepare the clean RSAR dataset with this layout:

```text
<RSAR_ROOT>/
  train/images/
  train/annfiles/
  val/images/
  val/annfiles/
  test/images/
  test/annfiles/
```

The project consumes corrupted RSAR images from:

```text
<RSAR_ROOT>/corruptions/<corruption>/<split>/images/
```

If you already have official or previously generated RSAR corruption images,
put them in that directory layout and skip the generation step. Otherwise,
generate the seven corruption folders used by the current configs and SARCLIP
patch workflow. The generator is migrated from the old
`/home/storageSDA1/liaojr/IRAOD` RSAR corruption pipeline.

```bash
export RSAR_ROOT=/path/to/RSAR

python tools/dataset/generate_rsar_corruptions.py \
  --data-root "${RSAR_ROOT}" \
  --splits train,val,test \
  --corruptions all \
  --workers 8 \
  --diff-samples 64
```

For a small write-through smoke test before generating the full dataset, use a
temporary RSAR root containing a few copied files under `val/images`:

```bash
python tools/dataset/generate_rsar_corruptions.py \
  --data-root /tmp/rsar_smoke \
  --splits val \
  --corruptions chaff \
  --max-images 8 \
  --workers 1 \
  --diff-samples 1
```

The supported corruption names are:

```text
chaff
gaussian_white_noise
point_target
noise_suppression
am_noise_horizontal
smart_suppression
am_noise_vertical
```

By default, the generator also creates legacy compatibility links:

```text
<RSAR_ROOT>/<split>/images-<corruption> -> <RSAR_ROOT>/corruptions/<corruption>/<split>/images
```

Pass `--no-link-legacy` if only the `corruptions/<corruption>/<split>/images`
layout is needed.

After the corrupted full images exist, build object patches for SARCLIP from
the clean RSAR annfiles and the corrupted image folders:

```bash
CORRUPTIONS=(
  chaff
  gaussian_white_noise
  point_target
  noise_suppression
  am_noise_horizontal
  smart_suppression
  am_noise_vertical
)

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
```

Main patch outputs:

```text
data/rsar_sarclip_patches/train_corrupt/metadata.csv
data/rsar_sarclip_patches/val_corrupt/metadata.csv
data/rsar_sarclip_patches/test_corrupt/metadata.csv
data/rsar_sarclip_patches/<split>_corrupt/aabb/<class_name>/*.png
```

Use `--use-corruptions 0` to build patches from clean RSAR images instead; in
that case the script reads `<RSAR_ROOT>/<split>/images`.

## 4. SARCLIP-LoRA Workflow

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
python tools/build_rsar_sarclip_patches.py --data-root "${RSAR_ROOT}" --split train --out data/rsar_sarclip_patches/train_corrupt --crop-modes aabb --crop-expands 0.4
python tools/build_rsar_sarclip_patches.py --data-root "${RSAR_ROOT}" --split val --out data/rsar_sarclip_patches/val_corrupt --crop-modes aabb --crop-expands 0.4
python tools/build_rsar_sarclip_patches.py --data-root "${RSAR_ROOT}" --split test --out data/rsar_sarclip_patches/test_corrupt --crop-modes aabb --crop-expands 0.4
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

## 5. Using the Fine-Tuned SARCLIP Adapter

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
were used during earlier debugging and may have different core dependency
versions. The current project workflow expects SARCLIP to run inside `iraod`.

## 6. Releasing Weights vs. Releasing Precomputed Scores

Publishing `lora_rsar.pth` is enough for users who can run SARCLIP. They still
need the SARCLIP code, the SARCLIP base checkpoint, and a SARCLIP-compatible
Python environment to compute rescoring results.

If users should avoid running SARCLIP entirely, release precomputed
rescoring artifacts instead. This is only valid for the exact dataset split,
corruption setting, teacher checkpoint, detector predictions, crop mode, and
class order used to generate the scores. A different detector checkpoint or
pseudo-label set requires recomputing the SARCLIP scores.

Current code supports loading the fine-tuned adapter for online SARCLIP rescoring
through `SARCLIP_LORA`. It does not yet include a detector-side cache reader for
fully precomputed CGA scores. To make detector training avoid importing SARCLIP
at runtime, add an offline score cache/export step and a matching cache lookup
path in `sfod/cga.py`.

## 7. Notes

- `iraod` is the stable environment for both detector training and SARCLIP
  rescoring.
- `cliptorch` is only a historical backup environment.
- Set `IRAOD_CONDA_PREFIX="$CONDA_PREFIX"` when running SARCLIP utilities from
  an activated `iraod` shell.
- The LoRA adapter keeps releases small, but it is not a standalone model; it
  must be loaded together with the SARCLIP base checkpoint.
