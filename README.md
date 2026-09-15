# IRAOD-New

This repository contains the MMRotate-based oriented detector, source-free
adaptation experiments, OrthoNet backbones, CGA/SARCLIP scoring, and the VLST
feature-level semantic teacher.

The repository root in the current workspace is:

```bash
cd /mycode/mycode/IRAOD-New
```

The training entry point is the root-level `train.py`. There is no
`tools/train.py` in this repository.

## Environment

Use the `iraod` environment for detector training, evaluation, and SARCLIP:

```bash
conda activate iraod
cd /mycode/mycode/IRAOD-New
export IRAOD_CONDA_PREFIX="$CONDA_PREFIX"
```

The detector stack is based on the versions in `requirements.txt`. SARCLIP
also needs the following packages in the same environment:

```bash
pip install "transformers<4.56" timm safetensors peft
```

Do not mix the old `cliptorch` environment into `iraod`.

## RSAR

### Dataset layout

The RSAR configurations use `/myfile/dataset/RSAR/` by default. The required
layout is:

```text
/myfile/dataset/RSAR/
  train/
    images/
    annfiles/
  val/
    images/
    annfiles/
  test/
    images/
    annfiles/
  corruptions/
    chaff/{train,val,test}/images/
    gaussian_white_noise/{train,val,test}/images/
    point_target/{train,val,test}/images/
    noise_suppression/{train,val,test}/images/
    am_noise_horizontal/{train,val,test}/images/
    smart_suppression/{train,val,test}/images/
    am_noise_vertical/{train,val,test}/images/
```

The clean `train` split has oriented annotations and is the source-domain
data. The corrupted `val` split is used as unlabeled target data during
adaptation. The corrupted `test` split is used for evaluation. Corrupted
images reuse the clean split annotations only for evaluation and diagnostics;
the adaptation loss does not use target annotations.

The RSAR detector classes are:

```text
ship, aircraft, car, tank, bridge, harbor
```

The seven supported corruption values are:

```text
chaff
gaussian_white_noise
point_target
noise_suppression
am_noise_horizontal
smart_suppression
am_noise_vertical
```

### Clean RSAR OrthoNet Source Baseline

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

### Source-free VLST training

The main OrthoNet + VLST configuration is:

```text
configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py
```

It starts from the clean-source checkpoint
`/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth`, uses
`weight_l=0` and `weight_u=1`, and loads the SARCLIP-LoRA adapter configured in
the file. For example:

```bash
cd /mycode/mycode/IRAOD-New

CUDA_VISIBLE_DEVICES=0 python train.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py \
  --work-dir work_dirs/vlst_rsar_noise_suppression \
  --seed 42 \
  --deterministic \
  --cfg-options \
    corrupt=noise_suppression \
    data.samples_per_gpu=8 \
    runner.max_epochs=1 \
    evaluation.interval=1 \
    checkpoint_config.interval=1 \
    log_config.interval=10
```

Replace `noise_suppression` with any supported RSAR corruption. For the
label-level CGA + VLST arm, use:

```text
configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_rsar_orthonet.py
```

For the self-training baseline without VLST, use:

```text
configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet.py
```

### RSAR testing

Use the EMA teacher checkpoint for the main reported result. Use `latest.pth`
when explicitly comparing the student checkpoint.

```bash
cd /mycode/mycode/IRAOD-New

CUDA_VISIBLE_DEVICES=0 python test.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_rsar_orthonet.py \
  work_dirs/vlst_rsar_noise_suppression/iter_1060_ema.pth \
  --eval mAP \
  --show-dir /myfile/experiment/iraod/vis_dir/rsar_noise_suppression \
  --cfg-options corrupt=noise_suppression
```

`--show-dir` saves painted detection images. It does not add ground-truth
boxes; use `tools/analysis/visualize_gt_pred.py` when a side-by-side GT and
prediction view is required.

### SARCLIP adapter

The RSAR VLST/CGA configurations expect:

```text
/myfile/pretrain/SARCLIP/ViT-B-32/vit_b_32_model.safetensors
/myfile/mycode/IRAOD-New/work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth
```

If the adapter is stored elsewhere, set the corresponding `SARCLIP_LORA` or
`vlst_lora_path` value in the selected configuration before training/testing.

## DIOR

### Dataset layout

The DIOR configurations use `/myfile/dataset/DIOR/` by default:

```text
/myfile/dataset/DIOR/
  JPEGImages-trainval/
  JPEGImages-test/
  ImageSets/Main/
    train.txt
    val.txt
    test.txt
  Annotations/
    Oriented Bounding Boxes/
  Corruption/
    JPEGImages-brightness/
    JPEGImages-cloudy/
    JPEGImages-contrast/
    JPEGImages-defocus_blur/
    JPEGImages-elastic_transform/
    JPEGImages-fog/
    JPEGImages-frost/
    JPEGImages-gaussian_blur/
    JPEGImages-gaussian_noise/
    JPEGImages-glass_blur/
    JPEGImages-impulse_noise/
    JPEGImages-jpeg_compression/
    JPEGImages-motion_blur/
    JPEGImages-pixelate/
    JPEGImages-saturate/
    JPEGImages-shot_noise/
    JPEGImages-snow/
    JPEGImages-spatter/
    JPEGImages-speckle_noise/
    JPEGImages-zoom_blur/
```

The DIOR VLST configuration is:

```text
configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_dior.py
```

It reads clean source images from `JPEGImages-trainval`, corrupted target
images from `Corruption/JPEGImages-${corrupt}`, and oriented XML annotations
from `Annotations/Oriented Bounding Boxes`.

The DIOR classes are the 20 classes defined in the configuration:

```text
airplane, airport, baseballfield, basketballcourt, bridge, chimney, dam,
expressway-service-area, expressway-toll-station, golffield,
groundtrackfield, harbor, overpass, ship, stadium, storagetank, tenniscourt,
trainstation, vehicle, windmill
```

### Source-free VLST training

The DIOR VLST configuration sets `model.cfg.weight_l=0` and
`model.cfg.weight_u=1`, so it is source-free adaptation. First train the
clean OrthoNet source detector, then initialize both the VLST Student and EMA
teacher from its checkpoint. The commands below use batch size 8, a
conservative setting for a 40 GB GPU.

In this project's SFOD split, `ImageSets/Main/train.txt` is the labeled source
domain (5,862 images), `val.txt` is reserved as the unlabeled target split,
and `test.txt` is the held-out evaluation set. Do not merge `val.txt` into
source training.

```bash
cd /mycode/mycode/IRAOD-New

# Step 1: clean DIOR source detector (100 epochs; random OrthoNet is unfrozen).
CUDA_VISIBLE_DEVICES=0 python train.py \
  configs/baseline/oriented_rcnn_orthonet_dior.py \
  --work-dir work_dirs/orthonet_dior_source_seed42 \
  --seed 42 \
  --deterministic \
  --cfg-options \
    data.samples_per_gpu=8 \
    optimizer.lr=0.01 \
    runner.max_epochs=100 \
    evaluation.interval=1 \
    checkpoint_config.interval=1

# Step 2: VLST + OrthoNet adaptation to cloudy (one epoch).
CUDA_VISIBLE_DEVICES=0 python train.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_dior.py \
  --work-dir work_dirs/vlst_dior_cloudy_seed42 \
  --seed 42 \
  --deterministic \
  --cfg-options \
    corrupt=cloudy \
    load_from=/mycode/mycode/IRAOD-New/work_dirs/orthonet_dior_source_seed42/epoch_100.pth \
    model.ema_ckpt=/mycode/mycode/IRAOD-New/work_dirs/orthonet_dior_source_seed42/epoch_100.pth \
    data.samples_per_gpu=8 \
    optimizer.lr=0.005 \
    runner.max_epochs=1 \
    evaluation.interval=1 \
    checkpoint_config.interval=1 \
    log_config.interval=10
```

For another corruption, replace both `corrupt=cloudy` and
`work_dirs/vlst_dior_cloudy_seed42` with the same corruption name. The source
checkpoint must be passed to both `load_from` and `model.ema_ckpt`, so the
Student and EMA teacher begin from identical clean OrthoNet weights. With
batch size 12, use `data.samples_per_gpu=12`, `optimizer.lr=0.015` for source
training, and `optimizer.lr=0.0075` for VLST adaptation.

If a labeled adaptation ablation is desired, override
`model.cfg.weight_l=1.0`; that is a different protocol from source-free
adaptation.

### Fully supervised DIOR-R baseline

For the reported Oriented R-CNN R50-FPN baseline, use the complete labeled
`train + val` split and evaluate on `test`. Create the combined list once:

```bash
cat /myfile/dataset/DIOR/ImageSets/Main/train.txt \
    /myfile/dataset/DIOR/ImageSets/Main/val.txt \
    > /myfile/dataset/DIOR/ImageSets/Main/trainval.txt
```

The strict ResNet-50 reproduction is configured in
`configs/baseline/oriented_rcnn_r50_dior_trainval.py` (12 epochs, SGD
`lr=0.0025`, decay at epochs 8 and 11):

```bash
cd /mycode/mycode/IRAOD-New

CUDA_VISIBLE_DEVICES=1 python train.py \
  configs/baseline/oriented_rcnn_r50_dior_trainval.py \
  --work-dir work_dirs/oriented_rcnn_r50_dior_trainval \
  --seed 42 \
  --deterministic \
  --cfg-options \
    data.samples_per_gpu=2 \
    optimizer.lr=0.0025 \
    runner.max_epochs=12 \
    evaluation.interval=1 \
    checkpoint_config.interval=1
```

The OrthoNet replacement is a separate supervised comparison, configured in
`configs/baseline/oriented_rcnn_orthonet_dior_trainval.py`. It is not expected
to reproduce 63.62 exactly because it starts from a randomly initialized
OrthoNet rather than an ImageNet-pretrained ResNet:

```bash
cd /mycode/mycode/IRAOD-New

CUDA_VISIBLE_DEVICES=1 python train.py \
  configs/baseline/oriented_rcnn_orthonet_dior_trainval.py \
  --work-dir work_dirs/oriented_rcnn_orthonet_dior_trainval \
  --seed 42 \
  --deterministic \
  --cfg-options \
    data.samples_per_gpu=8 \
    optimizer.lr=0.01 \
    runner.max_epochs=100 \
    evaluation.interval=1 \
    checkpoint_config.interval=1
```

The CGA-only DIOR configuration is:

```text
configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_cga.py
```

### DIOR testing

```bash
cd /mycode/mycode/IRAOD-New

CUDA_VISIBLE_DEVICES=0 python test.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_dior.py \
  work_dirs/vlst_dior_cloudy/iter_1060_ema.pth \
  --eval mAP \
  --show-dir /myfile/experiment/iraod/vis_dir/dior_cloudy \
  --cfg-options corrupt=cloudy
```

For a student-checkpoint comparison, replace `iter_1060_ema.pth` with
`latest.pth`. The EMA checkpoint is preferred for the main result.

## Data that is not required

For the current RSAR/DIOR detector workflows, do not upload these unused or
redundant paths:

```text
DIOR/JPEGImages/                         # link stubs in this workspace
DIOR/Annotations/Horizontal Bounding Boxes/
DIOR/Layout/
DIOR/Segmentation/
DIOR/labels/
```

For RSAR, the detector only needs the clean `images`/`annfiles` pairs and the
corrupted image folders used by the selected experiments. SARCLIP patch caches,
`work_dirs`, visualizations, and evaluation JSON files are derived artifacts;
they can be omitted and regenerated unless exact previous outputs must be
preserved.

## Checkpoints and outputs

Typical training outputs include:

```text
work_dirs/<experiment>/latest.pth
work_dirs/<experiment>/iter_*.pth
work_dirs/<experiment>/iter_*_ema.pth
```

`latest.pth` is the regular student checkpoint. `iter_*_ema.pth` is the EMA
teacher checkpoint and is the default choice for final evaluation.
