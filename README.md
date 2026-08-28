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
`model.cfg.weight_u=1`, so it is source-free adaptation. A DIOR source
checkpoint must be supplied with `load_from`; this workspace does not assume a
fixed DIOR checkpoint path.

```bash
cd /mycode/mycode/IRAOD-New

CUDA_VISIBLE_DEVICES=0 python train.py \
  configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_dior.py \
  --work-dir work_dirs/vlst_dior_cloudy \
  --seed 42 \
  --deterministic \
  --cfg-options \
    corrupt=cloudy \
    load_from=/absolute/path/to/dior_source_checkpoint.pth \
    data.samples_per_gpu=8 \
    runner.max_epochs=1 \
    evaluation.interval=1 \
    checkpoint_config.interval=1 \
    log_config.interval=10
```

Replace `cloudy` with any `JPEGImages-<corruption>` directory listed above.
If a labeled adaptation ablation is desired, override
`model.cfg.weight_l=1.0`; that is a different protocol from source-free
adaptation.

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
