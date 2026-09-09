# AASFOD OBB implementation — pre-result freeze

Independent implementation of Chu et al., *Adversarial Alignment for Source Free
Object Detection*, AAAI2023, DOI
[10.1609/aaai.v37i1.25119](https://doi.org/10.1609/aaai.v37i1.25119).
Source reference: `ChuQiaosong/AASFOD@c383cacdc050b804c95a6127d994cb1eba0596df`.
No unlicensed author source is vendored. This is the approved **post-hoc-dropout,
budget-rescaled OBB port**, not the original source-trained-dropout experiment.
No target GT, new source training, source-checkpoint mutation, or core-method
changes. The existing12 domains and adaptation seeds42/43/44 remain fixed.

## Frozen algorithm choice

- **TSD chooses released-code equations**, not paper Eq2–3:20 stochastic RoI
  passes on one deterministic source feature/RPN proposal set; complete class
  probabilities and unchanged five regression deltas; sum over proposals of
  class population-variance sum × delta population-variance sum. No entropy,
  decoded-angle variance, proposal reranking, or21st duplicated sample.
- Insert stateless p=.5 dropout after each existing shared FC/ReLU only during
  TSD. Ordinary teacher/student detection remains unchanged. This dropout was
  **not source-trained**, so this is not a source-trained posterior.
- Sort variance ascending, with filename tie breaking. Top `N//5` are
  source-similar; all others source-dissimilar (sigma=.8, repaired integer
  cardinality). Both are target VAL images. All-tied/partial/smoke results cannot
  drive a formal run. Run TSD separately for each adaptation seed; no seed/domain
  pooling. References: `select_by_uncertainty.py:169–221`, paper pp454–456.
- Alignment: similar **strong** detection using similar **weak** teacher labels;
  dissimilar **weak** features receive domain losses only. Existing OrthoNet
  shallow layer1 C256 and deep layer4 C2048 before FPN. Preserve the source's
  frozen layer1 and BN policy even though that limits local backbone gradients.
  Local256→256→128→1 pointwise sigmoid head; global2048→512→128→128
  stride2 Conv/BN/ReLU/dropout head, GAP,2-way classifier. GRL1; gamma3 focal
  global loss and squared local loss; domain similar0/dissimilar1; each of four
  terms has the code's .5 factor; lambda1. No inherited HTCN attention/context.
  References: `vgg16_adv.py:28–89`, `faster_rcnn_adv.py:285–310`,
  `trainval_adv_mt.py:341–406`.
- FNS: teacher predicts **four originals before mosaic**; common strong
  photometric views share weak-view flip geometry. Mean-image padded crop shifts
  ±20%; cuts20–80%; anisotropic resize back to canvas size; select shifted
  quadrants. Transform OBB corners, clip to source crop then quadrant, discard
  zero-area polygons, and project to canonical le90 min-area rectangles.
  This projection is an explicit OBB approximation, not exact rotation-preserving
  scaling. Bilinear tensor resampling replaces the original PIL resampler;
  normalized floating pixels avoid uint8 quantization. Only four standard
  detection losses; **no FNS adversarial loss**. References:
  `minibatch_mosaic.py:117–240,306–366`, `trainval_adv_mosaic_mt.py:233–439`.
- Both stages keep full RPN cls/box and RoI cls/box pseudo supervision. Common
  score threshold.7 and rotated NMS.1 override original code threshold.8/NMS.3
  and first10 per class. No class-specific regression head or HBB replacement.

## Exact common-budget mapping

One common padded epoch means `ceil(N/32)` **optimizer updates**.185/266 are
the common epoch-end checkpoint filename labels, not actual update counts.

| Dataset | VAL N | Total | Alignment | FNS | EMA interval |
|---|---:|---:|---:|---:|---:|
| DIOR |5863|184|110|74|1|
| RSAR |8467|265|159|106|1|

Stages are floor(3U/5) and the remainder, from README30 alignment +20 FNS epochs.
Reference duration: author batch1 ×10000 steps ×50 epochs =500000 updates.
Scale2500-step cadence by U/500000, nearest integer with minimum1; retain
momentum.99. Update strictly after a completed optimizer update, no step0 and no
unconditional final flush. **This compressed EMA changes original dynamics**;
it is not claimed numerically equivalent to the long author schedule.

Single GPU, global32 **original image inputs**, LR.02:
alignment16 similar +16 dissimilar (16 detection examples), FNS32 originals
→8 canvases. SGD momentum.9, wd.0001. Preserve the common initial100-update
linear warmup (.001 ratio) once in alignment; FNS starts at.02 with no repeated
warmup. No within-one-epoch decay. Each stage gets a fresh native runner and
optimizer; FNS loads alignment **student**, never alignment teacher; copy its
complete detector state including frozen parameters/buffers into the new
teacher before iteration1. Teacher EMA updates trainable detector parameters;
frozen parameters/buffers remain the copied source state.

TSD adds20×N RoI passes (one backbone/RPN per image) **per seed** outside the
detector optimizer budget. Alignment adds the dissimilar backbone branch;
FNS teacher processes32 originals while student processes8 canvases. These
costs are disclosed, not counted as free or equated to32 mosaic canvases.

## Queue and smoke interfaces (commands, not execution evidence)

CPU preparation uses the same accepted source/report binding as other ports:

```bash
python -m experiments.comparison.extension_training prepare \
  --base-plan BASE.json --core-report CORE_REPORT.json --core-paths CORE/paths.py \
  --out-dir NEW_METADATA --artifact-root NEW_ARTIFACTS \
  --eval-code NATIVE_331d_CHECKOUT --python /existing/env/bin/python --method AASFOD
```

This writes36 cells and `aasfod_tsd_commands.txt`, and **does not launch TSD or
create formal artifacts**. Each command binds exact source checkpoint/VAL/seed.
Under the parent's existing GPU-lock owner and unchanged native environment:

```bash
# Prerequisite; CUDA_VISIBLE_DEVICES selects the owner's already-locked GPU.
python -m experiments.comparison.aasfod_tsd \
  --queue NEW_METADATA --dataset DIOR --domain cloudy --seed 42
# Optional separate TSD smoke: --smoke-images 5 writes tsd_smoke.json, not tsd.json.

# Optional two-stage training smoke AFTER real TSD; distinct smoke_work artifacts:
python -m experiments.comparison.train_aasfod \
  --queue NEW_METADATA --dataset DIOR --domain cloudy --seed 42 --smoke-steps 1

# Formal queue dispatch (existing finite worker/with_gpu_lock.sh owns the lock):
python -m experiments.comparison.extension_training train \
  --queue NEW_METADATA --gpu 4 --dataset DIOR --domain cloudy --seed 42 --method AASFOD
```

Both GPU modules require `IRAOD_GPU_LOCKED=1`; they acquire no new lock and modify
no environment. Smoke results never satisfy the formal TSD or detector outputs.
TSD replaces only its in-memory shared pipeline with image loading followed by
metadata-only `flip=False, flip_direction=None`, and disables its strong pipeline.
The bound weak resize/normalize/pad/format/Collect chain stays intact. This fixes
the missing `Collect` flip metadata without any random-flip transform or RNG
consumption; the frozen dataset/config and source/target path binding do not
change. The existing prerequisite recipe and arguments remain valid once the
runtime owner selects the corrected `experiments.comparison.aasfod_tsd` entry
from the delivered checkout (not the old entry in frozen training code).
No new TSD, training, ROI, or Oracle execution is authorized by this correction.

`train_aasfod` uses native `train.py` twice: `work/alignment/iter_110.pth`
(DIOR) initializes both FNS detectors; FNS produces `work/fns/iter_74{,_ema}.pth`.
The completed final states are copied to common `work/iter_185{,_ema}.pth`
(RSAR uses159/106 and266). `stages.json` records stage-local iterations,
aggregate budget, commands and completed status; final file renaming does not
claim an additional update. Final evaluation remains the unchanged331d native
evaluator and final teacher role, with student diagnostic artifact retained.

### Resolved stage config import and retained-failure recovery

MMCV 1.x `Config.dump()` writes instantiated transforms as bare
`ColorJitter(...)` constructors without retaining their source imports.
The original inherited AASFOD config loads successfully, but native
`Config.fromfile(work/alignment.py)` then raises
`NameError: name 'ColorJitter' is not defined`. `train_aasfod` now writes the
same resolved text with an explicit `torchvision.transforms.ColorJitter`
import and deletes the imported name afterward so it is not a config field.
Both alignment and FNS preserve the real operator class, parameter ranges,
pipelines and all resolved stage settings. No frozen config, model checkout,
host-binding function or training protocol is changed.

After integration, select the delivered orchestration checkout's
`experiments/comparison/train_aasfod.py` entry, while keeping `training_code`
bound to `integration_code_36c2053` and the existing cell config/source/TSD
bindings. Do not execute the old helper from the frozen checkout or reuse its
already-generated `alignment.py`. Hostname/path binding for a different host
is a separate integration concern; this correction adds no host authorization.

For the runtime owner's observed pre-training import failure, inspect and
retain the actual failed layout before retrying. Generic finite
`archive_retry` deliberately refuses AASFOD; it remains unchanged.
The outer `extension_training._train` requires `cell.work_dir` not to exist.
The inner `train_aasfod.run` allows the work root but refuses an existing
`work/alignment` or `work/fns`; it otherwise rewrites stage configs and
`stages.json`. The outer entry rewrites `execution.json` and `train.log` and
appends `terminal_status`. Finite wrapper status can also retain the failure
in the mixed and originating queue.

Owner-specific preserved retry therefore needs the exact cell's failed work,
execution/log/terminal and wrapper evidence retained without overwriting it,
a fresh native work destination, and reconciliation of that cell's finite
failure state. Confirm there are no successful alignment/FNS checkpoints,
final teacher/student checkpoints or retained evaluations before treating it
as a pre-training retry; if any exist, do not restart from this recipe.
Keep the valid `tsd.json` at its existing binding, unchanged (including the
RSAR 8467-image, 1693/6774 partition and completed 20-pass evidence).
Do not archive the whole method directory if that would move the split, refit
TSD, clear global finite state, delete logs or weaken retry guards.
This is a recovery contract for the runtime owner, not a recovery command or
authorization to launch. Exact remote layout inspection and recovery are not
performed by the code fix.

## Narrow CPU validation

The stage import regression is a non-skipping CPU gate:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest -v \
  tools.tests.test_aasfod_config_import
```

It reproduces the exact NameError from plain MMCV dumping, then exercises the
actual two-stage entry with real recursive config loads and generated-Python
imports for RSAR/DIOR, with and without `native_config_paths`. Full resolved
config comparisons and real torchvision class/range assertions cover both
stages. Custom detector registration is disabled; only subprocess training is
replaced with CPU config reloads and explicitly synthetic checkpoint files.
Partial TSD, native failure and existing-stage refusal remain covered.
These tests establish config/import correctness, not full-detector imports,
formal GPU training completion or new TSD results.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m unittest \
  tools.tests.test_aasfod_mechanisms tools.tests.test_aasfod_native \
  tools.tests.test_extension_training
```

Mechanism tests run without mmdet/mmrotate; native tests skip when those packages
are absent. Native tests exercise registry/configs, actual image-only dataset
partition consumption, real detector-loss routing (using tiny CPU feature/RPN/
RoI seams), pre-mosaic teacher ordering, loss/backward separation and hook order.
They are not evidence that native full-detector/GPU training or TSD has run.

The deterministic TSD sampling regression is a separate, non-skipping CPU gate:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest -v tools.tests.test_aasfod_tsd_pipeline
```

It resolves both RSAR (6-class) and DIOR (20-class) configs, loads tiny real PNGs
through the actual `StrictSourceFreeDOTADataset` and installed MMDetection/
MMRotate loading, `RResize`, normalization, padding, formatting and `Collect`
modules, and exercises the real TSD entry/score loop. Both families fail at
`Collect` with `KeyError('flip')` before the fix. Assertions cover all required
metadata, exact unflipped pixel geometry, repeated-view and RNG invariance,
image-only reads, subset order, frozen checkpoint/BN state, full class and delta
variance, and exactly 20 stochastic ROI passes per image.

This gate requires installed `mmcv` 1.x, `mmdet` 2.x, `mmrotate` 0.3.x,
`pycocotools`, PyTorch and torchvision. To run without compiled `mmcv._ext`,
the test uses isolated registries and complete original pipeline modules;
unused native-op/annotation entry points fail if called. `Collect`, dataset
sampling, collate/CPU scatter and the TSD mechanism are not replaced. Only the
detector/build/CUDA boundary is a tiny CPU fixture and custom model imports are
disabled while resolving configs. This verifies the observed sampling-chain
failure, not native full-detector imports, GPU execution or formal TSD results.
