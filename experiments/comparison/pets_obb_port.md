# PETS (our reimplementation): OBB port and deviation log

Method id `PETS`, detector `PETSUnbiasedTeacher` (`sfod/rotated_pets.py`,
mechanics in `sfod/pets_dynamics.py`). Not method B.

- Paper: Liu et al., "Periodically Exchange Teacher-Student for Source-Free
  Object Detection", ICCV 2023.
- Official code: `RainbowNebula/PETS` at commit `1e87e45f7a56990bade2aba1e47667f9982b0fee`
  (detectron2, Faster R-CNN VGG16, HBB). Paths below are relative to that repo.
- Detector and source: the shared Oriented R-CNN + OrthoNet-50 used by every
  main-table method. RSAR source
  `/mnt/shared/zechuan/iraod_artifacts/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`,
  sha256 `e489f015cc2b67926a415436c97532ff814f826c4861b659cbc712fb9413b98c`.
  DIOR source: the protocol's DIOR `epoch_100.pth`.

## Official hyperparameters used

| Item | Official value | Where |
|---|---|---|
| Teacher score filter | `scores > 0.5` | `configs/UDA/C2F.yaml:30` (`BBOX_THRESHOLD`), `ubteacher/engine/trainer.py:220` |
| Static/Dynamic pairing | class-agnostic IoU `> 0.5`, all pairs, duplicates kept | `C2F.yaml:34` (`PSEUDO_THRESHOLD`), `trainer.py:251-268` |
| Fusion | WBF, `iou_thr=0.55`, `skip_box_thr=0.01`, `conf_type='max'`, weights 1/1 | `trainer.py:271-272`, `ubteacher/engine/wbf.py:153-162` |
| Fused box / score | score-weighted mean of the cluster; max score | `wbf.py:83-108`, `wbf.py:236-237` |
| Dynamic Teacher EMA | keep rate 0.9996 in warm-up, 0.999 from the first exchange on | `C2F.yaml:32`, `trainer.py:95,185`; paper §5.2 |
| EMA update interval | every iteration | paper Eq. (5); `C2B.yaml:31`, `K2C.yaml:32`, `S2C.yaml:30` (`C2F.yaml:31` uses 2) |
| Schedule | 10 epochs, warm-up 2, period 1 | `C2F.yaml:18-19`, `ubteacher/config.py:21-23` |
| Exchange rule | at epoch start if `epoch >= WARM_UP_EPOCH and epoch % Period == 0` | `trainer.py:181-185` |
| Teachers' init | Student, Static and Dynamic Teacher all from the source checkpoint | `trainer.py:165-174` |
| Evaluated model | Dynamic Teacher | paper §5.2; `trainer.py:541-544` |

## Iteration mapping (OBB/budget adaptation)

The official run has 10 epochs and 8 exchanges (start of epochs 2..9): the
first after 20% of training, then every 10%. The shared budget is 1 real
epoch of `N` iterations (global batch 32, lr 0.02), so it is cut into 10
virtual epochs and the Static Teacher and Student are exchanged before
iteration `N*k//10` (0-based), `k = 2..9`:

- RSAR, `N = 265` (8467 images / 32; checkpoint `iter_266`): before iterations
  53, 79, 106, 132, 159, 185, 212, 238 (8 exchanges; logged by the gate run).
- DIOR, `N = 184` (5863 / 32; checkpoint `iter_185`): before iterations
  36, 55, 73, 92, 110, 128, 147, 165 (8 exchanges).

`N` comes from the runner (`PETSScheduleHook`, `runner.max_iters`); training
without the hook raises.

## Differences from the official code

| # | Difference | Kind |
|---|---|---|
| 1 | Exchange is a real two-way swap (paper Eq. 2, Alg. 1). The official `swap_weight` (`trainer.py:566-577`) copies through aliased `state_dict()` views, so it ends as Static Teacher := Student with the Student unchanged (checked with a verbatim copy of the function: static 1.0, student 2.0 -> both 2.0). We follow the paper, the function's name and the method contract. | Faithfulness choice, **open question** |
| 2 | IoU for pairing and for WBF clustering is rotated IoU (`mmcv.ops.box_iou_rotated`), not HBB `box_iou` / `bb_iou_array`. | OBB adaptation |
| 3 | WBF averages `(cx, cy, w, h, angle)`. Before averaging, each box takes the equivalent representation (`w,h,a+kπ` or `h,w,a+π/2+kπ`) whose angle is nearest the cluster's top-score box; the result is put back into le90. | OBB adaptation |
| 4 | No `[0, 1]` normalisation and clipping of box coordinates before WBF (`trainer.py:256-261`, `wbf.py:41-64`): it exists only for HBB WBF. Zero-area and non-finite boxes are still skipped (`wbf.py:65-67`). | OBB adaptation |
| 5 | Teacher predictions use the shared Oriented R-CNN test config (score 0.05, rotated NMS 0.1, 2000 per image) instead of detectron2's (0.05, NMS 0.5, 100). All NMS is rotated. | OBB adaptation (shared detector) |
| 6 | Fused pseudo labels go through `UnbiasedTeacher.create_pseudo_results`, so the repo's degenerate rotated-box rejection applies. Its `>= 0.5` threshold drops nothing, because every fused score is the max of scores `> 0.5`. | Repo invariant |
| 7 | Exchange period mapped to virtual epochs of the 1-epoch run (above). | Budget adaptation |
| 8 | Optimiser and schedule are the shared ones (SGD lr 0.02, 1 epoch, global batch 32, 100-iter linear warm-up), not 8e-4 with WarmupMultiStepLR over 10 epochs at batch 8. | Shared protocol |
| 9 | Loss weights: `weight_u=1`, `use_bbox_reg=False` (RPN and ROI box regression off) instead of `UNSUP_LOSS_WEIGHT=2`, box regression weight 1 and focal loss. | Shared protocol |
| 10 | Weak/strong views are the shared strict pipelines (flip; colour jitter, grayscale, blur), not the official ones (random erasing, flip, colour). | Shared protocol |
| 11 | EMA updates every iteration; the C2F config alone updates every 2. | Faithfulness choice |
| 12 | EMA keep rates are the official per-step values, not rescaled to the shorter run. | Not adapted, **open question** |
| 13 | The Static Teacher is not saved in checkpoints; only the Student and the Dynamic Teacher (`iter_*_ema.pth`, the evaluated model) are. | Implementation |

No CGA, SARCLIP, VLST, SOGC or SLRP. No target labels are used in training or
for choosing hyperparameters.

## Deployment

Code dir `/home/zechuan/IRAOD-New-pets-20261101`: this branch plus the two
gitignored OrthoNet files every deployed main-table dir carries
(`mmdet_extension/models/backbones/orthonet.py`, `mmdet_extension/models/utils/`),
copied unchanged from `IRAOD-New-dru-20260919`.

`main` carried `StrictSourceFreeDOTADataset` but not the detector half of the
strict dataflow (commit `4c052fe`), so strict batches crashed in
`forward_train`. This branch ports it; `sfod/rotated_unbiased_teacher.py` is
byte-identical to `82e08b0` (strict dataflow plus degenerate rotated pseudo-box
rejection). RSAR cells run with
`rsar_pets_seed.sh GPU SEED DOMAIN...`; DIOR cells with
`train_d17_seed_1gpu.sh GPU PETS DOMAIN SEED`.

Tests (CPU): `python -m unittest tools.cga_research.tests.test_pets_obb -v`
from the code dir.
