# IRAOD strict Source-Free comparison protocol

## Claim
Fair OBB Source-Free comparison of A–F on RSAR corruptions (and DIOR-R when staged) with a shared OrthoNet-50 + Oriented R-CNN source, final-EMA evaluation, no target GT in adaptation/selection.

## Frozen scientific identity
- Code: `0f98a48b539f9260055fc305784ffbc924f50451`
- Source: `/mnt/shared/zechuan/iraod_artifacts/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`
- SHA256: `e489f015cc2b67926a415436c97532ff814f826c4861b659cbc712fb9413b98c`
- Seed 42, deterministic, 1 epoch unlabeled VAL image-only (8467), effective global batch 32, LR 0.02
- B/C/D: 1×32; E/F: 2×16 (topology-only after single-GPU OOM)
- `weight_l=0`, `weight_u=1`, `use_bbox_reg=False` (SFOD recipe; unlabeled bbox loss zeroed by design)
- Final checkpoint: last-iteration EMA Teacher; no test-set selection
- Eval: `test.py configs/baseline/oriented_rcnn_orthonet_rsar.py` `--eval mAP` (DOTA le90 AP50), TEST images of that corruption + `RSAR/test/annfiles/`

## Methods
| ID | Name | Train | Notes |
| A | Source Only | no | same source on clean and each corruption TEST |
| B | ST / Mean Teacher | yes | CGA off, VLST off |
| C | CLIP-CGA | yes | vanilla CLIP RN50x64, legacy CGA |
| D | SARCLIP-CGA | yes | frozen base SARCLIP, veto_soft, no LoRA |
| E | VLST | yes | CGA off, independent frozen SARCLIP VLST |
| F | CGA+VLST | yes | D's CGA + E's VLST; F cell used preflight workdir after full budget |

## Metrics
- mAP = VOC-style AP50 on 6 RSAR classes
- Δada = mAP_method − mAP_A (same corruption)
- recovery = (mAP_m − mAP_A) / (mAP_A_clean − mAP_A) if A_clean > A_corr else N/A
- mPC = mean over completed corruptions of a method; incomplete → N/A
- rPC = mPC / A_clean × 100%
- seed=42 only → mean±std N/A until more seeds

## Status
RSAR 7 corruptions × A–F final-EMA seed=42 completed (42 cells + A clean TEST). Single seed; mean±std N/A until more seeds. Numbers taken from gpu-67 eval jsons under `/mnt/shared/zechuan/iraod_artifacts/comparison/rsar/*/seed_42/`. DIOR-R staging incomplete on `.67`. IRG/LPLD not started.
