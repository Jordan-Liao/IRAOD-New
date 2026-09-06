# IRAOD strict Source-Free comparison protocol

## Claim
Fair OBB Source-Free comparison of A–F on RSAR and DIOR-R corruptions with a shared OrthoNet-50 + Oriented R-CNN source within each dataset, final-EMA evaluation, no target GT in adaptation/selection.

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
RSAR 7 corruptions × A–F seed=42 completed (42 cells + A clean TEST), unchanged from b852746. DIOR-R brightness/cloudy/contrast × A–F completed (18 cells + A clean val/test). A is source-only; B–F are final EMA. Single seed; mean±std N/A. RSAR eval jsons remain under `/mnt/shared/zechuan/iraod_artifacts/comparison/rsar/*/seed_42/`.

DIOR uses its own `epoch_100.pth` source (SHA256 and paths in `results/paper_comparison/dior_checkpoint_manifest.json`), 5863 image-only val inputs, one epoch / 185 iterations, global batch 32 and LR 0.02. B–D: 1×32; E/F: 2×16. Test has 11738 images and 20 classes. Clean val/test are 0.40987735986709595 / 0.2948998510837555; rPC uses TEST, not val.

Final report: `results/paper_comparison/IRAOD_full_comparison_report_cn.docx`.
DIOR per-class has 400 rows; visualizations have 288 file records; RoI has 33 directory records (528 feature files). RoI features are pre-NMS while prediction metadata is post-NMS without row mapping: class-colored instance t-SNE is not yet justified. RSAR per-class/qualitative export coverage is not equivalent to its completed 42-cell metrics.

IRG/LPLD/DRU/AASFOD/SF-YOLO remain N/A, with concrete port boundaries in `results/paper_comparison/dior_ports_executability.md`; Simple-SFOD/SF-UT is covered by equivalent B on both datasets.

Aligned qualitative/RoI completion tooling is documented in
[`RESULT_COMPLETION.md`](RESULT_COMPLETION.md). Its v3 full-TEST outputs are
separate from legacy unaligned and v2 subset exports. Seed42 RSAR/DIOR source, final-EMA and
Student coverage and joint t-SNE remain pending remote extraction evidence;
planning does not upgrade the result manifests above.

The separate CPU multi-seed report consumer is documented in
[`REPORT_CONSUMER.md`](REPORT_CONSUMER.md). It keeps fixed-source seed-block
statistics, full-TEST predictions and full-TEST aligned RoIs distinct from the
frozen 32/16-image visualization subset and sampled embeddings; sampling never
substitutes for complete per-instance exports or all requested items.

Prediction identity is now captured in the same inference loop and saved next
to each new `--out` pickle; see
[`PREDICTION_IMAGE_ORDER.md`](PREDICTION_IMAGE_ORDER.md). An independent evaluation
checkout records its own SHA separately from the checkpoint's training SHA.
Old pickles without native sidecars remain unverified; live training/runtime
files are not changed by this delivery.
