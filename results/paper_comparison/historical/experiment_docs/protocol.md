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

Legacy seed42 report (unchanged, not the final multi-seed report):
`results/paper_comparison/IRAOD_full_comparison_report_cn.docx`.
Its DIOR per-class table has 400 rows, visualizations 288 file records, and
RoI 33 directory records (528 feature files). Those legacy RoI features are
pre-NMS without row mapping and are not valid class-colored instance features;
these historical counts do not describe the new full-test evidence below.

IRG/LPLD/DRU/AASFOD/SF-YOLO remain N/A, with concrete port boundaries in `results/paper_comparison/dior_ports_executability.md`; Simple-SFOD/SF-UT is covered by equivalent B on both datasets.

Aligned qualitative/RoI completion tooling is documented in
[`RESULT_COMPLETION.md`](RESULT_COMPLETION.md). Its v3 full-TEST outputs are
separate from legacy unaligned and v2 subset exports. Seed42 RSAR/DIOR source, final-EMA and
Student qualitative coverage is now independently validated as complete;
no legacy/subset artifact was promoted based on planning.

### Validated qualitative integration, 2026-09-07

The CPU streaming consumer validated all **132 groups / 1,267,816 image-role
NPZs / 18,468,211 detection rows**, all **3,520 frozen-subset visualizations**
and **24 full-ROI-source joint embeddings**. The authoritative detection total
is the actual row sum, correcting the earlier manual estimate by 7,000.
The embeddings contain 141,984 sampled points, not 18 million full instances;
their fixed settings are shared across all 24 comparisons.

Small versioned evidence is under `results/paper_comparison/`:
`qualitative_full_test_summary.json`, `qualitative_roi_groups.csv`,
`qualitative_embedding_index.csv` and `qualitative_report_input.json`.
The full per-image CSV, NPZs, images and embedding binaries remain remote at
the paths recorded in the summary. Collection/integration commands are in
[`QUALITATIVE_INTEGRATION.md`](QUALITATIVE_INTEGRATION.md).

The 180 missing RSAR seed42 non-chaff B-F AP rows are now appended to
`per_class_summary.csv` (78 -> 258 rows). Original rows and exact canonical
mAP strings are unchanged; new class AP values retain the three-decimal
precision of their actual class tables. Per-group source paths are in
`rsar_nonchaff_per_class_evidence.json`. This table covers all seven corruption
A-F comparisons plus source clean, not a completed multi-seed/clean-B-F table.

**Quantitative completion and the final all-item report remain pending.**
The coordinator's 08:31 UTC progress report was 58/130 training runs and
85/192 native-ID evaluations; this qualitative collection did not re-audit or
upgrade those counts. No final multi-seed CSV/DOCX or all-eight completion is
claimed, and the existing quantitative report is not overwritten.

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
