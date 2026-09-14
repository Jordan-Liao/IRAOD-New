# DIOR-R leakage / protocol audit

- code: 0f98a48b539f9260055fc305784ffbc924f50451
- source: /mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth SHA256 74d7eafdc547976a86f8f35b2652dba384948e33ab5c15bb908fa3e54025782a
- train unlabeled: StrictSourceFreeDOTADataset on val-only symlink dirs (5863). Empty gt_bboxes/gt_labels in dataset.
- weight_l=0, weight_u=1, use_bbox_reg=False, --no-validate, seed 42 deterministic.
- C: CLIP RN50x64 legacy, no SARCLIP_LORA. D/E/F: frozen SARCLIP ViT-B-32, LoRA unset.
- test GT only in test.py --eval mAP with ImageSets/test.txt + Oriented Bounding Boxes XML.
- class order: DIORDataset.CLASSES (expressway before dam, lowercase).
- JPEGImages-clean is a dangling symlink, not a corruption; excluded.
- GPU7 briefly occupied by unrelated rsi-pr3-qwen3 during E/F; E/F used physical GPU4-6 only. Not a scientific change.

## Local integration evidence (2026-09-06)

- 20 original eval JSON values agree with the raw CSV: 18 corruption cells
  (A source, B–F final EMA) and source clean val/test.
- `dior_per_class.csv`: 400 rows = 20 evaluations × 20 classes; AP50 per class
  is rounded to three decimals, unlike the full-precision aggregate mAP.
- 288 visualizations = 3 corruptions × 6 methods × the same 16 image IDs.
  `vis16.txt` matches the first 16 IDs of `ImageSets/test.txt` (11726–11741).
  Each directory has `vis_exit=0`; shared show threshold is 0.3.
- 33 RoI groups = 3 × (A source + B–F EMA/Student), each with `roi_exit=0`,
  16 index records and 16 existing NPY files. A's `ema` directory means source.
- Extraction uses `fc_cls` input, but features are pre-NMS while saved `preds`
  are post-NMS without row mapping. Predicted labels/boxes cannot be assigned
  directly to feature rows; the original per-instance alignment requirement
  is not fully met. No claim of ready-to-use class-colored t-SNE.
- Only small metadata and execution snapshots were copied. Remote logs,
  checkpoints, images and features were not added to git.
