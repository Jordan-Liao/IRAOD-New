# IRAOD strict Source-Free comparison protocol (revised)

## Claim
Fair OBB Source-Free comparison of A–F and OURS (本文方法 strict, DIOR-C only) plus IRG, LPLD, SFUT, **DRU**, AASFOD, and SFYOLO on RSAR corruptions and DIOR-C. Shared OrthoNet-50 + Oriented R-CNN source, final-EMA Teacher TEST, no target GT in adaptation/selection. **DRU is required** (DRUUnbiasedTeacher; not B / Mean Teacher).

## Frozen scientific identity
- RSAR source: `/mnt/shared/zechuan/iraod_artifacts/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`
- DIOR source: `/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`
- Source recipes (resolved config + command, copied from each run's `reproducibility/`; RSAR code at b474aaa, DIOR at 0f98a48): `experiments/comparison/rsar_source/` (+ `launch.sh`), `experiments/comparison/dior_source/`. `configs/baseline/oriented_rcnn_orthonet_rsar.py` on main is not the RSAR source recipe (see `rsar_source/`).
- Seed 42 on new DIOR-17 cells; existing 42/43/44 cells reused when valid
- `weight_l=0`, `weight_u=1`, `use_bbox_reg=False`
- Final checkpoint: last-iteration EMA Teacher; no test-set selection
- Required TEST role: **ema** only

## Methods (main table)
| ID | Name | Train | Notes |
| A | Source Only | no | same source on clean and each corruption TEST |
| B | ST / Mean Teacher | yes | UnbiasedTeacher; CGA off, VLST off |
| C | CLIP-CGA | yes | vanilla CLIP RN50x64, legacy CGA |
| D | SARCLIP-CGA | yes | frozen base SARCLIP, veto_soft, no LoRA |
| E | VLST | yes | CGA off, independent frozen SARCLIP VLST |
| F | CGA+VLST | yes | D's CGA + E's VLST |
| OURS | 本文方法 (strict) | yes | F with CLIP RN50x64 (CGA + VLST) instead of SARCLIP; method knobs of sogc-orthonet-phase1 @ 8b42687 `..._vlst_cga_dior_orthonet.py`; DIOR only, DDP2 × 16 |
| IRG | IRG-OBB | yes | required; rerun NaN cells once |
| LPLD | LPLD-OBB | yes | required; rerun NaN cells once |
| SFUT | SFUT-OBB | yes | required; not B |
| DRU | DRU-OBB | yes | required; Dynamic Retraining-Updating + Historical Student Loss; RSAR 1×32, 1 epoch (iter_266) for every seed |
| AASFOD | AASFOD-OBB | yes | required; standard recipe: per-seed TSD + alignment/FNS two stages (`train_aasfod`), valid only with tsd.json + stages.json |
| SFYOLO | SFYOLO-OBB | yes | required; fill TAM/TEST then DIOR-17 |

## Out of queue (defensive)
Student-as-required-TEST, LoRA-CGA oracle, B_REG, F_text_only, F_veto_only, ROI expansion, 3-seed on new DIOR-17.

## DIOR domains
Official DIOR-C (19, severity 3). Already compared: brightness, contrast. Extra (keep, not a substitute): cloudy. **Remaining 17:** gaussian_noise, shot_noise, impulse_noise, speckle_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, gaussian_blur, snow, frost, fog, spatter, elastic_transform, pixelate, jpeg_compression, saturate. New cells seed=42.

## Metrics
- mAP = VOC-style AP50
- Δada = mAP_method − mAP_A (same corruption)
- mPC = mean over official completed corruptions; DIOR mPC uses the 19 DIOR-C names
- rPC = mPC / A_clean × 100%

## OURS deployment
Code = `IRAOD-New-strict-af` @ 0f98a48 (F's code) + `ours_strict_vlst_model.patch`, which adds `model.cfg.vlst_model`: strict VLST hardcoded `ViT-B-32`, so `VLST_BACKEND=clip` would have built CLIP ViT-B/32 (512-d), not RN50x64 (1024-d). Default `ViT-B-32` keeps E/F unchanged. Per host:
```
cp -a ~/IRAOD-New-strict-af ~/IRAOD-New-ours-strict && cd ~/IRAOD-New-ours-strict
patch -p1 < <repo>/experiments/comparison/ours_strict_vlst_model.patch
cp <repo>/configs/unbiased_teacher/sfod/unbiased_teacher_oriented_rcnn_selftraining_vlst_cga_dior_orthonet_ours_strict.py configs/unbiased_teacher/sfod/
train_d17_seed_1gpu.sh a,b OURS <domain> <seed>
```
The resolved config differs from F's dumped `dior_F_strict.py` only in `vlst_model`, `vlst_pretrained`, `vlst_cache_dir`, `vlst_vlm_dim` (512 → 1024) and the CGA/VLST env (`clip`, `CLIP_MODEL=RN50x64`). Two code defaults also change with backend=clip on DIOR (strict-af `sfod/cga.py`): `CGA_EXCLUDE_IDS` defaults to [7, 8, 11] (expressway-toll-station, dam, harbor) instead of [] for sarclip (`_build_cga`), and the prompts become "an aerial image of a {}" (CGA `CLIP_TEMPLATES` default and VLST `_normalize_templates`) instead of the SAR templates. Both match the original method (sogc-orthonet-phase1 `sfod/cga.py` has the same defaults).
