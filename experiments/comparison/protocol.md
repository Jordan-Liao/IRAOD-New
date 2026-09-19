# IRAOD strict Source-Free comparison protocol (revised)

## Claim
Fair OBB Source-Free comparison of A–F plus IRG, LPLD, SFUT, **DRU**, AASFOD, and SFYOLO on RSAR corruptions and DIOR-C. Shared OrthoNet-50 + Oriented R-CNN source, final-EMA Teacher TEST, no target GT in adaptation/selection. **DRU is required** (DRUUnbiasedTeacher; not B / Mean Teacher).

## Frozen scientific identity
- RSAR source: `/mnt/shared/zechuan/iraod_artifacts/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`
- DIOR source: `/mnt/shared/zechuan/iraod_artifacts/dior_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`
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
| IRG | IRG-OBB | yes | required; rerun NaN cells once |
| LPLD | LPLD-OBB | yes | required; rerun NaN cells once |
| SFUT | SFUT-OBB | yes | required; not B |
| DRU | DRU-OBB | yes | required; Dynamic Retraining-Updating + Historical Student Loss |
| AASFOD | AASFOD-OBB | yes | required; fill missing TEST |
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
