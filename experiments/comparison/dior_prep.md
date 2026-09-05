# DIOR-R OrthoNet strict prep (not launched)

## Staging (.67) 2026-09-06T06:16+08 live tar, not terminal
- Source: gpu-183 `/mnt/HDD_6TB/zechuan_files/IRAOD/dataset/DIOR` (21G)
- Dest: `/mnt/shared/zechuan/iraod_data/DIOR`
- Oriented XML: 23463 (complete)
- HBB XML: 23463 (complete)
- JPEGImages: 9889 and growing ( tar writing `09889.jpg` )
- ImageSets: missing
- Corruption/: missing
- Launch blocked until JPEG complete + ImageSets + Corruption present
- Do not use R50; source config `configs/baseline/oriented_rcnn_orthonet_dior.py`
- No GPU on .183

## Strict A–F
Overlays start at `unbiased_teacher_oriented_rcnn_selftraining_st_baseline_dior_orthonet_strict.py`.
C–F copy RSAR strict env (CLIP / SARCLIP / VLST) with DIOR 20 classes and image-only Corruption/${corrupt}.
Same freeze: seed42, final EMA, weight_l=0, use_bbox_reg=False, no target LoRA.

## Ports
IRG/LPLD/Simple-SFOD not in this repo as OBB implementations. Executable only after faithful port; until then N/A with reason, not approximate HBB numbers.
