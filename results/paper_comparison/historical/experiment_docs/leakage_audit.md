# Leakage audit (RSAR strict A–F, code 0f98a48)

- weight_l: 0.0 in all strict configs and B/E resolved dumps. Producer requires weight_l=0 when strict_source_free.
- labeled dataloader: `StrictSourceFreeDOTADataset` + image-only `img_prefix=.../corruptions/<c>/val/images`. No training annfiles passed in launch `--cfg-options`.
- pseudo-label branch: teacher on unlabeled images; extra_info `pseudo_num(acc)` uses GT only when not strict_source_free (E/F still log the key; iter1 NaN on empty rank is diagnostic).
- CGA/VLST/LoRA: C uses CLIP RN50x64; D/E/F use frozen base SARCLIP (`SARCLIP_PRETRAINED` safetensors). Launch unsets `SARCLIP_LORA`. Strict configs raise if LoRA set.
- prototype: VLST online from teacher pseudos (`vlst_prototype_momentum=0.9`); no target-label init in strict cfg.
- test labels: only `test.py --eval mAP` with `data.test.ann_file=RSAR/test/annfiles/` at offline eval. Train `--no-validate`.
- unlabeled bbox: `use_bbox_reg=False` by design; bbox unlabeled loss zeroed. Not leakage.
- F workdir is preflight path; scientific train args match formal except log interval/port/workdir.

No target-label leakage found in executed A–F cells.
