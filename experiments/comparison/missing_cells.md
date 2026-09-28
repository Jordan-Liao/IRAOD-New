# Reusable vs missing cells (snapshot 2026-09-28, complete)

Source of truth: `results/paper_comparison/raw_results.csv` (one row per dataset × domain × method × seed),
built by `build_summary.py` from `scan_eval_cells.py` output of gpu-67/68/94/134/138/183.
Summary: `results/paper_comparison/result_summary_cn.md`.

## Status
- RSAR 272/272 and DIOR 714/714 cells `done` (A–F, IRG, LPLD, SFUT, DRU, AASFOD, SFYOLO × seeds 42/43/44; A seed-free).
- Every non-A cell's final EMA iteration equals its method's seed-42 iteration.

## How the 2026-09-27/28 reruns were produced
- DIOR seeds 43/44 (all methods except AASFOD/SFYOLO): `train_d17_seed_1gpu.sh` (1 epoch, global batch 32,
  lr 0.02; E/F DDP2×16), queued by `d17_queue.sh`; outputs `~/iraod_jobs/ext_runs_e1/` on gpu-67/68/183/138.
  Two SFUT cells failed under DDP (frozen teacher has no trainable params) and were rerun 1×32.
- RSAR DRU seeds 43/44: `rsar_dru_seed.sh` (seed-42 command, only `--seed` changed; iter_266).
- Collapsed cells (IRG/LPLD s44 point_target+noise_suppression, SFUT s44 noise_suppression, SFYOLO s43
  noise_suppression): root cause = a zero-height pseudo box admitted by score only → `log(0/0)` in bbox delta
  encoding. Fixed in `UnbiasedTeacher.create_pseudo_results` (branch `fix/port-nan-seed44`, commit 85046c8;
  deployed as `repo_code_a39c832_fix1` / `accepted_d668a77_fix1`). Healthy-cell probe (IRG s43 point_target)
  0.5261 → 0.5281, within run-to-run GPU nondeterminism (old code vs old code already differs at iter 10).
- AASFOD: standard recipe only (per-seed TSD + alignment/FNS, `tsd.json` + `work/stages.json`):
  RSAR 24 + DIOR old-4 12 cells from the 09-11/12 runs (gpu-183), DIOR-17 × 3 seeds = 51 new cells
  (gpu-67 `aasfod_runs_36c2053_std`, 8 on gpu-138 `/mnt/SSD_4TB/zechuan/...`).
- SFYOLO DIOR s44 elastic_transform: rerun at iter_369 (old cell had fallen back to iter_185 after a truncated
  checkpoint write); needed the VGG encoder weight copied to gpu-183.

## Not in queue
Student-required TEST, LoRA oracle leftovers, B_REG / F_text_only / F_veto_only, ROI expansion.
