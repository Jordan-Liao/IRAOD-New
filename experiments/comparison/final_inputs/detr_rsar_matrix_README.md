# DETR-group RSAR seed-42 matrix: README (mixed hardware, commits, std definition)

Files next to this README:
- `detr_rsar_matrix_final.csv`: 72 data rows, one per (method, domain). Methods A, B, E, PETS, SFUT, SFYOLO, C, D, F; domains clean, am_noise_horizontal, am_noise_vertical, chaff, gaussian_white_noise, noise_suppression, point_target, smart_suppression. Seed 42 only.
- `detr_rsar_matrix_summary.csv`: per-method means over the cells, for AP50 and mAP50-95, over the 8 domains and over the 7 corruptions (clean excluded). `std` is the sample standard deviation (n-1) across domains, not across seeds: this matrix has one seed (42), so there is no seed variance.

## Mixed hardware (per-cell host and gpu_type columns)

- gpu-138: RTX A6000 48GB. A, B, E, PETS, SFUT, SFYOLO on clean and the six non-chaff corruptions; C and D on all 8 domains; F on 7 domains (all except gaussian_white_noise).
- gpu-67: A100-SXM 64GB. A, B, E, PETS, SFUT, SFYOLO on chaff (6 cells); F on gaussian_white_noise (1 cell).
- Affected domains:
  - chaff: the six baselines (A, B, E, PETS, SFUT, SFYOLO) ran on A100, while C, D, F ran on A6000.
  - gaussian_white_noise: F ran on A100, while C and D ran on A6000.

Cross-hardware checks (not resolved):

- B clean training on <other-host> (RTX 4090D 48GB): AP50 0.561453, against 0.568461 on gpu-138, Δ = -0.0070. The two replicate runs on <other-host> were stopped by the user before they finished, so the gap is unexplained and remains a caveat.
- A clean, eval-only (public ARS-DETR RSAR checkpoint): gpu-67 portcheck 0.6115977 (separate deploy dir, not a matrix cell; `/mnt/shared/zechuan/iraod_artifacts/ars_detr/portcheck_rsar_test/ap50_95.json`) vs gpu-138 matrix 0.6114685 (CSV 0.611468493938446): Δ1.29e-4; <other-host> 0.6114667: Δ1.8e-6; all evals batch 1 (test.py@917774b reads data.test.samples_per_gpu / test_dataloader, which this config does not set, so batch 8 was never applied, including portcheck).

## Commit mapping (code_commit column)

The code_commit column holds the commit recorded in each run's train log. After the rebase, these commits map to the following (audit doc line 249 of the detr-sfod branch):

- 6ebd39c = cf1e474
- a79aa92 = 728364b
- 67e3c80 = 09174b1

67e3c80 is the commit of 38 rows (B clean and the other baseline rows on gpu-138, plus E, SFUT, SFYOLO chaff on gpu-67). The original hashes are no longer on any branch.

Behaviour of the affected commits (audit doc line 249):

- a79aa92 -> 6ebd39c changes only the config setdefault and the tests; it does not change the B or PETS computation.
- 6ebd39c -> 67e3c80 adds the SFUT and SFYOLO ports and a behaviour-preserving refactor (before_student_step and weight_losses). PETS has no CGA, so the CGA fix does not affect it.
- So the two chaff cells (B with code a79aa92, PETS with code 6ebd39c) are counted, unlike the superseded_buggy_cga C/D/F gates on gpu-67, which ran on a CGA commit with the known bug and are excluded.

## Excluded directories (not cells)

- `*.invalid_crop_bug`: 24 original C/D/F runs that crashed on the CGA crop bug (cga.py:513).
- `*.stopped_by_user_*`: interrupted runs (F clean and F am_noise_horizontal on gpu-67; several D and F cells on gpu-138). The counted reruns for these cells are on gpu-138.
- `*.KILLED_BY_REQUEUE*` and `*.failed_path_bug*`: killed or mis-pathed earlier attempts.
- gpu-67 `rsar/[CDF]_s42/chaff`: the old C/D/F chaff gates on the buggy CGA commit (superseded_buggy_cga).

## Notes on specific cells

- F clean and F am_noise_horizontal: the interrupted originals are on gpu-67 at `sfod_cgafix2/rsar/F_s42/<domain>.stopped_by_user_20261103T135117` (not counted); the counted reruns are on gpu-138.
- A clean: matrix AP50 0.611468 rounds to 0.6115. The 61.16 in the audit doc (line 116) is the gpu-67 portcheck value. Use 0.6115 in tables.
- mAP50-95 for each cell is computed by `tools/eval_map50_95.py` on the host that produced the predictions.pkl, because RSAR file order is host-dependent.

## C/D/F caveat (from the train logs; not a confirmed bug)

- C/D/F score far below A on RSAR (7-corruption AP50: C 0.228, D 0.145, F 0.146 vs A 0.368). The non-CGA self-training baselines (B 0.353, E 0.356, PETS 0.398, SFUT 0.421) stay near A, so the gap comes from the CGA path.
- C keeps all pseudo-labels (dropped=0) but blends about 60% of targets; D and F keep only 23–28% (about 75% vetoed); CGA label probabilities are low on DETR outputs (mean 0.04–0.36), and the final unlabeled classification loss (main head, last logged iteration of epoch 1) is ≈1e-4 in most D/F cells (range 0.0000–0.0003 for the completed D and F cells except F point_target = 0.0373; its last decoder-layer auxiliary losses reach 0.0472), against 0.10–0.20 for C.
- Read: consistent with a calibration mismatch between the CGA thresholds and DETR outputs. The y-by-width decode path was not checked. Reported as-is with this footnote (no tuning).
