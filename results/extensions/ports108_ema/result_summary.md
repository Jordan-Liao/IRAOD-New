# IRG/LPLD/SFUT final EMA results

**108/108 new native evaluation artifacts are complete**, plus12 reused fixed
source measurements. The consumer covers1,037,304 new EMA TEST image records
and18,229,448 post-NMS detections. This is not completion of pending methods,
Student evaluations or qualitative exports.

Values below are mPC in percent, mean +/- sample SD over seeds42/43/44;
delta is against the same fixed source A in percentage points. Raw120 rows,
including the12 source references, are in `raw_results.csv`; `per_seed.csv`
preserves each seed. A is one measurement, not three independent sources.
CSV `mAP50`, `mPC` and `delta_A` retain0-1 units; the display table converts
them to percent/percentage points without changing stored precision.

| Method | RSAR mPC (%) | RSAR delta A (pp) | DIOR mPC (%) | DIOR delta A (pp) |
|---|---:|---:|---:|---:|
| A, fixed source |31.7142|0|27.1185|0|
| IRG |30.6249 +/-5.9715|-1.0893|27.3923 +/-0.1109|+0.2738|
| LPLD |28.8620 +/-6.2491|-2.8521|27.5468 +/-0.1335|+0.4283|
| SFUT |32.2515 +/-1.3188|+0.5373|27.2890 +/-0.0059|+0.1705|

1. **RSAR is seed-sensitive.** IRG seed mPC is34.0998/34.0452/23.7296%;
   LPLD is32.6856/32.2499/21.6506%. Seed44 has five zero-detection native cells:
   IRG/LPLD on `point_target`, and IRG/LPLD/SFUT on `noise_suppression`.
   Each still contains all8538 TEST image entries. These zero scores are
   retained in the means and statistics; their cause is not diagnosed here.
2. **DIOR has small positive observed gains.** Relative mPC changes against A
   are+1.0095/+1.5793/+0.6288% for IRG/LPLD/SFUT, versus RSAR
   -3.4347/-8.9933/+1.6943%. This is descriptive, not evidence that every
   method or domain improved.
3. **Inference is low-power.** With three seeds, exact two-sided sign-flip
   Holm p values for mPC deltas are1.0 on RSAR and0.75 on DIOR. DIOR SFUT's
   t-test Holm p is0.00118795 under the normal-difference assumption and its
   very small observed variance; it is not distribution-free significance.
   This three-method phase is not the final all-extension comparison family.

The zero-detection rows are explicitly listed in `zero_detection_cells.csv`.
They provide no eligible instances for the current equal-sample joint-tSNE
rule in the seed44 RSAR `point_target/ema` and `noise_suppression/ema` groups.
Do not invent points, replace seeds, drop zero scores or tune the display
threshold. Existing training evidence may be investigated to explain the
collapse; no rerun, training change or extra experiment was performed here.

The source/code/checkpoint/native-ID records are retained in the report and
`artifact_manifest.json`. Large prediction coverage remains remote with its
size and SHA256 recorded. CPU collection used the existing method-aware
consumer; the accepted owner terminal event was recorded without repeating
the owner's job reconciliation. Core and Student180 deliveries remain intact.
