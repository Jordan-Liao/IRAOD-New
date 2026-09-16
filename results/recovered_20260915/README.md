# Recovered comparison results

**Metric cutoff: 2026-09-15 22:34:16 UTC. Native TEST coverage is 912/912;
full-test ROI computation and verified delivery are 732/732. Archive delivery
closed at 2026-09-16 03:07:07 UTC, independently of the earlier metric cutoff.**

| Result unit | Complete | Interpretation |
|---|---:|---|
| Canonical detector training budgets | 540/540 | Six previously invalid cells received separate full-budget corrected runs |
| Corrected final checkpoints | 12/12 | Student and EMA for those six cells; originals retained |
| Native TEST metrics | 912/912 | 900 unchanged historical valid rows plus 12 new native results |
| Full-test ROI groups | 732/732 | Previous eligible 720 plus the corrected 12 |
| New prediction/ROI image roles | 102456 each | 12 models, 8538 RSAR TEST images per model |
| Prerequisites | TSD 36, TAM 12, LoRA 2, source 2 | Complete; not extra detector seeds |

The source models are fixed seed42 references. All non-source summary groups
now have three complete adaptation seeds, 42/43/44. B_REG, F_text_only,
F_veto_only and the Oracle variants require EMA TEST only; their absent Student
or ROI rows are not missing planned experiments.

## Actual corrected-checkpoint measurements

Values below are native mAP50 percentages, rounded only for display. Full
precision is retained in [raw_metrics.csv](raw_metrics.csv) and
[recovered_results.json](recovered_results.json).

| RSAR domain | Seed | Method | Student mAP50% | EMA mAP50% |
|---|---:|---|---:|---:|
| noise_suppression | 43 | SFYOLO | 2.2609 | 11.7867 |
| noise_suppression | 44 | IRG | 14.4198 | 18.6924 |
| noise_suppression | 44 | LPLD | 12.4575 | 18.2282 |
| noise_suppression | 44 | SFUT | 11.9847 | 16.7275 |
| point_target | 44 | IRG | 21.3561 | 52.6953 |
| point_target | 44 | LPLD | 21.8070 | 52.1095 |

Low but finite native scores remain valid observations. Neither the old NaN
checkpoints' zero scores nor held-role placeholders were used as recovered
metrics or as performance baselines.

The updated RSAR EMA summaries below average all eight domains, including clean,
within each complete seed, then report mean and sample standard deviation across
the three seeds. They are descriptive, not a cross-budget ranking.

| Method | Mean mAP50% +/- sample std | Detector/prerequisite budget |
|---|---:|---|
| IRG | 36.622 +/- 0.105 | Common one detector epoch |
| LPLD | 34.859 +/- 0.459 | Common one detector epoch |
| SFUT | 35.747 +/- 0.082 | Common one detector epoch |
| SFYOLO | 32.210 +/- 0.680 | Extended two detector epochs plus TAM 160k |

All 52 dataset/method/role summaries, including Student and the separately
labeled supervised Oracle variants, are in [summary.csv](summary.csv).
[per_seed.csv](per_seed.csv), [per_domain.csv](per_domain.csv),
[per_class.csv](per_class.csv) and [coverage.csv](coverage.csv) retain the
underlying support. Per-class AP keeps the native printed precision; native
full-precision mAP is not reconstructed by averaging rounded class AP.

## Repair and execution provenance

The causal defect was an admitted zero-height teacher pseudo box, also appended
as a positive proposal. Native rotated-box encoding produced NaN dy/dh targets.
The shared pseudo-admission mask now requires finite geometry and strictly
positive width and height before assignment/sampling. Labels and semantic fields
use the same mask. No weight replacement, loss masking, batch skipping,
threshold change or TEST-based checkpoint selection was used.

The five original a39 cells use corrected training revision
`e6599be4fab9b2da07ea1abd87cc8473be2e0d60`; SFYOLO uses
`82e08b0b5f15278e24f949ae1670f91f86713f8d`. Prescribed finals remain
`iter_266.pth` / `iter_266_ema.pth` for the one-epoch ports and
`iter_531.pth` / `iter_531_ema.pth` for two-epoch SFYOLO.

These twelve evaluations use native evaluator
`331d2131b84651f0a2930a3d53faeefad8701531` and the already-validated single-forward
native/ROI exporter `92068ccb7ca19570e02880415671b6627ce769ca`.
Each result retains the actual evaluated checkpoint path and training revision.
An obsolete destination-staging path is never substituted for a Host67 local
checkpoint. Actual native commands are retained as provenance; rerunning them
requires the existing host setup, an approved free GPU and its owner lock.

The evaluation scope consumed 15 starts and 5593 allocated GPU-wall seconds.
This includes three pre-model portability failures totaling 18 seconds, not
15 successful model evaluations. Missing native `data.val` was resolved by
reusing the established evaluator-compatible config, not by changing training.
The first IRG pair's missing original-plan binder metadata was repaired on CPU
against authentic rows, without rerunning the GPU forward or changing metrics.
Original pre-binder packets and their post-binder completion evidence remain
distinct in the imported records.

Production did not request a second independent ordinary-native baseline.
`ordinary_native_exact_parity=false` in that mode means no independent baseline
was requested; it is not a failed comparison or a claim of historical
cross-host bitwise equivalence. Genuine native completion, full image identity
coverage and strict same-forward ROI/prediction alignment are required.

## Immutable history and reproducibility

The [September 13 report](../current_20260913/README.md), at its original
2026-09-13 16:59 UTC cutoff, remains unchanged: 900 valid metrics, seven held
roles and five invalid native EMA records. This new current view replaces only
those twelve unavailable/ineligible slots. Every field of the other 900 raw
rows is retained exactly, including its original provenance and cutoff.
Original invalid records remain available in the earlier snapshot.

The importer selects scientific metadata and actual class-AP text, not private
session stores, models, predictions or ROI payloads. The result generator reuses
the earlier report's aggregation functions and refuses a partial twelve-key
recovery, duplicate/mismatched identities, wrong final roles/revisions, incomplete
coverage or invalid native metrics.

```sh
python3 -m unittest discover -s results/recovered_20260915 -p 'test_completed_report.py'
python3 results/recovered_20260915/build_completed_report.py --check
```

To rebuild the six generated CSVs, run the second command without `--check`.
`import_recovered_results.py OPERATOR_PACKET` performs the one-time import from
the accepted operator packet and its referenced small evidence files. It is not
a new evaluation command.

## Delivery boundary

All three delivery sets are closed, with pending=0 and active=0. The
[delivery manifest](delivery_manifest.json) records exact ROI membership,
destinations, provider receipts and the inherited historical proof references.

| Delivery set | Scientific units | Provider files | Bytes |
|---|---:|---:|---:|
| Original canonical ROI scope | 720 ROI keys | 681 | 326938328325 |
| Corrected TEST + ROI | 12 ROI keys | 12 | 4928696320 |
| TRAIN_REPAIR | 6 training cells | 6 | 5091061760 |
| Total | 732 unique ROI keys, training assets separate | 699 | 336958086405 |

Provider file counts differ from ROI-key counts because historical archives
also contain companion assets and grouped outputs. The six TRAIN_REPAIR
archives are not six additional ROI keys.

The receipt gate is matching provider size and whole-file MD5, not merely a
successful process exit or a local TAR. The manifest preserves per-file
rollout/corrected receipts and reuses the accepted historical377 and training
proofs without rehashing or retransmitting payloads. The final checker tests the
exact original 732-key union, disjoint delivery sets and file/byte accounting.

All destinations are private paths under `/apps/bypy/IRAOD-New/results/`.
The corrected results use `post_repair_test_roi12_20260915/host{67,183,221}/`;
training-repair assets use `train_repair_20260915/`. Original destinations and
individual rollout file paths are retained in the manifest. Existing account
access is required; no credentials or public share links are published.

[validation_evidence.json](validation_evidence.json) records the final
cartesian-matrix audit, source-evidence boundaries and zero missing/extra keys.
