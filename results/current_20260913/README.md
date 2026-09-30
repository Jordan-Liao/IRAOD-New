# Current accepted TEST results: partial overall delivery

**Cutoff: 2026-09-13 16:59 UTC. Actual eligible native metrics are integrated for
900/900 keys, not inferred from completion statuses. ROI, archival and numerical
repair remain unfinished; this is a draft current-results report, not final
completion of the experiment plan.**

| Result unit | Accepted at cutoff | Still open / separate |
|---|---:|---|
| Canonical TEST roles | 900 eligible metrics / 912 keys | 7 held; 5 invalid native EMA |
| Native metric integration | 671 frozen eligible + 229 newly transcribed | 0 missing eligible metrics |
| Prescribed detector TRAIN budgets | 540/540 | 6 numerically invalid cells; 0 repaired |
| Auxiliary TSD / TAM / LoRA / source | 36/36; 12/12; 2/2; 2/2 | Not additional detector seeds |
| Eligible ROI keys | 377/720 | 343 pending; 12 held outside denominator |
| Cloud-delivered unique ROI keys | 286/377 currently complete | 91 complete but not yet delivered |
| Verified cloud archives | 247; 167044220165 bytes | PR21 + batches 1-7; not 247 experiment keys |
| Batch 8 assignment | 12 assigned / packaging | 0 counted as cloud delivered at cutoff |

ROI/archival counts are accepted operator accounting, not a new filesystem or
cloud recount. The ROI observation is 16:51 UTC (active 0); archival is 16:59 UTC.
See [current_report_summary.json](current_report_summary.json) for denominators,
source receipts, pending-family counts and open blockers. Later packets are not
silently incorporated.

## Actual performance and support

All numeric raw values and native paths are in [raw_metrics.csv](raw_metrics.csv)
and [per_class.csv](per_class.csv). The 5 invalid EMA native zero scores are
preserved **only** as `native_mAP50`/`native_AP50`; eligible `mAP50`/`AP50` stay
empty. Held roles have no substituted metric. Empty CSV fields mean unavailable,
never zero.

| EMA example (not a ranking) | RSAR mAP50% +/- sample std (complete seeds) | DIOR mAP50% +/- sample std (complete seeds) | Budget / interpretation |
|---|---:|---:|---|
| A/source, not EMA | 34.392 +/- NA (1) | 27.711 +/- NA (1) | One fixed source per dataset |
| AASFOD | 11.643 +/- 0.783 (3) | 23.207 +/- 0.177 (3) | One detector epoch plus TSD; disclosed port |
| B_REG | 35.253 +/- 0.260 (3) | 28.166 +/- 0.047 (3) | Common one detector epoch |
| F_text_only | 32.148 +/- 0.389 (3) | 27.574 +/- 0.044 (3) | Common one detector epoch |
| F_veto_only | 35.352 +/- 0.342 (3) | 28.171 +/- 0.031 (3) | Common one detector epoch |
| SFYOLO | 31.918 +/- 0.642 (2: 42,44) | 27.378 +/- 0.022 (3) | EXTENDED: two detector epochs + actual TAM 160k |
| LoRA-CGA | 34.865 +/- 0.526 (3) | 28.124 +/- 0.055 (3) | Supervised Oracle appendix only |
| LoRA-CGA+VLST | 36.004 +/- 0.252 (3) | 28.132 +/- 0.053 (3) | Supervised Oracle appendix only |

These are selected examples; **all 52 dataset/method/role summaries**, including
Student, are in [summary.csv](summary.csv). Roles are never substituted.
[coverage.csv](coverage.csv) gives eligible, held, invalid, frozen/new and
complete-seed counts for every summary.

1. Full AASFOD EMA coverage now supports a descriptive source delta of
   **-22.750 pp RSAR / -4.504 pp DIOR**. Its Student means are 7.931% / 19.140%;
   neither low valid scores nor role differences justify exclusion. The cause
   of the performance gap is not established by this report.
2. RSAR F_veto_only now has all three complete seeds: **35.352% (+0.960 pp)**
   versus F_text_only **32.148% (-2.244 pp)**. This is descriptive, not a
   significance or best-checkpoint claim.
3. RSAR SFYOLO's 46/48 eligible roles do not provide three complete seeds:
   seed43/noise_suppression is held for both roles. Its extra detector/TAM
   budget precludes equal-cost ranking against common one-epoch methods.

### Aggregation protocol

The fixed protocol and authorizations are unchanged; see the
[frozen protocol/budget disclosure](../completed_snapshot_20260912T014556Z/README.md#3-预算前置与未完成边界)
and manifest lineage in the raw rows and
[provenance.json](provenance.json). All methods are disclosed **Oriented R-CNN
OBB ports**, not reproductions of the papers' upstream detector architectures.
SFUT is the distinct paper-defined method, not B/ST. DRU has no authorized
matrix or runs: the frozen September 8 exclusion remains because the source
head lacks source-trained decoder-depth outputs. DRU is not compared.

Each seed first averages **all domains including clean**, equally weighted:
8 RSAR / 4 DIOR. Only full-domain complete seeds enter `summary.csv`; the
sample standard deviation uses ddof=1 and is NA for n=1. The fixed source seed42
is not duplicated into three independent repetitions. `per_seed.csv` keeps
incomplete seeds and lists their missing/invalid domains without a partial-domain
mean. `per_domain.csv` retains all supported domain-level means with exact seed
identities and denominators. RSAR IRG/LPLD/SFUT have complete seeds42/43 only;
SFYOLO has seeds42/44 only. Unequal seed support is not equivalent evidence.

Raw deltas use matching dataset/domain/source identity, A/source/42:
`delta_vs_source_pp = 100 * (method_mAP - source_mAP)` and
`relative_vs_source_percent = 100 * (method_mAP / source_mAP - 1)`.
Summary deltas average paired domain deltas within complete seeds first.
No cross-budget ranking, significance test, interpolation, hidden incomplete-seed
mean or TEST-based checkpoint selection is performed. Printed class AP precision
(usually three decimals) is preserved; full-precision mAP comes from native
`metric.mAP`, not the rounded `metric.AP50` or class table.

## Lineage and reproducibility

The canonical key is `(dataset, domain, seed, method, role)`. Frozen PR21
`raw_metrics.csv` contributes its 676 native records unchanged: 671 eligible plus
5 invalid. Only the 229 previously absent eligible keys are added from accepted
final native evaluations. Duplicate attempts/copies are not extra keys.
`metric_origin`, original checkpoint/evaluator paths, source identity, recorded
code revisions, evaluated/collected times and cutoff are explicit in each raw row.
`manifest` retains the original authorization lineage even when a host/path was
relocated; `evidence_basis` points at actual native execution metadata.

[native_metadata.json](native_metadata.json) preserves the 229 small native
metric JSONs, printed class AP, terminal sentinels, prediction-count receipts and
selected execution metadata used for this transcription. No model, predictions,
dataset or ROI NPZ was opened or uploaded by this report. Historical native
prediction admission is inherited from the accepted producer receipts, **not
revalidated with model inference or a fresh prediction scan**. Newly transcribed
post-NMS detection totals remain NA because those payloads were not reopened.

The original `test_completion_status.csv` and `test_completion_summary.json`
remain the **02:44 status-only checkpoint** at
`1217bf1d05fb91c754403e39a29f9bffe2736171`, not metric evidence or a replacement for
this later report. PR21 and its frozen tables are unchanged.

```sh
python3 results/current_20260913/build_report.py --check
```

This scoped check reuses the frozen validator's CSV/key helpers and standard
descriptive statistics; it checks the exact 912-key partition, all 229
native role/case/source/final-checkpoint/sentinel/count bindings, cutoff and
class coverage, frozen-value preservation, and regeneration of every table.
It does not invoke the manifest-v2 experiment aggregator, whose runner-completion
schema is different from these accepted native comparison reports.

## Unfinished parent objective

The remaining **343 ROI keys** are B-F seeds43/44, both roles (240), and
IRG/LPLD/SFUT EMA (34/34/35). Host67's approved GPUs4-7 are foreign occupied;
the tested Host221 EMA case still fails strict native prediction equality.
Host68 is deferred; Host134 remains paused. The 377 completed ROI groups are
not 720/720, and 286 delivered groups are not the 247 physical archives.

The single-image PR24 capture has 32 actual / 32 native detections, but 191/192
finite box/score scalars and 32/32 rows differ; maximum box absolute difference
0.05194091796875 and score difference 0.0002319812774658203. This is not order-only,
not valid ROI output and not numerical repair. No tolerance or native output was
changed; no index/output was counted. Historical cuDNN/TF32/CUBLAS/RNG flags and
the NaN root cause are **UNVERIFIED**, with no causal hardware/library claim.
The numerical grant remains 2/2 diagnostic starts used, 30/2400 GPU seconds,
0 reserved and no starts left; the separate one-image capture was 1/1, 22 seconds.
CPU startup repairs and PR23 admission are not model fixes. Further execution
requires the existing sole operator and authorization, not this report.
