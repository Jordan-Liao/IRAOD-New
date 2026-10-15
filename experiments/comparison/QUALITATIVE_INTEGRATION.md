# Qualitative completion and RSAR per-class integration

Qualitative evidence is collected independently of the unfinished multi-seed
quantitative campaign. Completing this collector does not mark all eight
requirements complete and does not generate a final comparison DOCX.

This section describes the earlier qualitative collection boundary. The
quantitative campaign and final report were subsequently completed on 2026-09-08:
192 native cells / 1,844,096 prediction-image records / 2,048 per-class rows.
See `REPORT_CONSUMER.md` and the current combined reports. The accepted
qualitative summary's dated quantitative progress is historical, not current.

## Full qualitative collection (CPU)

```bash
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ART=/mnt/shared/zechuan/iraod_artifacts/comparison
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
"$PY" -m experiments.comparison.complete_qualitative \
  --plan "$ART/full_test_roi_v3_331d213/completion-plan.json" \
  --embedding-root "$ART/full_test_roi_v3_331d213/tsne" \
  --out-dir /absolute/new/qualitative-evidence \
  --expected-detections 18468211
```

The delivered streaming consumer validates every image NPZ once, including
row alignment, zero detections, complete TEST IDs, checkpoint/run identity and
frozen visualization selection. Embedding validation subsequently reads only
the NPZs referenced by sampled points, rather than scanning the entire dataset
a second time. All six panels, joint coordinates, point provenance and shared
normalization are still checked.

Completion requires all 132 RoI groups, 1,267,816 image-role records, 3,520
selected visualizations and 24 full-ROI-source joint embeddings. The detection
total is accumulated from validated rows and reconciled with the supplied
expected count; it is never copied from an operator's hand sum.

Remote outputs:

```text
roi_vis_coverage.csv            # large, complete per-image manifest; do not commit
roi_group_summary.csv          # 132 compact rows
embedding_index.csv            # 24 compact rows
qualitative_summary.json       # qualitative complete/partial; quantitative pending
report_input_fragment.json     # qualitative_plan + actual embedding paths for later report input
collector_sources/             # exact scripts used for reproducibility
```

The fragment can be merged into the final report manifest after real
quantitative cell/checkpoint evidence is collected. It does not supply or
invent any quantitative completion. Source snapshots and the base commit/
working-tree flag distinguish a tested staged collector from a clean committed
checkout.

The completed production collector at
`/mnt/shared/zechuan/iraod_artifacts/comparison/qualitative_complete_20260907_stream/evidence`
must be reused for the final consumer, not run again. Supply that directory via
`collect_report_manifest --qualitative-evidence`; see `REPORT_CONSUMER.md`.
Reuse checks plan/index bindings and reconciles counts without reading feature
NPZs. The accepted full streaming audit remains the evidence of row alignment.

Only the small summary, group/embedding indices and provenance are versioned
in `results/paper_comparison/`. NPZs, images, embedding binaries and the
1.26-million-row CSV remain remote.

## The 180 missing RSAR AP values

The standard table initially contains 78 rows: source on clean plus seven
corruptions, and chaff B-F. Exactly 180 further rows cover six non-chaff
corruptions x B-F x six classes. This does not add clean B-F rows or make
multi-seed quantitative statistics complete.

Collect the real class tables/metric JSONs without running evaluation:

```bash
"$PY" -m tools.complete_rsar_per_class collect \
  --paths-module "$ART/xaf_s424344/paths.py" \
  --owner-format "$ART/result_completion_v2_91548e4a/consumer_format.json" \
  --out /absolute/rsar_nonchaff_ap_evidence.json
```

Then, in the integration checkout, append the collected batch:

```bash
python -m tools.complete_rsar_per_class append \
  --evidence /absolute/downloaded/rsar_nonchaff_ap_evidence.json
```

The append above is the historical seed42 operation and must not be repeated
against the now-published combined schema. Its original inputs and the
258-row result are frozen in `results/paper_comparison/historical/seed42/`;
the final `per_class_summary.csv` contains all 2,048 native per-class rows.

Every source `metric.mAP` must match the frozen seed42 raw value. The output
table copies the exact mAP text from `raw_results.csv`, while class AP retains
the explicit three-decimal precision printed in `class_ap.txt`. The whole
30-group/180-row batch is validated before any write; existing table bytes
are preserved, duplicate re-application is a no-op, and conflicting existing
values are never overwritten. The evidence JSON records each actual source
class table and evaluation JSON.

## Control transport

If the public GPU endpoint refuses connections, the coordinator-provided
internal route is:

```bash
ssh -p 20183 zechuan@1.14.177.180 \
  'ssh -oBatchMode=yes -oStrictHostKeyChecking=yes zechuan@192.168.0.167 bash -s'
```

This is a transport change only. Do not modify live queues, training goals,
GPU assignments or runtime scripts to collect these CPU evidence files.
