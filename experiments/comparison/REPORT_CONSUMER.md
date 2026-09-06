# CPU multi-seed result/report consumer

This is a separate consumer delivery, not a completed experiment report. It never
launches models, changes a queue, or writes into an existing result directory.
Use an already collected/mounted artifact tree or run this CPU command on the
artifact host. The existing report/binaries and main-checkout user files remain
untouched.

## Three different coverage scopes

| Evidence | Exact scope | Not evidence of |
| --- | --- | --- |
| `predictions.pkl` + ordered IDs | Full TEST post-NMS predictions, 8,538 RSAR / 11,738 DIOR images per cell | fc_cls feature export |
| aligned-v3 full-test NPZ + index | Every TEST image and post-NMS detection; 132 groups / 1,267,816 image-role records | Completion of unrelated training/report requirements |
| legacy aligned-v2 NPZ + index | Fixed 32 RSAR / 16 DIOR images only | Full TEST RoI features |
| joint t-SNE | 24 dataset/domain/role comparisons with sampled predicted instances | Full image coverage or all predicted instances |

The RoI exporter does **not** filter saved detections at the visual threshold
0.3. It saves all detections returned by the actual NMS, with the detector's
original score threshold, suppression and `max_per_img` still in force. A
regression explicitly retains a post-NMS score of 0.25. Rendering and t-SNE
eligibility separately apply `score > 0.3`; t-SNE then subsamples equally per
method. Neither changes the stored NPZ arrays.

The current v3 full-test plan and streaming extraction commands are in
[`RESULT_COMPLETION.md`](RESULT_COMPLETION.md). Full RoI `image_ids` are separate
from frozen `visualization_image_ids` (32/16). Capture supports one image / one
test augmentation. An existing export
directory is refused rather than resumed; an interrupted run needs a new output
root. A run-level index is written only after extraction finishes, so partial
NPZ files do not count as completed group evidence. A valid plan's individual
ready run can be extracted independently: the extractor does not wait for all
training cells or other export groups.

## Actual upstream format, inspected read-only

The known owner scripts are:

```text
/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_s424344/paths.py
/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_s424344/run_eval_full.sh
```

`paths.py` resolves:

```text
comparison/<rsar|dior>/<domain>/seed_<42|43|44>/0f98a48/methods/<method>/
```

RSAR chaff seed42 instead uses tag `f2145b4`. B-D work is `work/`;
E/F work is `ddp2/work/`, except RSAR chaff seed42 F uses
`preflight_ddp2/work/`. A uses the dataset's fixed source under its source
run's `train/epoch_100.pth`. The final iterations in this actual resolver are
RSAR **266** and DIOR **185**; no mtime/latest/best selection.

Full eval directories mirror the topology:
`eval_full_<domain>/`, `ddp2/eval_full_<domain>/`, or the chaff-F special
`preflight_ddp2/eval_full_chaff/`.

Each eval emits:

```text
eval_<timestamp>.json  # {"config": "...", "metric": {"mAP": 0.2446555644273758, "AP50": 0.245}}
eval_status           # eval_exit=0 <time> name=B domain=brightness seed=42 ema=<checkpoint>
class_ap.txt          # | class | gts | dets | recall | ap | ... | mAP | ...
predictions.pkl       # list[image][class] -> ndarray(N, 6): cx,cy,w,h,angle,score
pred_count.txt        # 11738 expect=11738   (RSAR: 8538)
```

Use **`metric.mAP`**, not rounded `metric.AP50` or the rounded class-table mean.
Per-class AP retains its printed precision (usually 3 decimals).
The eval script appends `eval_exit=0` **before** class-table and prediction-count
checks finish. Therefore status text alone cannot prove a complete cell. The
consumer reads the last eval-status record and requires the actual artifacts.
Both observed status forms are supported: the original `ema=<checkpoint>`
record, and the native wrapper's `sidecar=predictions.pkl.image_ids.json` record.
The latter binds checkpoint/config through the native sidecar itself.
Select an explicit eval JSON filename; do not choose an arbitrary successful
retry by modification time. If `class_ap.txt` includes the full-precision metric
dictionary, it must match the selected eval JSON (the owner script searches
appended logs for a table, so a stale table must not silently accompany a new eval).

**Same-inference provenance:** the updated `test.py --out` saves
`predictions.pkl.image_ids.json` from the actual inference batch metadata,
including distributed collection order. See
[`PREDICTION_IMAGE_ORDER.md`](PREDICTION_IMAGE_ORDER.md) for the independent
evaluation-checkout wiring. Old driver outputs without this native sidecar are
not retroactively verified. A matching count, separately rebuilt dataset,
sorted filenames or bare `test.txt` cannot certify an old pickle's order.

## Input manifest

Paths below must be accessible on the CPU consumer host; checkpoint paths are
provenance references and need not copy large weights. Pickles are loaded only
as trusted, owner-produced artifacts, never from untrusted third parties.

```json
{
  "schema": "iraod-comparison-report-v1",
  "roles": ["ema"],
  "source_ids": {
    "RSAR": "e489f015cc2b67926a415436c97532ff814f826c4861b659cbc712fb9413b98c",
    "DIOR": "74d7eafdc547976a86f8f35b2652dba384948e33ab5c15bb908fa3e54025782a"
  },
  "cells": "/absolute/cells.json",
  "checkpoints": "/absolute/checkpoints.json",
  "qualitative_plan": "/absolute/full-test-roi-plan-v3.json",
  "embeddings": [
    {
      "dataset": "DIOR",
      "domain": "clean",
      "comparison": "ema",
      "directory": "/absolute/tsne/DIOR/clean/ema"
    }
  ]
}
```

`cells` and `checkpoints` accept JSON arrays, CSV filenames, or inline arrays.
These are the consumer's explicit manifest records referring to the actual
files above; the existing eval scripts do not themselves emit these manifests.
The owner/coordinator supplies the records from collected evidence, not
invented successful completion fields. Missing cells may simply be omitted:
the consumer emits them as `not_started`, without calculating complete means.
`qualitative_plan` and `embeddings` may be absent while evidence is pending.

Cell record:

```json
{
  "dataset": "DIOR",
  "domain": "clean",
  "method": "B",
  "seed": 42,
  "role": "ema",
  "eval_dir": "/absolute/comparison/dior/clean/seed_42/0f98a48/methods/B/eval_full_clean_ids_v1",
  "eval_json": "/absolute/comparison/dior/clean/seed_42/0f98a48/methods/B/eval_full_clean_ids_v1/eval_TIMESTAMP.json",
  "prediction_image_ids": "/absolute/comparison/dior/clean/seed_42/0f98a48/methods/B/eval_full_clean_ids_v1/predictions.pkl.image_ids.json"
}
```

Image-order JSON must use the native `iraod-prediction-image-order-v1` schema
with `origin=inference_batch_img_metas`, ordered records and separate training/
evaluation SHAs. Bare ID arrays/text files are not accepted as verified
evidence. `eval_dir`, `eval_json` and `prediction_image_ids` must refer to the
same new evaluation. Optional cell `config` must match `eval_json.config`.

Checkpoint record, tied to the same dataset/domain/method/seed/role:

```json
{
  "dataset": "DIOR",
  "domain": "clean",
  "method": "B",
  "seed": 42,
  "role": "ema",
  "path": "/absolute/comparison/dior/clean/seed_42/0f98a48/methods/B/work/iter_185_ema.pth",
  "selection": "final",
  "iteration": 185,
  "verified": true,
  "source_id": "74d7eafdc547976a86f8f35b2652dba384948e33ab5c15bb908fa3e54025782a",
  "gpu_provenance": {"topology": "1x32", "physical_gpus": [4]}
}
```

`verified: true` is an owner assertion backed by its actual checkpoint/training
evidence, not a flag to set merely because a path is planned. The consumer
checks the declared source identity, final iteration/filename, and checkpoint
path against the actual eval-status identity; it does not independently
deserialize training weights or certify their full history. Preserve real GPU
metadata if supplied; E/F DDP2 need not share any GPU UUID with B-D. CSV booleans
use lowercase `true`.

For A, use **one** `seed: 42`, `role: "source"`, `selection: "source"` record
per dataset/domain, with the fixed source path. Do not make A seed43/44 copies.
Clean B-F must refer to independent clean-adapted final checkpoints. The
quantitative default is final EMA only, covering 192 cells:
12 source measurements + 12 domains x 5 methods x 3 adaptation seeds.
Student quantitative rows are accepted only if explicitly declared in
`roles`; they are not required to create the standard EMA report and are never
mixed with EMA statistics. This does not request any additional GPU runs.

## Statistical contract

- Unit: seed 42/43/44. Within each seed, aggregate all seven RSAR / three DIOR
  corruptions into mPC and mean paired delta to the fixed A on the same domains.
- Only after complete within-seed aggregation compute across-seed mean,
  **sample** standard deviation (`ddof=1`) and CI. Missing domains cannot be
  replaced by the mean of the available domains; missing seeds do not produce a
  complete mean/std/CI row. Clean has its own same-seed measured metrics.
- The numerical helpers `paired_t_test`, `sign_flip_test`,
  `bootstrap_mean_ci`, `holm_adjust` are reused from
  `tools/cga_research/statistics.py`. Its `align_pairs`, GPU-UUID admission and
  same-GPU sentinel policy are **not** used. No GPU identity is fabricated.
- The source is one fixed reference, not three independent measurements. All
  intervals are conditional on that source and reflect adaptation randomness
  only. Source mean is displayed as one value; source SD/CI are N/A.
- At **n=3**, power is low. Report 95% percentile bootstrap CI (100,000
  seed-block resamples, RNG seed42), noting its fragility. Two-sided t assumes
  normal differences and has df=2; zero variance makes t undefined. Exact
  sign-flip uses 8 assignments and cannot produce two-sided p below **0.25**.
- Holm families are the five B-F comparisons for each dataset/role/metric
  (`delta_A` or `clean_delta_A`), separately for each test. Incomplete families
  get no adjusted p-values. Undefined tests retain their planned slot as p=1
  internally; their own reported p-values stay N/A.
- Historical rPC remains `100 * mPC / fixed_source_clean_TEST`.
  `method_clean_normalized_percent` is separate:
  `100 * mPC / same_method_same_seed_clean`. Zero/missing denominators are N/A.

The consumer always loads and locks this checkout's original
`raw_results.csv`, `dior_raw_results.csv`, `per_class_summary.csv` and
`dior_per_class.csv`. It copies them unchanged into the new `historical/`
directory, including the old DIOR clean-val row (not used in TEST statistics).
Optional `historical_raw` / `historical_per_class` lists add evidence, not
disable these defaults. Original seed42 mAP/AP values are never overwritten.
A disagreeing re-evaluation is recorded separately (`eval_mAP50` / `eval_AP50`),
marked `metric_conflict` and excluded from complete statistical blocks.

## CPU invocation and outputs

Use existing scientific-Python and python-docx environments, for example the
provided local validation environments:

```bash
cd /home/ubuntu/repos/IRAOD-New.worktrees/strict-af-integration
export CPU_PY=/home/ubuntu/.copilot/session-state/036653d8-0a48-4e53-a199-800723b93937/files/roi-cpu-venv/bin/python
export DOCX_PY=/home/ubuntu/.copilot/session-state/7694cab0-8869-4a0c-abbf-4418844af06a/files/docx-venv/bin/python

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$CPU_PY" -m experiments.comparison.final_report \
  --manifest /absolute/collected-report-manifest.json \
  --out-dir /absolute/new/comparison-report-attempt \
  --docx-python "$DOCX_PY"
```

Use corresponding existing environments on the artifact host; do not pass
unmounted remote paths to a local process. Output directories are never reused.
Outputs have English filenames, with Chinese text in `comparison_report_cn.docx`:

```text
raw_results.csv                 # every expected quantitative cell, including pending/conflict
per_class.csv                   # actual parsed per-class evidence, precision and status
per_seed.csv                    # complete-corruption seed blocks; null when missing domains
per_domain.csv                  # same-domain across-seed descriptive results
summary.csv                     # mPC, delta, both separately named normalizations, clean metrics
paired_statistics.csv           # transparent seed-level differences and tests
prediction_image_coverage.csv   # full TEST image-index/ID/detection counts, streamed cell by cell
roi_vis_coverage.csv            # streamed 1,267,816 full ROI rows; 3520 selected vis rows
embedding_index.csv            # 24 expected comparisons, completion requires actual files/points
RSAR_ema_mpc.{pdf,png}           # only available complete blocks; no title/cherry-picking
DIOR_ema_mpc.{pdf,png}
historical/                     # byte-preserved original seed42 CSVs
report.json
comparison_report_cn.docx
build_status.json
```

No real plan means unknown RSAR selection identity: the RoI CSV is header-only,
not invented TEST image IDs. With the plan, pending rows remain pending until
validated NPZ/PNG evidence exists. Embedding admission checks all six panels,
the common coordinate array, point-to-feature provenance and source-normalized
vectors. Merely having `protocol.json` is insufficient.

`build_status.report_build=complete` only means the consumer finished writing.
`result_status=partial` remains partial. Even if every declared scope completes,
the explicit status is `declared_scopes_complete`, never
“all eight tasks complete”.

CPU regression:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  IRAOD_DOCX_PYTHON="$DOCX_PY" \
  "$CPU_PY" -m unittest tools.tests.test_comparison_report -v
```

Fixtures are temporary and synthetic; no synthetic metric, image, DOCX or
completion record is committed as a real result.

## Reusable collection and bounded production inspection

`collect_report_manifest` reads the owner's inspected path-only helper and
`consumer_format.json`. It builds `cells.json`, `checkpoints.json`,
`collection_inventory.{json,csv}` and `report-manifest.json` from actual metadata.
It never loads a pickle/NPZ, runs a model or invokes a queue. Checkpoint admission
records the actual owner `train_verified` result plus matching source identity,
file size, root code record and terminal evidence. No GPU UUID is fabricated.
The resolver/owner-format/plan are snapshotted into the new input directory;
live files are not modified, including no resolver pycache.

Lookup order is the native `eval_full_<domain>_ids_v1`, standard
`eval_full_<domain>`, then legacy `eval_<domain>`. The owner format supplies the
two exceptional A/clean JSON locations. Only a unique eval JSON is selected;
ambiguous retries are listed as missing evidence, never picked by mtime.
Legacy subset paths in the owner format are reference-only and never passed
as full-test RoI or embedding evidence.

For the explicitly bounded three-group snapshot, use a fresh output root:

```bash
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
ART=/mnt/shared/zechuan/iraod_artifacts/comparison
SNAP=/absolute/new/partial-collector-snapshot
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
# Run from the independently deployed report checkout (or set PYTHONPATH to it).
"$PY" -m experiments.comparison.collect_report_manifest \
  --paths-module "$ART/xaf_s424344/paths.py" \
  --owner-format "$ART/result_completion_v2_91548e4a/consumer_format.json" \
  --roi-plan "$ART/full_test_roi_v3_331d213/completion-plan.json" \
  --quant-seed 42 \
  --roi-run-id RSAR/clean/A/source \
  --roi-run-id RSAR/chaff/A/source \
  --roi-run-id RSAR/chaff/B/ema \
  --out-dir "$SNAP/input"
"$PY" -m experiments.comparison.final_report \
  --manifest "$SNAP/input/report-manifest.json" \
  --out-dir "$SNAP/report" --metadata-only
```

`inspect_roi_run_ids` freezes the exact groups permitted for NPZ inspection.
All other groups (including an in-progress B/student) remain `not_inspected`;
their index/NPZ files are not touched. The 132-group/full-image denominator
remains unchanged. `roi_group_coverage.csv` reports each group's checked image
count and detection rows; `roi_vis_coverage.csv` contains only inspected image
rows, not a million invented pending rows. Seeds excluded by `--quant-seed`
remain `not_inspected` in the inventory/report.

`--metadata-only` runs the complete evidence/numeric consumer without importing
plotting or requiring a DOCX renderer. It writes `report_build=metadata_complete`
and `rendering=deferred_metadata_only`, not a final document. Existing training
environments need no package changes for this partial inspection. Missing
native sidecars stay missing even where legacy quantitative mAP is known.

For final collection, repeat into a fresh snapshot without the seed/group
filters to inspect the declared complete matrix; add only actual v3 embedding
directories to `embeddings`. Then run the report consumer with
`--docx-python /existing/docx/python` instead of `--metadata-only`. Reuse all
numeric artifacts; do not rerun training to fill a report.
