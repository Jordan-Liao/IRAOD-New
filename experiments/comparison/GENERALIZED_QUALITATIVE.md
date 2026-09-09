# Explicit ports: full TEST ROI, fixed views and grouped joint t-SNE

Actual prepared root:
`ROOT/qualitative_manifests_b87f34e`, with exporter
`ROOT/integration_code_b87f34e`, where ROOT is the existing
`/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908`.
The360 Student/EMA native dependencies match the explicit release bindings.
All360 ROI input records were pending when prepared; no extraction ran.
These b87 plans retain their original GPU4,5,6 binding. Future newly prepared
plans allow the ledger's GPUs4,5,6,7; old plans and exports remain valid unchanged.
On the selected221 host only, use the operational overlay below: its physical
GPU0-4 mapping does not rewrite either preserved plan's original GPU list.
Preserved48 B-F embeddings below means the existing approved job/plan scope,
not an assertion that48 completed files have been observed.

This is preparation/collection code, **not execution evidence**. No GPU work,
training change, source re-export, numerical detector change, scheduler, or
queue regeneration is introduced. DRU, the three ablations and oracle methods
are not qualitative methods. Their quantitative scope is separate.

## Frozen arithmetic and reuse

| Scope | New ROI groups | Full-TEST image-role records | Fixed views | Embeddings |
|---|---:|---:|---:|---:|
| Accepted core seed42 A-F | 132 existing | 1,267,816 existing | 3,520 existing | 24 preserved |
| Already-approved B-F seeds43/44, EMA + Student | 240 | 2,305,120 | 6,400 | 48 preserved |
| IRG, LPLD, SFUT, AASFOD, SFYOLO, seeds42/43/44, EMA + Student | 360 | 3,457,680 | 9,600 | **72 new joint groups**, including reused A-F features |

The five ports are 180 trained models, not 360 models. Each gets two native
evaluation/ROI roles. Each new embedding groups one dataset/domain/seed/role:
12 domains × 3 seeds × 2 roles = 72. All five ports give **11 panels**:
A/source42 + B-F + five ports. These are new embeddings in new directories,
not replacements for the accepted 24 core or 48 B-F embeddings.

Source A is the same 12 existing exports in all three plans, not 36 new
exports. Unique source/core/B-F/ports ROI groups across the plans total 732;
the per-seed plans each contain 252 references (756 references, with A reused).
Do not sum the three collector summaries and call their repeated A references
new images. Total unique ROI image-role records are 7,030,616.

Each dataset's exact ordered TEST IDs come from the base plan: RSAR 8,538,
DIOR 11,738. Every role/domain retains the same frozen RSAR32/DIOR16 view IDs.
All post-NMS boxes/features, including scores below .3, are exported. Only
visualization and t-SNE eligibility use .3. There is no GT/class-stratified
sampling. Sampling seed42, source-only shared normalization, equal per-method
sample sizes, and the existing t-SNE defaults remain unchanged.

Common method final labels remain RSAR266/DIOR185. SFYOLO uses
RSAR531/DIOR369, **two detector epochs plus TAM**, with
`extended_two_epoch_TAM` / `rank_with_common_one_epoch=false` in runs,
group indices and embedding budget labels. These plots are qualitative
geometry, not a statistical ranking across budgets.

## CPU preparation from actual existing paths

Run from the delivered integration checkout, using the existing environment.
The parent fills these variables from the actual deployment; no paths below
authorize regenerating the authoritative old B-F metadata or Student180 queue.
`OLD_BF` is the existing `repo_manifests_37c65b6` metadata directory.
`CODE` is the parent-delivered successor to `integration_code_699c6bf`, **not**
a reason to touch the frozen `0f98...`/`32e325...` source/core checkouts.

```bash
CUDA_VISIBLE_DEVICES="" "$PY" -m experiments.comparison.extension_manifest prepare-qualitative \
  --base-plan "$BASE_CORE_PLAN" \
  --bf-plan "$OLD_BF/qualitative_seed43.json" \
  --bf-plan "$OLD_BF/qualitative_seed44.json" \
  --core-paths "$CORE_PATHS" \
  --runtime "$IRG_LPLD_SFUT_QUEUE/runtime.json" \
  --runtime "$AASFOD_QUEUE/runtime.json" \
  --runtime "$SFYOLO_QUEUE/runtime.json" \
  --methods IRG LPLD SFUT AASFOD SFYOLO \
  --eval-code "$CODE" --python "$PY" --cpu-python "$NATIVE_CPU_PY" \
  --out-dir "$NEW_QUAL_ROOT"
```

`--runtime` may repeat: each selected method must have exactly the 36 approved
domain/seed cells, without overlapping selected cells between runtimes.
Explicit approved subsets are supported; only declared methods get jobs.
Existing core/B-F checkpoint paths are checked against `CORE_PATHS`. Every
source run object must equal the core object, including its export path.
Every new method's source checkpoint, full TEST split/domain and image prefix
must match the corresponding frozen reference.

Outputs:

* `qualitative_seed{42,43,44}.json`: old A-F run objects unchanged, new IDs
  `DATASET/DOMAIN/seed_SEED/METHOD/ROLE`; method sets declared explicitly.
* `roi_jobs.json`: finite exact extraction and visualization argument vectors,
  native checkpoint/evaluation dependencies, code checkout/revision and current
  native-input evidence. `pending_inputs` is not a failed experiment or a
  completed result. `native_inputs_ready` is not ROI completion.
* `embedding_jobs.json`: 72 finite CPU command vectors and the exact 11
  checkpoint/export-index dependencies per group. No GPU commands are launched.
* `collection_commands.json`: one streaming collector command per seed.
* `qualitative_scope.json`: preparation arithmetic, input provenance and reuse,
  never an execution/completion claim.

GPU extraction retains the existing native interpreter. `--cpu-python` binds
the separate existing CPU environment with scikit-learn/plotting/native
visualization dependencies, without injecting packages into the GPU environment.
Generated visualization, embedding and collection commands explicitly hide
CUDA and restrict numerical libraries to one CPU thread.

No `train.list`, `eval.list`, Student queue or scheduler is written here.
The original `extension_manifest prepare` and `seeded_plan` APIs remain the
legacy B-F APIs; **do not rerun them to prepare this extension**.

## Actual extraction and artifact binding

On the original host, only the compute owner executes `export_argv` in its recorded `cwd` with its
actual GPU lock (`IRAOD_GPU_LOCKED=1`) and `CUDA_VISIBLE_DEVICES` set to exactly
one physical GPU allowed by that plan. Future plans allow GPU4,5,6,7, while
preserved b87 plans still allow4,5,6. GPUs0-3 and multi-GPU export visibility
remain rejected; the existing execution owner avoids actual foreign occupancy.
A port run binds `eval_config`, final EMA or
Student checkpoint and `native_prediction.eval_dir`. EMA uses the runtime's
`eval_dir`; Student uses the existing sibling convention
`eval_full_DOMAIN_student_ids_v1`. The native evaluator must produce that
explicit role binding; this module does not execute evaluation or create its
success files.

Before importing the GPU stack, extraction checks real nonempty checkpoint,
successful final `eval_status`, native `execution.json`, `predictions.pkl` and
native same-inference ID sidecar.
The sidecar must match checkpoint, evaluation config, TEST split/domain,
training/evaluation revisions, exact full TEST ID set and ordered native
records. The actual training revision comes from native execution, while the
earlier prepared revision is separately recorded; a stale queue revision must
not reject a correctly recorded later producer. A later failed status record
does not inherit an earlier successful attempt. Native order may differ from
export order: the ID-to-prediction
mapping is used, never a sorted/count-only backfill.

The original NMS observation hook still exports the exact selected fc_cls
input rows. For new ports, **every class's full post-NMS boxes and scores must
exactly equal the bound native predictions**, before each NPZ is written.
Numerical drift or wrong-role native results block extraction rather than
being relabeled. The original native pickle is a list: extraction loads this
list once to compare actual predictions; the later CPU collector does not
load it and streams one image NPZ at a time.

The completed index records the actual export revision, native evidence,
checkpoint size and `post-nms-fc-cls-input-v1` feature version. Legacy v3 indices
remain unchanged. New-port consumers require the explicit feature/native
binding; they do not upgrade old exports or infer success from counts.
Existing ROI, visualization or embedding directories are refused, including
partial directories. Investigate/retain an interrupted directory and prepare
a fresh explicitly bound output; do not delete/relabel a completed export.

## Selected221 host: preserved frozen exports and current CPU readers

The owner's2026-09-09T01:48:25 CPU evidence found the old `/mnt/shared/zechuan`
and `/mnt/SSD2_8TB/zechuan` aliases absent on221. A source-ready
`DIOR/brightness/seed_42/IRG/ema` native binding had matching logical
checkpoint/config/code/TEST identities, but raw physical reads on221 failed.
At that observation, the original BF/port job lists and plans still needed
restoration from source67. BF inputsets were240 on source and0 on target;
port checkpoints/native inputsets were216/108 on source and72/36 on target.
All inspected ROI indices were0. These are dated readiness counts, not current
restoration or ROI completion claims. Runtime pH restores only the exact
original artifacts; this code does not copy data, regenerate plans or launch GPU work.

`roi_host_binding` selects exactly one original `roi_jobs.json` entry by
seed/run ID, verifies the recorded frozen checkout SHA, and invokes that
checkout's unmodified exporter. BF uses native
`331d2131b84651f0a2930a3d53faeefad8701531`; ports use
`b87f34ef06eb85589cbc7dd4d385666c37696083`. Its frozen `aligned_roi.py` and
detector import root remain native. Only the current result reader, mapped
Path/config IO, in-memory path view and actual GPU0-4 ownership check are
overlaid. The index retains `code_commit` as the genuine frozen exporter SHA,
restores the original logical run object, and separately records
`host_binding.wrapper_code_sha`, wrapper dirty state, frozen checkout,
original export argv and physical GPU. No frozen file is modified.

After the runtime owner restores the exact original inputs, the following
are **single-job launch recipes**, not a second scheduler. `CODE` is a complete
clone of this ROI-binding delivery, `GPU` is the owner's selected free physical
device0-4, and the existing shared lock entry checks ownership/occupancy.
Do not fabricate `IRAOD_GPU_LOCKED=1` or invoke an unlocked exporter.

```bash
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
ART=/home/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908
cd "$CODE"
PYTHONPATH="$CODE" "$PY" -m experiments.comparison.host_binding lock "$GPU" -- \
  "$PY" -m experiments.comparison.roi_host_binding \
  --jobs "$ART/repo_manifests_37c65b6/roi_jobs.json" \
  --seed 43 --run-id RSAR/clean/B/ema

PYTHONPATH="$CODE" "$PY" -m experiments.comparison.host_binding lock "$GPU" -- \
  "$PY" -m experiments.comparison.roi_host_binding \
  --jobs "$ART/qualitative_manifests_b87f34e/roi_jobs.json" \
  --seed 42 --run-id DIOR/brightness/seed_42/IRG/ema
```

The wrapper does not acquire a second lock, install dependencies or change the
stored command vectors. A missing restored plan/checkpoint/native result remains
an error or pending input, never substitute inference. Source/core132 exports,
completed Student180 and ports108 quantitative outputs are reused, not replayed.
BF43/44 ROI240 and port ROI360 remain the exact required extraction scopes.
No Oracle/ablation qualitative job is added.

CPU visualization, joint embedding, collection and accepted-report reuse must
load **this delivery's** `result_completion`, `joint_tsne`,
`complete_qualitative` and `report_qualitative`, not the frozen exporter copy.
Run the stored CPU modules/arguments from `CODE` with the existing CPU
interpreter; retain the original plan, run ID, output paths, cap, perplexity
and seed. For example, after the port export exists:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$CODE" PYTHONNOUSERSITE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$NATIVE_CPU_PY" -m experiments.comparison.result_completion visualize \
  --plan "$ART/qualitative_manifests_b87f34e/qualitative_seed42.json" \
  --run-id DIOR/brightness/seed_42/IRG/ema
```

These readers map physical files and compare equivalent native home/HDD/old
path aliases only on221. All non-path identity, checkpoint/code revisions,
ordered TEST IDs, feature rows, NMS arrays, thresholds and sampling remain
strict. Genuinely different files or revisions still fail. Old-host exact
path/GPU behavior remains unchanged. Completed collector reuse checks the
compact indices without reopening NPZs, and returns mapped CSV/image paths
for downstream report links.

## CPU embedding and streaming collection

Execute individual `embedding_jobs.json` commands only after their referenced
exports exist. Missing exports are dependencies, not permission to invent
points. The producer verifies identical class order/NMS/feature point/TEST IDs,
uses one reservoir per declared method, and writes the shared embedding,
point-level checkpoint/ID/ROI-row provenance, predicted-class panels and budget
labels. Memory is one image NPZ plus bounded per-method reservoirs.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m experiments.comparison.complete_qualitative \
  --plan "$NEW_QUAL_ROOT/qualitative_seed43.json" \
  --embedding-root "$NEW_QUAL_ROOT/tsne/seed_43" \
  --out-dir "$NEW_QUAL_ROOT/evidence/seed_43"
```

Use the same explicit command for seeds42/44, or the stored command vectors.
Each collector validates each NPZ and streams a CSV instead of retaining
millions of rows. It emits group/method/seed/role/budget status, a 24-row
seed-specific embedding index, a summary and an input fragment. Missing
outputs remain `partial`; malformed completed artifacts are errors. It checks
sampled feature rows, IDs, coordinates and source normalization, not just
equal counts. Qualitative completion never certifies quantitative or all-
extension completion.

The lower-level `result_completion collect` CLI also accepts these plans, but
requires an explicit fresh `--out /absolute/coverage.csv`; it does not infer a
CSV directory from the deeper seed-qualified run ID. Prefer the complete
collector above for report-ready indices and embedding evidence.

`final_report` accepts an explicit-method manifest with a `qualitative_plan`
(or its accepted `qualitative_evidence` directory) rather than forcing all
extended reports to quantitative-only. Merge **one seed-specific** collector
fragment into such a report input and keep the declared seed/method scope
visible. Existing A-F feature references may appear in the joint qualitative
plan without forcing their accepted quantitative results to be recollected;
the report records these `qualitative_reference_methods` separately. Every new
port in the qualitative plan must belong to the declared report method set.
Three seed-specific summaries/indices bind the 72 new groups;
neither one seed-specific report nor a count sum is the parent extension
completion gate. No oracle/ablation qualitative jobs are implicitly requested.

## Regression commands (no experiment results)

Local synthetic gate, no native ops or installed scikit-learn required:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_generalized_qualitative -v
```

The synthetic embedding test replaces only rendering and the estimator with
explicit test doubles; it tests the real reservoir, normalization, artifact
writer and consumer. It is not a claim that real t-SNE or native NMS ran.
An additional eleven-panel test uses real scikit-learn when available and
otherwise explicitly skips; it still uses synthetic, not experimental, features.
Parent's already-provisioned **CPU native** environment should run:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$NATIVE_CPU_PY" -m unittest \
  tools.tests.test_generalized_qualitative \
  tools.tests.test_aligned_roi_completion \
  tools.tests.test_final_qualitative_integration \
  tools.tests.test_comparison_report \
  tools.tests.test_nontraining_extensions -v
```

The target overlay's CPU regression first failed on absent old-prefix plan
reads and falsely pending native evidence. Its acceptance gate exercises both
actual frozen exporter files and frozen ROI capture code with explicitly
synthetic CPU model/NMS doubles, then validates NPZ arrays, visualization,
embedding and compact collector reuse. It is not real detector/GPU evidence.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest \
  tools.tests.test_roi_host_binding tools.tests.test_generalized_qualitative \
  tools.tests.test_host_binding.HostBindingTest.test_stdlib_only_import \
  tools.tests.test_host_binding.HostBindingTest.test_exact_host_prefixes_and_scientific_values
```

Before committing/pushing this scoped branch, run the documented
[staged HEAD-diff secrets gate](ORACLE_TRAINING.md#integration-pre-push-secrets-gate).
No environment installation, remote operation, GPU execution or production
metadata preparation is part of this code delivery.
