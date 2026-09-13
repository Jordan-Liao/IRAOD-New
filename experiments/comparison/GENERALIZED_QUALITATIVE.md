# Explicit ports: full TEST ROI, fixed views and grouped joint t-SNE

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

## Bind pending B-F seed43/44 exports to native evaluations

The existing B-F seeded plans already have unique output roots. Their legacy
run objects do not contain native prediction bindings, however, and must not
be used for new unbound exports. New B-F seed43/44 extraction now rejects
missing native bindings before GPU/runtime imports. Seed42 legacy readers and
completed exports remain unchanged.

After the operator has located/staged the authentic native evaluation, prepare
only the missing group(s), using the existing CPU environment:

```bash
CUDA_VISIBLE_DEVICES="" "$NATIVE_CPU_PY" -m experiments.comparison.extension_manifest \
  prepare-bf-qualitative \
  --bf-plan "$EXISTING_SEED43_PLAN" \
  --native-report "$AUTHENTIC_CORE_REPORT" --core-paths "$ORIGINAL_CORE_PATHS" \
  --native-eval "$EXACT_NATIVE_EVAL_DIR" \
  --eval-code "$EXPORT_CODE" --python "$NATIVE_GPU_PY" \
  --physical-gpu "$OWNED_PHYSICAL_GPU" --out-dir "$NEW_BF_METADATA"
```

Repeat `--bf-plan` for seed44, `--native-report` for the accepted Student report,
and `--native-eval` for additional explicitly selected EMA/Student groups.
Use the exact native directory spelling recorded in the report. The authentic
`iraod-comparison-report-v1` report's `raw_results` entry supplies the
dataset/domain/seed/method/role, source identity and training/evaluation
revisions; its verified checkpoint and source-provenance entries must agree.
Historical B-F evaluations have no `execution.json`: the binding explicitly
records the report path, producer SHA and row index instead of inventing that
file or falling back from a missing port execution record. The original
`xaf_s424344/paths.py` must resolve the same final role-specific checkpoint.
The preparation-time training revision is the actual native producer, not an
invented earlier queue revision. The export SHA comes from `--eval-code`.

The shared `native_binding_evidence` check must pass before publishing:
checkpoint, config, successful final status, sidecar code/TEST IDs/order,
split and domain must match the existing run. Missing or incompatible native
metadata is an explicit preparation error, not permission to reconstruct an
`execution.json`, weaken equality, or launch the legacy plan. The original
port execution-record path remains supported unchanged. Staging and runtime
compatibility remain the operator's responsibility.

Original report/sidecar strings are retained. A trailing directory slash is
equivalent; different absolute namespaces are accepted only when actual
filesystem aliases identify the same files/directories (`samefile`), or through
the explicit verified path binding below. There is no implicit prefix or suffix
remapping. This code neither creates aliases nor rewrites historical metadata. Historical report
checkpoint byte counts are checked when recorded. The accepted Student report
instead records its verified final checkpoint's exact paired EMA path; it did
not retain a historical byte-count field. Every new B-F binding separately
records and rechecks the selected checkpoint's current byte count. Transfer
hashes, when independently verified by the operator, remain transfer evidence
rather than invented historical report fields.

This writes only new `qualitative_seed43.json` / `qualitative_seed44.json`
metadata and `roi_jobs.json`. Selected runs gain `native_prediction`,
`export_code_sha` and `allowed_gpus`; their original output paths and all
scientific fields remain unchanged. All unselected run objects are retained
exactly. Existing selected output directories are refused; preserve completed
or quarantined unverified files rather than deleting/resuming them. The
metadata destination must also be new.

### Explicit relocated native inputs

For an already-prepared, still-pending port or native-bound B-F run, create a
separate plan with the shared CPU/native-consumer binding:

```bash
python -m experiments.comparison.result_completion bind-native-paths \
  --plan EXISTING_PLAN --run-id EXACT_RUN_ID \
  --binding APPROVED_PATH_BINDING_JSON \
  --out NEW_PLAN_JSON
```

The binding is explicit operator evidence, not a discovered/default mapping:

```json
{
  "resolver": {"path": "/absolute/reviewed/host_binding.py", "sha256": "APPROVED_RESOLVER_SHA256"},
  "artifacts": {
    "checkpoint": {"bytes": 381069439, "sha256": "VERIFIED_TRANSFER_SHA256"},
    "config": {"bytes": 11637, "sha256": "VERIFIED_TRANSFER_SHA256"},
    "execution_json": {"bytes": 3348, "sha256": "VERIFIED_TRANSFER_SHA256"},
    "prediction_sidecar": {"bytes": 2067394, "sha256": "VERIFIED_TRANSFER_SHA256"}
  }
}
```

Use the independently verified source-to-staged byte counts and hashes, not
newly invented historical fields. B-F `comparison-report-v1` bindings use
`native_report` instead of `execution_json`. The resolver must expose
`map_path`; its exact file hash is checked before importing it. A reviewed
host-specific resolver may map the recorded paths to existing selected
files/directories, but an undeclared namespace or missing mapped path is not
equivalent.

Preparation rechecks all four artifact identities and requires
`native_binding_evidence(...)["status"] == "complete"` before publishing.
The command returns JSON with that explicit status and the checked native
evidence; absence of an exception alone is not admission. The new plan embeds this same binding in
the selected run's `native_prediction.path_binding` and binds that run's
export SHA to the preparation module's own checkout (not a selectable old
exporter). All other run objects, native history and scientific identities
remain unchanged. Existing selected output directories and plan destinations
are refused.

Run the original `extract_roi_pre_fc_cls --plan NEW_PLAN_JSON --run-id EXACT_RUN_ID
--physical-gpu OWNED_GPU` from the new export checkout. The actual consumer
and subsequent `load_export` recheck the resolver, content identities, tuple,
source/code revisions, full TEST IDs and split before accepting the binding.
The forward-time native prediction equality check is unchanged. CPU
`complete` means native inputs are ready, not that a GPU ROI export completed.
Unbound legacy plans/readers retain their previous behavior; no frozen
resolver, source asset, numerical-diagnostic file or completed output changes.

### Absolute native-wrapper entry on host183

An absolute Python script does not automatically put its working directory at
the package import root. The audited `host_binding.py native ABSOLUTE_SCRIPT`
entry adds the consumer script's directory, not the checkout root. A successful
CPU binding gate therefore does not prove that the actual consumer can import
`experiments.comparison.result_completion`.

Use the versioned `dior_recovery/host183_roi_entry.sh` envelope with `bash`.
It fixes cwd and sets `PYTHONPATH` to only the accepted860 checkout **before**
Python starts, preserving the existing anaconda interpreter, audited183
wrapper and native arguments. It does not edit the frozen consumer, alter the
resolver, force runtime-ready, create an environment or change model settings.
The existing runtime helper copies this package-root environment if it re-execs.

```bash
# CPU import-boundary check only; no plan/data/model work.
CUDA_VISIBLE_DEVICES= bash /VERSIONED_CODE/experiments/comparison/dior_recovery/host183_roi_entry.sh --help

# Only after separate operator/parent GPU admission, not an automatic retry.
IRAOD_GPU_LOCKED=1 CUDA_VISIBLE_DEVICES=OWNED_GPU \
bash /VERSIONED_CODE/experiments/comparison/dior_recovery/host183_roi_entry.sh \
  --plan /home/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908/qualitative_manifests_roi183_860d4a0/qualitative_seed42.irg-brightness-ema.json \
  --run-id DIOR/brightness/seed_42/IRG/ema --physical-gpu OWNED_GPU
```

Validate the actual native `--help` return code and Python import trace:
`experiments.comparison.result_completion` must load from that accepted860
checkout, not an old331d checkout or installed package. `PYTHONVERBOSE=1`
records this import origin without changing the script's package-search
semantics. This narrow entry check proves neither configuration/native-binding
completion nor strict forward prediction equality; those remain separate.

### Authorized host183 local-native reference pilot

At `2026-09-13T18:33:05.691Z` the user accepted the single-case pilot
`iraod-roi183-native-reference-pilot-20260913`: fixed
`DIOR/brightness/42/IRG/ema`, one predeclared host183 GPU, one fresh11738-image
native TEST reference solely for ROI, then one full11738-image ROI pass.
The same checkpoint/config/source/split/model protocol is retained. This is
not a rollout to other cases or permission to select a better TEST score.

The operator alone owns at most2 GPU-bearing command starts and1800 cumulative
allocated GPU-wall seconds, including loading, held-GPU metadata gaps,
verification waits and teardown. Startup failure consumes a start; no retries
or old-grant renewal. Old67 predictions/execution, PR25 raw900 metrics and
failed outputs remain immutable. The canonical720 ROI denominator is unchanged.

Invoke `roi183_pilot_native.py` by absolute path with the selected GPU integer.
It uses the existing host-aware7ae `evaluate_binding` (not the fixed4-7 writer),
establishes its package/runtime context, and records the actual frozen331d
evaluator identity. `native_pass.json` appears only after the real producer
returns its full AP/status/sidecar checks.

After native success, run `roi183_pilot_bind.py` on CPU with the same GPU integer.
It creates one versioned plan from the real new reference, computes only the
new execution/sidecar content proofs, reuses verified checkpoint/config proof,
and preserves unselected runs. It writes `roi_entry.sh`, which calls the
already-tested host183 envelope and frozen860 consumer. Do not insert repeated
old preflights; the existing producer and consumer checks remain authoritative.

Both stages use fresh directories under
`roi183_native_reference_pilot_20260913`. Strict `np.array_equal` to the new
reference is unchanged. Count this existing ROI key once only after complete
validated output; new local native metrics are not canonical TEST replacements.

Only the operator executes a selected job's `export_argv` in its recorded
`cwd`, under the existing real GPU lock and matching single-device visibility.
The command includes `--physical-gpu`; native prediction equality still runs
for every exported image. Other unbound B-F references in a partial derived
plan are not jobs and cannot be extracted. `native_inputs_ready` means metadata
binding passed, not that ROI extraction or scientific validation is complete.

## Actual extraction and artifact binding

Only the compute owner executes `export_argv` in its recorded `cwd` with its
actual GPU lock (`IRAOD_GPU_LOCKED=1`) and `CUDA_VISIBLE_DEVICES` set to exactly
one physical GPU4,5,6. New bindings and the port exporter reject foreign GPU7
and multi-GPU export visibility. A port run binds `eval_config`, final EMA or
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

No environment installation, remote operation, GPU execution, production
metadata preparation, commit or push is part of this worker delivery.
