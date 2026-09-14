# Explicit extension quantitative consumer (CPU only)

This is **not** the original 192-cell A–F publication or the Student180 export.
It reads existing artifacts without changing source weights, producers, queues,
or reports. Collection and report build both require new output directories.
No model launch, remote operation, dependency installation, RoI extraction or
t-SNE is performed.

## Declare the planned scope before looking at completed results

Create a consumer-owned `scope.json`, for example:

```json
{
  "methods": ["A", "IRG", "LPLD", "SFUT"],
  "roles": ["ema", "student"]
}
```

`A` must appear once, first. The approved additional identifiers are `IRG`,
`LPLD`, `SFUT`, `AASFOD`, `SFYOLO`, `B_REG`, `F_text_only`, `F_veto_only`,
`LoRA-CGA`, and `LoRA-CGA+VLST`. The last two are the explicit approved oracle
names, not the obsolete ambiguous `G_oracle`. B–F may also be explicitly
declared. Methods never come from whichever runtime rows happen to be finished.
Omitted methods are outside this report, **not** completed parent work.

Every declared adaptation method expects 36 cells **per declared role**:
RSAR eight domains + DIOR four domains, each with seeds42/43/44. The same
12 accepted A/source42 measurements are reused once, regardless of method or
role count. The example therefore expects228 rows, not432 or216.

## Exact CLI

From the integration checkout, with mounted/copied artifacts accessible at
their bound paths:

```bash
/tmp/iraod-int-venv/bin/python -m experiments.comparison.collect_extension_report \
  --scope /absolute/consumer/scope.json \
  --runtime /absolute/irg_queue/runtime.json \
  --runtime /absolute/lpld_queue/runtime.json \
  --runtime /absolute/sfut_queue/runtime.json \
  --core-report /absolute/accepted_core/report.json \
  --out-dir /absolute/new-extension-collection

/tmp/iraod-int-venv/bin/python -m experiments.comparison.final_report \
  --manifest /absolute/new-extension-collection/report-manifest.json \
  --out-dir /absolute/new-extension-report --metadata-only

/tmp/iraod-int-venv/bin/python -m experiments.comparison.final_report \
  --render-dir /absolute/new-extension-report \
  --docx-python /tmp/iraod-int-venv/bin/python
```

Alternatively replace `--metadata-only` with `--docx-python ...` to collect
and render in one build. Render only the newly owned extension report, never
the accepted core directory. It writes CSV tables, numeric JSON, separate
comparison-group/role LaTeX tables and PNG/PDF figures, and a method-aware
Chinese DOCX. LaTeX preserves missing complete seed blocks as `--`; extended
budgets and supervised oracles never share a table with common one-epoch methods.
`--runtime` is repeatable and may be omitted for an honest entirely-unbound
planned matrix.

## Bindings and evidence

The collector reads the existing `extension_training` `runtime.json` `cells`
mapping (EMA), plus an optional **explicit** `student_cells` mapping in the
native evaluation format. A binding contains dataset/domain/method/seed/role,
source_id, source_checkpoint, checkpoint, config, training_code_sha, eval_dir;
the runtime contains evaluation_code_sha. EMA uses `eval_config` for inference
when emitted separately from the training config. SFYOLO also requires
`final_checkpoint_iteration`. `checkpoint_bytes`, when supplied, must match.

Student paths are **not** inferred from `student_checkpoint`, EMA `eval_dir`,
or the old B–F queue. Ports without actual Student evaluation bindings remain
`not_started`; their expected rows are retained. The native owner may supply
an explicit Student runtime with the fields above once it exists; this tool
does not create a queue or claim an evaluation happened.

Checkpoints must be existing nonempty final files, bound to the accepted source.
Common one-epoch labels are RSAR266/DIOR185; SFYOLO is RSAR531/DIOR369.
EMA names have `_ema`; Student names do not. Collection does not deserialize
model checkpoints or independently attest their training semantics.
It records the native eval `execution.json` and actual training/evaluation
code identities. Actual training SHA comes from that execution, not a stale
prepared-runtime commit. `final_report` inspects the actual full-precision
`metric.mAP`, class AP table, successful final eval status, prediction pickle,
and same-inference ordered native-ID sidecar, including checkpoint/config/code
identity. Counts alone or producer status text cannot complete a cell.

There must be **exactly one** `eval_*.json` in the bound directory. Missing or
ambiguous metrics stay incomplete: no retry selection by mtime, no core eval
directory substitution, and no fallback search. Duplicate bindings and wrong
source/final checkpoint identities are rejected. Artifacts must be trusted
owner-generated pickles; never use untrusted prediction files.
`collection_inventory.json` is only a binding inventory, not metric acceptance.

## Statistics and remaining scope

Manifest-declared methods define planned Holm families per dataset/role/metric.
Incomplete methods keep their slots; no adjusted p until the family is complete.
The existing seed-block/three-seed, fixed-source, low-power tests are unchanged.

* `common_one_epoch`: B–F, IRG/LPLD/SFUT/AASFOD and the three approved ablations
  when declared. AASFOD remains a disclosed budget-rescaled port.
* `extended_two_epoch_TAM`: SFYOLO only; separate tables, figures and tests,
  never pooled or ranked with one-epoch methods.
* `target_supervised_appendix`: the two named oracle methods when declared,
  never strict-main evidence. Bindings are required; no oracle path guessing.

Tables/figures follow declared method order, not performance ranking. Fixed A
may be shown as a reference in multiple groups but is stored only once per
domain. Default manifests without `methods` retain the original ABCDEF matrix,
five-test B–F families, historical seed42 locks and legacy figure filenames.
Historical CSV copies and source-producer corrections remain read-only.

Extension builds always have parent `status: partial`, even if
`quantitative_status: complete`. Full TEST RoI, fixed visualization and joint
t-SNE are **pending, not collected**; no inherited132-group/192-cell qualitative
denominator or full-RoI claim is used. Native detector-stack acceptance remains
with the parent; these CPU fixtures do not replace it.

Focused local check:

```bash
/tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_extension_report tools.tests.test_report_collection
```

The existing `test_comparison_report` and `test_report_publication` suites
cannot import in this local environment: both reach
`tools/tests/test_aligned_roi_completion.py:15`, then fail with
`ModuleNotFoundError: No module named 'mmrotate'`. Run those unchanged native
fixtures in the parent's existing detector environment after integration;
do not install or emulate the detector stack for this consumer task.
