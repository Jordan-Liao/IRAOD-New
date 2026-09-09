# Finite event-driven xaf resumption

This is a code-only replacement entry point. It does not replace live queue
scripts, change the training/evaluation checkouts, or launch a GPU until the
coordinator explicitly invokes it.

## Input admission and selected recovery

Prepared target/mixed queues now resolve each cell through the existing
`extension_training.load_cell` route, including the original source queue's
`runtime.json`, before admission. Training requires its nonempty bound source
checkpoint, target image directory and at least the frozen `unlabeled_epoch_size`
supported images, plus existing AASFOD/SFYOLO/oracle prerequisites. Evaluation
requires its bound final checkpoint, DIOR TEST list (or RSAR annotation
directory), TEST image directory, and the original
`method_dir/execution.json`. Admission and native evaluation share
`extension_training.require_native_execution(cell)`: the file must be a JSON
object with nonempty string `training_code` and `training_code_sha` fields.
These record the actually executed training version; neither the prepared
queue identity nor a final checkpoint substitutes for missing metadata.
Absent or malformed required identity yields an explicit `eval_input_reason`
and `waiting`, before any evaluation attempt or output creation, for both
EMA and Student. Valid identity is passed unchanged to the native binding
(with the existing host path mapping); no checkout/SHA is inferred or certified.
The runtime owner restores original metadata independently of this code fix.
This detects missing inputs such as RSAR
`train/epoch_100.pth`, DIOR `ImageSets/test.txt`, and
`formal_ports_manifests_a39c832/runtime.json` omissions before an attempt.
It does not decode images, restore data, change sample IDs, reduce epoch size,
or certify complete image/annotation contents; frozen-data restoration remains
the experiment owner's responsibility. Historical non-runtime shell queues
retain their existing prerequisite interface.

Admission applies to ordinary F training as well as B_REG and its F deletions,
and resolves the correct queue route and checkpoint for **both** eval roles.

Missing inputs produce `waiting` and `train_input_reason` / `eval_input_reason`,
not a failed attempt. Admission is reconsidered at startup and tracked job/owner
exit events. The worker checks again before GPU probing/locks or runner invocation.
The producer does not probe GPUs or acquire GPU locks when no unheld work is
ready. Worker rechecks retain the existing model-serialization lock and blocked
receipt/notification, but invoke no GPU runner and create no native eval output.
Independent ready work still proceeds. With no active event source, the finite
producer terminates blocked; it does not poll for restored files. Start a new
invocation at the authorized boundary to reconsider those inputs.

### Boundary-only controls, not a live control channel

These options are consumed **only at startup by the new executable**:

| Option (repeatable except previous state) | Effect |
| --- | --- |
| `--previous-state OLD_RUN/state.json` | Carry forward exactly the same finite scope and training ownership, failed/blocked states, attempts and holds. Keep original list ordering. |
| `--hold-cell DS/DOMAIN/SEED/METHOD[/student]` | Persist a hold on new train/eval submissions for that exact cell. Never stop an already running job. |
| `--release-cell DS/DOMAIN/SEED/METHOD[/student]` | Release that hold; this does **not** clear failures or authorize retry. |
| `--retry-cell DS/DOMAIN/SEED/METHOD[/student]:train` or `:eval` | With previous state, authorize one new attempt for only that failed/blocked phase. A hold still applies. No wildcards or blanket retry. |

Use the same queue, finite lists, external ownership/reservations and GPU
assignment as the existing approved invocation. Add only these options and a
**new** run directory. For example, the experiment owner can append:

```bash
--previous-state "$OLD_RUN/state.json" \
--hold-cell DIOR/brightness/43/F \
--retry-cell RSAR/clean/43/B_REG:train \
--retry-cell DIOR/clean/43/B_REG:eval
```

An eval retry requires completed training. Training retry is limited to
producer-owned EMA training, refuses any retained final checkpoint or
evaluation output, and never retries Student training. A held retry is blocked
before any archive movement; only an explicit release can remove that hold.
AASFOD supports only the pre-first-stage recovery described below, never
generic staged replay or relocation of its method/TSD directory.
Conflicting completed artifacts also require owner resolution, not deletion.

Retry ownership is checked by the producer against the finite lists and prior
ledger, before archive movement. Generated worker specs contain only execution
fields (`cell`, `phase`, `gpus`, `queue`, `receipt`, `tmux`), not producer ownership.
Mixed queues declare Student evaluation bindings in `runtime.json.student_cells`:
an exposed `eval_student_dir` function alone does not mean every model has one.
Likewise, a model's `student_checkpoint` is a training output, not a declaration
of a Student evaluation role.
Training retry checks Student outputs only for declared bindings; a missing
mapping remains an error, never an implicit empty ownership set. Legacy resolvers
without runtime metadata retain their existing Student-path checks.

At startup, after canonical discovery and under the selected model's cell lock,
a retry refuses any live canonical job for that model. It renames only the
selected failed method directory (ordinary train), pre-stage AASFOD outputs
(below), or role-specific eval directory (eval),
and that phase's queue/source-queue wrapper statuses, to adjacent
`NAME.finite-retry-NEW_RUN_BASENAME` archives. Existing archives are never
overwritten. Native runners then use their **unchanged original destinations
and frozen bindings**. Use a unique run basename. Archive movement is journaled
in `NEW_RUN/recovery/*.json`; interruption leaves a `planned` journal, and an OS
rename failure records `archive_failed` and its error. Either is non-success:
preserve the journal and have the owner resolve its listed moves before another
retry. Already moved files remain archived; no automatic rollback or destructive
cleanup occurs.

The old ledger, job specs, receipts and logs are untouched. The new run also
retains `previous_state.json`, accumulated reasons and per-cell `attempts`
(including retry authorization and archive locations). Old-code ledgers without
an attempts array remain preserved in full. Unselected failures remain failures;
completed cells are not rerun; new attempts write new run-local job/receipt/log
files. Existing failed/partial native destinations are blocked even when a
caller omits previous state, rather than silently overwritten. Always carry
`--previous-state` forward to retain failed attempts that created no native files.

To release the example hold at a later approved boundary, retain the same
finite lists and add `--previous-state "$LAST_RUN/state.json"
--release-cell DIOR/brightness/43/F`. Add a `--retry-cell` only if that phase is
failed/blocked and explicitly authorized; waiting inputs need no retry flag.

### AASFOD: retained-TSD recovery before the first stage

The observed `RSAR/clean/42/AASFOD:train` failure was a generated-config
`ColorJitter` import error, before alignment created a stage directory or any
checkpoint. Its completed TSD (8,467 images, 20 passes, 1,693 similar and 6,774
dissimilar; the owner's original file is 484,231 bytes) is not a failed phase
and must not be refit. The generated-config import repair is already in the
base; this recovery does not modify the frozen model code or stage helper.

The existing selected retry now accepts only this bounded native layout:
`method_dir/work` contains exactly `alignment.py` and `stages.json`, with no
alignment/FNS directory, other stage output, or retained `.pth` anywhere in
the method directory. The native execution must record failure (nonzero
terminal), retain its invoked-not-complete status, and match every prepared
binding field except that lifecycle status, using the existing host path
equivalence. The stage ledger must match the unchanged formal budget and
`train_aasfod.stage_specs`, with only alignment invoked and no smoke/completed
stage. Missing, malformed, different, or ambiguous evidence stays blocked for
owner recovery; even an empty stage directory is outside this path.

Before any move, the existing `require_prerequisites` / `validate_split`
validates the complete bound TSD identity, algorithm, exact partition and
finite, non-tied scores. The split must remain outside the relocated outputs.
Only `work`, `train.log`, `execution.json`, `terminal_status`, and the selected
TRAIN queue/source-queue wrapper statuses are archived. The method directory,
`tsd.json` bytes and file identity, other method artifacts, all earlier
archives, original native logs/receipts and previous ledgers stay retained.
An archive collision blocks rather than overwriting history.

**Sole runtime owner, after acceptance and at the natural producer boundary:**
retain the original full finite lists, same prepared queue, external ownership,
holds, GPU assignment and all scientific bindings. Use the accepted recovery
checkout and a fresh unique run directory, carrying the final previous ledger;
append only the authorized selection:

```bash
--previous-state "$FINAL_PREVIOUS_STATE" \
--retry-cell RSAR/clean/42/AASFOD:train
```

Do not copy these options into the already running producer, reset ledgers,
move the method directory, rewrite TSD or manufacture exit-zero evidence.
Do not release unrelated NaN/OOM holds or add Student TRAIN. The unchanged
training entry starts fresh alignment at its canonical original work path,
consuming the retained TSD, then follows its original FNS budget; this is
recovery of one invalid pre-stage attempt, not new experiment budget or
midstage resume. Current independent work need not be interrupted.

CPU regression coverage uses the native failed-artifact layout and real
previous-state/archive/tmux-worker path, then the real helper and MMCV
serialization/import chain with the full-size synthetic valid split. Native
dispatch is deliberately stopped before model/GPU construction with a nonzero
result; this is recovery/config evidence, not trained checkpoints or a claim
that the owner's remote TSD bytes were inspected.

### AASFOD: accepted FNS code for fresh jobs

`--aasfod-fns-code "$FNS_CODE"` now selects the accepted model snapshot for
the FNS stage of **every newly submitted AASFOD TRAIN job**, independently of
recovery. It requires neither captures nor a retry flag for fresh jobs.
Alignment still uses the original prepared36c2053 checkout, config and source;
FNS alone uses a separate clean checkout pinned exactly to
`cbd0f75ab147fea6728f61d6fd696325cc195296`. Both stages consume the original
bound TSD. RSAR remains159+106=265 updates with266 final labels; DIOR remains
110+74=184 with185 labels. Optimizer resets, warmup, EMA cadence, batch/geometry
and initialization remain the existing `stage_specs`.

The source selection travels in each generated AASFOD TRAIN job, through
the real worker and `extension_training` CLI into the current `train_aasfod`
helper. Each native subprocess selects its own `train.py`, cwd and `PYTHONPATH`.
`stages.json` records the actual alignment and FNS code/SHA separately;
`execution.json` identifies the final FNS model source, records alignment
source separately, and keeps the original config/source/TSD identity.
Successful native EMA/Student evaluation uses the actual FNS execution SHA.
Other methods and evaluation jobs receive no model-code override.

At the sole owner's safe boundary, retain the full original invocation and
final previous ledger, use a new run directory and this updated orchestration
checkout, and add:

```bash
--aasfod-fns-code "$SEPARATE_CLEAN_CBD0F75_CHECKOUT"
```

Repeat this selection on each subsequent producer invocation that may submit
fresh AASFOD. **Omitting it retains the prepared36c2053 FNS path and its known
stack bug.** This flag is code selection, not retry permission: failed/blocked
phases are not reset, completed training is not rerun, holds are not released,
and an existing work directory is not overwritten. The normal complete-TSD
prerequisite and all native/finite locks still apply.

The35 fresh AASFOD cells temporarily held for this wiring gap may be released
only by the owner after acceptance. Carry the same full finite lists and all
other holds forward; append one `--release-cell DATASET/DOMAIN/SEED/AASFOD`
for each exact temporary fresh-cell hold being released. Do not bulk-release
NaN/OOM/Student exclusions, remove previous state, or turn a failed attempt
into a fresh row. TSD completions can then make the released fresh jobs ready
normally. No `--retry-cell` is needed for those unattempted jobs.

For the one retained-alignment failure, combine this flag with the explicit
retry and genuine captures below. Code selection alone, even with a selected
retry, does not bypass the old pre-stage guard or authorize alignment replay.

### AASFOD: FNS-only continuation after completed alignment

The later `RSAR/clean/42/AASFOD` mixed-size FNS failure is **not** eligible
for the preceding pre-stage retry. That guard remains unchanged for every
ordinary retry. A separate explicit selection accepts the now-available
original completed-alignment/FNS-failure captures described in
[AASFOD.md](extensions/AASFOD.md#mixed-size-fns-batching-and-explicit-post-alignment-continuation).

**Sole runtime owner, only after acceptance at a safe producer boundary:**
keep the same queue, full finite train/eval lists, GPU assignments, external
ownership/reservations, NaN/OOM holds and Student training exclusions.
Use this delivery's orchestration checkout, a NEW run directory, and the final
previous ledger. `FNS_CODE` must be a separate clean checkout at exact
`cbd0f75ab147fea6728f61d6fd696325cc195296`; it is not the frozen36c2053 checkout
and not this later orchestration branch HEAD. The full delivery bundle contains
both commits. `FNS_EVIDENCE` is the original read-only capture directory, with
`EVIDENCE.json`, `checkpoint_evidence.json`, `owner_console_evidence.json`,
`work/stages.json`, both resolved stage configs and both native stage logs.
Keep captures unchanged; retain original checkpoint payloads at their bound
locations. No new per-stage exit file is required or accepted as a substitute.

Append these options to the owner's unchanged finite invocation:

```bash
PYTHONPATH="$CONTINUATION_CODE" "$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$Q" --run-dir "$NEW_RUN" \
  --train-list "$ORIGINAL_TRAIN_LIST" --eval-list "$ORIGINAL_EVAL_LIST" \
  --previous-state "$FINAL_PREVIOUS_STATE" \
  --retry-cell RSAR/clean/42/AASFOD:train \
  --aasfod-fns-evidence "$FNS_EVIDENCE" \
  --aasfod-fns-code "$FNS_CODE" \
  --gpus "$ORIGINAL_GPU_SET" --handoff-confirmed
```

Retain any additional original lists, scientific hold flags and external-owner
arguments unchanged. Release only the explicitly authorized temporary
fresh-AASFOD holds described above. Do not edit previous state, stop healthy
jobs or invoke a helper outside the selected finite retry to make it pass.
The selection requires a producer-owned failed/blocked TRAIN row, no live
canonical model job and the existing nonblocking cell lock. Completed final
checkpoints/evaluations remain non-retriable. All unselected ledger rows,
attempts, holds and artifacts retain their existing behavior.

Before moving anything, the validator checks the bound159/106/265 budget,
original stage invocations and resolved configs, CPU checkpoint metadata
(without rereading large tensors), native/owner logs and the retained complete
TSD's identity/size/hash. Only `work/fns`, `work/fns.py`, `work/stages.json`,
root `train.log`, `execution.json`, `terminal_status`, and selected TRAIN
queue/source-wrapper statuses are renamed to the existing adjacent
`.finite-retry-NEW_RUN_NAME` archives. `work/alignment`, `work/alignment.py`,
TSD, the method directory and all earlier archives remain in place. The
existing recovery journal records every move and the explicit evidence/model
bindings; archive collisions or partial archive failures do not authorize work.

The job passes its archived recovery record through the real finite worker
and `extension_training` into `train_aasfod`. The existing-work exception is
local to that selected record, not a global pending-state reset. The native
entry writes a new execution record and real terminal status; only successful
FNS outputs produce final aliases and permit a successful worker receipt,
ordinary training adoption and EMA/Student evaluation. The final execution
identifies accepted FNS model code honestly and retains original alignment
provenance. If both the code flag and recovery record are supplied, they must
select the same model checkout. Never manufacture a success status to clear
an older failure.

If interrupted or blocked, preserve the new run's `previous_state.json`,
`state.json` and `recovery/xaf-RSAR-clean-42-AASFOD.json`. Its move list is the
rollback map; restore only under the sole owner's lock and only if no new
outputs conflict. A newly entered or failed FNS attempt is not automatically
retryable through this interface. A replacement producer can adopt an existing
canonical continuation worker through its persisted job/receipt, or recognize
genuine completed finals normally; never rearchive an active attempt.

### Restored, never-attempted Student dependencies

A future resumer can reconsider an eval-only Student row whose persisted
`train` and `eval` are both `blocked`, with no attempts or adoption history
and no hold. This is dependency re-evidence, **not an experiment retry**.
It calls the unchanged training-evidence path: both nonempty final EMA and
Student checkpoints, the original successful native training terminal, and
no conflicting source-queue/current-queue wrapper status are still required.
Missing or conflicting evidence leaves the row blocked. Do not manufacture
an exit-zero terminal or replace this check with checkpoint sizes.

Only after that original evidence is complete does the dependency become
`train=complete`; no TRAIN ownership or TRAIN worker is created. The existing
native EVAL evidence/admission path then recognizes a genuinely completed
Student evaluation, queues only a pending Student EVAL, or preserves a
partial/conflicting native destination as blocked. Missing EVAL inputs still
wait through ordinary admission. Recorded attempts, prior adoption, failed
evaluations and held references are not reset or re-admitted by this path.
Original ledger bytes, reasons, attempts, completed models and artifacts are
retained; there is no retry archive, ledger edit, new flag or file-arrival poll.
Re-evidence uses startup and the existing tracked exit-event boundaries.

The actual2026-09-09 control was `RSAR/point_target/43/IRG/student:eval`.
Its persisted row was eval-only, blocked/blocked with `attempts=[]`;
both final files were present (380,953,727-byte EMA and411,401,179-byte
Student), but the real training `terminal_status` was missing on221.
Source had the original `tmux_wrap_exit=0`; the origin wrapper was successful
and the target queue wrapper did not conflict. **The blocked state was correct
until the owner restored the original evidence.** The regression isolates
the separate stale-state bug after that restoration, without relaxing evidence.

Only this matched control is certified finite by the prior bounded diagnosis
at `a013744d926f265c72f2c00cb3adc19015748002`,
`results/extensions/ports108_ema/artifact_manifest.json:numerical_diagnosis.control`
(Student/EMA/optimizer all0 nonfinite tensors). The67 nonexcluded Student
records are **repair candidates, not67 certified-healthy models**. No new
tensors are inspected. The five diagnosed NaN Student keys and their ten ROI
roles remain excluded by the owner's existing policy; the library contains
no scientific exclusion list and does not control ROI selection.

**Future natural boundary only:** let the current producer reach its authorized
terminal boundary, preserve its final ledger, and use a new checkout/run
directory with the **same queue and full original train/eval lists**, original
external ownership/reservations and GPU assignment. Do not make a67-row scope,
turn the eval list into TRAIN, inject code into the current producer, or use
`--retry-cell`/ledger resets to release these dependencies. The owner supplies
the actual `FINAL_PREVIOUS_STATE`, `NEW_RUN` and original invocation bindings.
Persisted holds carry forward; the existing hold flags add the five explicit
Student exclusions, without a blanket F hold:

```bash
PYTHONPATH="$FUTURE_CODE" "$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$Q" --run-dir "$NEW_RUN" \
  --train-list "$ORIGINAL_TRAIN_LIST" --eval-list "$ORIGINAL_EVAL_LIST" \
  --previous-state "$FINAL_PREVIOUS_STATE" \
  --gpus "$ORIGINAL_GPU_SET" --handoff-confirmed \
  --hold-cell RSAR/noise_suppression/44/IRG/student \
  --hold-cell RSAR/noise_suppression/44/LPLD/student \
  --hold-cell RSAR/noise_suppression/44/SFUT/student \
  --hold-cell RSAR/point_target/44/IRG/student \
  --hold-cell RSAR/point_target/44/LPLD/student
```

Retain any original external-list/reservation arguments unchanged. pH alone
restores inputs and exercises the real control later; this delivery does not
deploy, probe the host, launch GPU work or certify new models.

The narrow CPU regression first failed with the real fixture resumer still
blocked after restoring its original terminal. The affected module gate
exercises missing/conflicting terminals and wrappers, a held exclusion analog,
prior attempts/adoptions, retained completed models, partial native output,
ordinary admission and completed-EVAL reuse using CPU-only fixture runners:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_finite_resumer
```

Before committing/pushing the future snapshot, use the existing
[staged HEAD-diff secrets gate](ORACLE_TRAINING.md#integration-pre-push-secrets-gate).

### Producer deployment boundary

**Do not replace or inject code into the running old producer. It cannot consume
these options, and creating/editing a request or ledger does not make it do so.**
Wait for its natural terminal state, or obtain a separate explicit authorization
from the experiment owner for a safe producer-only deployment boundary. At that
boundary the owner must stop further old-producer submissions and ensure its
producer lock is released, preserve every running canonical GPU job, and start
the new runner from a separate checkout with the same frozen queue/lists and a
new run directory. Retry-selected models must have no live canonical job; other
live canonical jobs are adopted unchanged.

This code delivery does not perform that cutover; deployment, GPU operations
and data restoration remain with the experiment owner. The dated cutover
examples below are historical, not current authorization.

For owner deployment, retain the exact original train/eval lists and their role
columns. Entries are **mixed records, not a count of training jobs**; never turn
the eval list into a training list.
`--previous-state` enforces the same finite keys and training ownership.
Student `:train` retries and eval-only training are rejected, as are retries of
completed training. A hold/release never grants training ownership. Keep all
valid trained models, original attempts/failure artifacts and existing valid
F2-update smokes; do not enqueue or repeat those smokes.

After the owner restores the required **frozen** dependencies, the clean-boundary
invocation uses the new checkout, original terminal ledger and a new run
directory. This native-host example restricts new submissions to GPUs0-3,
excluding independent GPU4 work (owner-only; not executed by this delivery):

```bash
PYTHONPATH="$RUNNER_CODE" "$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$Q" --run-dir "$NEW_RUN" \
  --train-list "$ORIGINAL_TRAIN_LIST" --eval-list "$ORIGINAL_EVAL_LIST" \
  --previous-state "$OLD_RUN/state.json" \
  --gpus 0,1,2,3 --handoff-confirmed \
  --retry-cell RSAR/clean/43/B_REG:train
```

Retain any additional original list/external-ownership arguments unchanged.
Select only actual failed, owned phases from the terminal ledger; repeat
`--retry-cell` for each explicitly authorized selection. IRG/LPLD/SFUT reference
Student records remain evaluation-only. An eval selection needs completed training.
With no retry flags, failed states remain failed; with inputs still absent,
selected retries wait without a runner attempt. No option restores data, consumes
a request in the old executable, changes budgets or touches independent GPU4 work.

## Legacy-wrapper defects

The actual `resume_empty_gpu.sh` waits after every train and every eval, then
finishes the entire B-C-D list before considering E/F. CPU replay of that
actual shell script with four free cards produced peak concurrency **one**.
The legacy double-GPU wrapper API is also different from its caller:

```text
run_train_1gpu.sh GPU DS DOMAIN SEED METHOD
run_train_2gpu.sh GPU_PAIR PORT DS DOMAIN SEED METHOD
run_eval_full.sh GPU DS DOMAIN SEED METHOD
```

The old caller omits `PORT` and tries `4,6`, while the actual double-GPU script
accepts only **4,5 / port 29804** and **6,7 / port 29806**. In legacy-wrapper mode,
the replacement uses these exact APIs/pairs without changing their hyperparameters.
Global batch32, LR0.02, seeds42/43/44, single1x32/double2x16, final checkpoints,
training tree0f98 and evaluation tree331d remain unchanged.

## What changes

`experiments.comparison.finite_resumer` handles only the explicitly supplied
finite cell lists. There is no matrix expansion, daemon, fixed polling loop,
automatic model retry or new scheduling service.

- Mixed ready singles and pairs can occupy `1+1+2`; pairs alone can occupy
  `2+2`. Long C jobs precede other singles. Supported pairs are kept intact.
- A task exit immediately triggers another allocation. No unrelated task,
  BCD wave or per-cell evaluation must finish first.
- Training that fits the currently free cards has priority over evaluation.
  Native evaluation uses a free single card only when no ready training fits.
  Already running evaluations are not preempted.
- Availability requires no compute PID, memory below500MiB and the actual
  shared GPU flock being available. Legacy-wrapper mode accepts physical GPUs4-7,
  with pairs4,5 / port29804 and6,7 / port29806.
- On native host `73F3-5xA6000-221`, `host_binding` exposes GPUs0-4 and
  pairs0,1 / port29804 and2,3 / port29806. `--gpus` restricts submissions to the
  owner-authorized subset; host support does not authorize use of every card.
  Native mode invokes `extension_training.py` with the bound runtime and paths,
  not the copied legacy shell wrappers. Frozen cell settings remain unchanged.
- Native host `73F3-8x4090-134` uses the same mapped runtime/lock entry, but only
  physical GPUs4-7 and pairs4,5/29804,6,7/29806. The shared lock root is local
  to each host; owner15be must assign disjoint unstarted approved work, not
  duplicate a full-Q producer across hosts. See `HOST_BINDING.md`.
- Each independent worker holds the existing GPU lock files and an additional
  cell-wide lock (shared by train/eval and independent of GPU assignment).
  It sets `IRAOD_GPU_LOCKED=1` only while holding those actual locks, preventing
  the existing scripts from double-locking themselves.
- Canonical tmux session creation is atomic. Existing canonical jobs are
  adopted, not restarted; their actual runner arguments determine cell,
  phase and GPU assignment. Legacy adopted jobs get a producer-held cell lock.
  New workers retain their own locks if the producer is stopped.
- Linux pidfd plus `selectors` observes actual pane-process exits,
  including adopted non-child processes. No `sleep 30`, tmux completion-signal
  polling or dependency on another controller consuming a notification.

The actual host is Linux x86_64/kernel6.8. Its Conda Python3.10 omits
`os.pidfd_open`, so the runner uses the verified kernel syscall through stdlib
`ctypes` directly. No package install or polling fallback is used; unsupported
hosts return blocked.

The `launch` entry puts the producer itself in `xaf-finite-producer`, keeping
the existing tmux server alive between jobs. `resume` is its foreground body;
use `launch` for production, not a short-lived unanchored tmux server.

## Mandatory producer handoff — preserve live GPU jobs

Before starting, the coordinator must stop further submissions from the old
primary resumer. Keep all already running canonical GPU sessions. A declared
external owner may finish its explicit reservation: exclude its training
ownership with `--external-train-list`, even when those cells also occur in
the stale main lists, and monitor its lifetime with `--external-owner-session`.
Do not stop that declared external producer or its GPU jobs.

Read-only discovery of the actual old producer PIDs:

```bash
export PYTHONPATH="$RUNNER_CODE"
"$PY" - <<'PY'
from experiments.comparison.finite_resumer import old_producers
print(old_producers("/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_s424344"))
PY
```

For each returned PID, inspect its current command (`ps -p PID -o pid,ppid,args`)
and only then have the coordinator send `kill -TERM PID` to that **producer
PID**. Do not kill a process group, tmux server, `xaf-RSAR-*`, `xaf-DIOR-*` or
`xafE-*` GPU session. Do not stop the GPU5 auxiliary training.
`--handoff-confirmed` is an explicit acknowledgement of this operator step;
the entry also rejects a still-running `resume_empty_gpu.sh`. No producer or
GPU process is automatically terminated by this implementation.

All participating canonical jobs must use the same default tmux server as the
existing xaf queue. Do not switch tmux sockets/TMUX_TMPDIR in production.
Tests use an isolated server and CPU-only fake runners.

## Launch / recover from actual remaining lists

Use a separate updated runner checkout, not changes to the running training
tree. `PY` is the existing iraod Python (NumPy is needed to read native
prediction pickles). Lists are whitespace-separated `DS DOMAIN SEED METHOD`,
one cell per line; duplicates across lists are coalesced. Source A is allowed
only in an eval-only list at seed42.

```bash
Q=/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_s424344
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
export PYTHONPATH="$RUNNER_CODE"

# ONLY the coordinator/GPU owner runs this after the handoff above.
"$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$Q" --run-dir "$Q/finite_runs/NEW_ATTEMPT" \
  --train-list "$Q/resume_1gpu_missing.txt" \
  --train-list "$Q/resume_2gpu_missing.txt" \
  --gpus 4,5,6,7 --handoff-confirmed
```

The observed files contained22 single-GPU and48 double-GPU entries; these are
stale input lists, not a claim that70 models still need training. Each cell's
current terminal/final evidence is checked, so successful cells are not
retrained. A trained listed cell still gets its missing native eval.
An optional `--eval-list /actual/missing_native_eval.txt` includes already
trained eval-only work outside those lists, without authorizing new training.

### Accepted production cutover

Coordinator report at **2026-09-07 19:08 UTC** confirms production uses the
already accepted **c91bc3541b0bfa144d6bbe2550f76e8ed291b3db** checkout:

```text
/mnt/SSD2_8TB/zechuan/IRAOD-New-finite-c91bc35
Q/finite_runs/full-20260907T185024Z
```

The reported producer PID was3301246, with explicit train.list130 /
eval.list192; train92 complete +2 running +36 ready, eval141 complete +13 ready
+38 pending, blocked0. It adopted DIOR/contrast/44/D on GPU5, launched
DIOR/brightness/43/E on6/7, and excluded the foreign GPU4 occupancy.
Two old serial producer PIDs were retired and the old script archived under
history; no GPU model was stopped.

This is a dated coordinator-confirmed snapshot, not a new runtime scan.
GPU5's earlier two-C external reservation is finished, so this production
invocation needs no external-reservation flags. **Do not upgrade/replace the
running controller or launch a second producer for subsequent report-only
metadata changes.** Final quantitative statistics remain pending.

### Pre-cutover capacity history

Coordinator update at **2026-09-07 18:57 UTC**: the GPU5 auxiliary owner has
completed both DIOR/cloudy/44/C and DIOR/contrast/44/C, with final iter185
EMA/Student and successful terminals, and released GPU5/ownership. New launches
should **omit the former GPU5 external-reservation flags** and use the normal
success checks to skip both training cells.

Capacity update at **2026-09-07 18:59 UTC**: the coordinator reports an external
`beichen/python` on GPU4 (PID3259071, 57,662MiB). A free queue `gpu4.lock` is
not permission to allocate GPU4 or the 4,5 pair. The existing producer checks
real compute applications/memory before allocation, and the worker checks them
again after acquiring GPU locks. It never terminates external processes; no
new monitoring/polling loop is required.

The reported live queue tasks are DIOR/cloudy/44/D on GPU5 and
`xaf-RSAR-smart_suppression-43-E` on GPUs6/7. Adopt their current canonical
panes/PIDs without restarting them. GPU5 can become available as a single card
after its own task releases it; it is not interchangeable with a safe 4,5 pair
while GPU4 is externally occupied. RSAR seed43 clean F, chaff E/F, point_target E/F and
noise_suppression E were also reported complete and must not be retrained.
These are coordinator handoff facts, not new scans or GPU actions by this
code delivery. The producer-only cutover rule still applies.

### Historical GPU5 reservation example

During the earlier handoff, the auxiliary owner retained
**DIOR/cloudy/44/C and DIOR/contrast/44/C**.
The second canonical session may not exist yet; that does not transfer its
training ownership. Create a separate reservation list (do not edit the live
main remaining lists):

```text
DIOR cloudy 44 C
DIOR contrast 44 C
```

Add these flags to `launch`/`resume`:

```bash
--external-train-list /absolute/gpu5-external-training.txt \
--external-owner-session gpu5-xaf-pair-owner=5
```

The external list overrides `--train-list`: the new producer never submits
these two training jobs, but still adopts any existing canonical task and
queues its eval after real completion. It rechecks a reserved pending
dependency on completion events, not a timer. The existing owner session is
observed using its current pane PID (not a stale hardcoded PID), and GPU5 stays
reserved across the cloudy-to-contrast gap until the owner exits/releases it.
The observer does not acquire the owner's GPU lock or change its process.

If contrast finishes between events, the owner's exit wakes the finite loop
to validate its final files and enable eval. If the owner disappears without
finishing the reserved cell, the new producer reports blocked rather than
stealing training ownership. Keep these flags on recovery until the owner
has completed both cells or the coordinator explicitly transfers that scope.
`--handoff-confirmed` here applies to the competing old primary producer;
**do not terminate `gpu5-xaf-pair-owner`**.

Use a **new run directory** on recovery with the same actual remaining lists.
Do not delete checkpoint/prediction files to make the scheduler proceed.
Existing canonical work is adopted, including jobs from a stopped previous
finite producer; successful cells are skipped. Global producer and cell locks
are based on the actual `LOCKDIR` in `with_gpu_lock.sh`, so another queue copy
cannot bypass them by choosing a different run directory.

Output:

```text
NEW_ATTEMPT.producer.log
NEW_ATTEMPT/state.json          # producer PID, exact finite scope, active/adopted jobs, terminal states
NEW_ATTEMPT/jobs/*.json         # immutable per-invocation job specs
NEW_ATTEMPT/results/*.json      # per-job identity, allocated GPUs, rc and complete/failed/blocked
NEW_ATTEMPT/logs/*.log
```

Launching the producer is not completion. Only `state.json.status=complete`
means all cells in the supplied finite lists and their required native evals
completed. It never means the whole research/report campaign is complete.

## Strict terminal semantics

Training needs both nonempty exact final files plus the **last** training
terminal exit (and canonical wrapper exit when present). Existing final files
with conflicting or partial evidence are blocked, never blindly retrained.

Evaluation needs the last successful eval/wrapper status, class table,
one metric JSON, actual prediction count, native same-inference image IDs and
ordered records, correct checkpoint/config, and evaluation SHA331d. An old
success substring, bare file existence, repeated/incorrect IDs or incomplete
output is insufficient. Conflicting existing eval outputs are preserved and
reported blocked; the operator must resolve them or choose a fresh native
output path through the existing evaluation configuration.

A failed task blocks its own dependent eval, not unrelated ready training.
There are no automatic retries; selected boundary retries use the explicit
controls above. If no GPU group can be
allocated and no tracked active task can produce an exit event, the finite
process returns `blocked` immediately. Return codes: complete0, failed1,
blocked2 (individual worker lock conflicts use75).

On producer interruption, active GPU tmux jobs are preserved. The next
invocation adopts them. For any new cutover, use the recorded producer PID;
do not terminate worker sessions or a whole process tree.

## CPU regression

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_finite_resumer -v
```

The tests run the actual entry, tmux, kernel flocks and pidfd exit loop against
temporary CPU runners. They exercise4-card mixed packing, simultaneous pairs,
immediate refill before eval, legacy GPU5 canonical adoption, producer
replacement, cell uniqueness across different GPU locks, success skipping,
bad terminal/ID rejection, delayed external canonical creation/GPU reservation,
blocked-with-no-active-work, missing checkpoint/TEST-list/source-runtime
admission, event-triggered input recovery, selective train/eval retries,
held/released cells and retained failure/completion evidence. Use the existing
CPU environment with NumPy (prepared runtime resolution imports report helpers).
No real training,
GPU probing or live queue is used by the tests.
