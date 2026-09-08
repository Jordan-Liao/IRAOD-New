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
directory), and TEST image directory. This detects missing inputs such as RSAR
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
evaluation output, and never retries Student training. Staged AASFOD training
retry is explicitly blocked: its method directory also holds the required
TSD split, so it needs owner-specific recovery rather than generic relocation.
Conflicting completed artifacts also require owner resolution, not deletion.

At startup, after canonical discovery and under the selected model's cell lock,
a retry refuses any live canonical job for that model. It renames only the
selected failed method directory (train) or role-specific eval directory (eval),
and that phase's queue/source-queue wrapper statuses, to adjacent
`NAME.finite-retry-NEW_RUN_BASENAME` archives. Existing archives are never
overwritten. Native runners then use their **unchanged original destinations
and frozen bindings**. Use a unique run basename. Archive movement is journaled
in `NEW_RUN/recovery/*.json`; if interrupted or an OS rename fails, preserve the
`planned` journal and have the owner resolve its listed moves before another
retry. No automatic rollback or destructive cleanup occurs.

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
