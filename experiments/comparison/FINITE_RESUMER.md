# Finite event-driven xaf resumption

This is a code-only replacement entry point. It does not replace live queue
scripts, change the training/evaluation checkouts, or launch a GPU until the
coordinator explicitly invokes it.

## Observed defects

The actual `resume_empty_gpu.sh` waits after every train and every eval, then
finishes the entire B-C-D list before considering E/F. CPU replay of that
actual shell script with four free cards produced peak concurrency **one**.
The current double-GPU API is also different from its caller:

```text
run_train_1gpu.sh GPU DS DOMAIN SEED METHOD
run_train_2gpu.sh GPU_PAIR PORT DS DOMAIN SEED METHOD
run_eval_full.sh GPU DS DOMAIN SEED METHOD
```

The old caller omits `PORT` and tries `4,6`, while the actual double-GPU script
accepts only **4,5 / port 29804** and **6,7 / port 29806**. The replacement uses
these exact APIs/pairs and never changes their commands or hyperparameters.
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
  shared GPU flock being available. Only physical GPUs4-7 are accepted.
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
resumer and coordinate the auxiliary producer's handoff. Keep all already
running canonical GPU sessions, including the GPU5 auxiliary instances.
If an auxiliary owner retains an unstarted cell, exclude that cell from this
invocation until ownership transfers; do not run two producers for the same
pending work.

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
There are no automatic retries in an invocation. If no GPU group can be
allocated and no tracked active task can produce an exit event, the finite
process returns `blocked` immediately. Return codes: complete0, failed1,
blocked2 (individual worker lock conflicts use75).

On producer interruption, active GPU tmux jobs are preserved. The next
invocation adopts them. For any new cutover, use the recorded producer PID;
do not terminate worker sessions or a whole process tree.

## CPU regression

```bash
CUDA_VISIBLE_DEVICES="" python3 -m unittest tools.tests.test_finite_resumer -v
```

The tests run the actual entry, tmux, kernel flocks and pidfd exit loop against
temporary CPU runners. They exercise4-card mixed packing, simultaneous pairs,
immediate refill before eval, legacy GPU5 canonical adoption, producer
replacement, cell uniqueness across different GPU locks, success skipping,
bad terminal/ID rejection and blocked-with-no-active-work. No real training,
GPU probing or live queue is used by the tests.
