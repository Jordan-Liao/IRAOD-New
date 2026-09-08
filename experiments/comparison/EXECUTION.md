# Finite detector execution bindings

For the selected `73F3-5xA6000-221` target (endpoint20138), use
`HOST_BINDING.md`: immutable copied-Q runtime path views, GPUs0-4, original
pair ports29804/29806 bound to0,1 and2,3, and the single mapped shared lock
directory. The supervisor alone deploys and executes; this is not a new
scheduler or a change to scientific1x32/2x16 budgets.

## Preserved .67 execution bindings

Current future code: `ROOT/integration_code_752a139`.
Current future queue: `ROOT/release_manifests_752a139`,180 train/360 eval
(180 EMA +180 five-port Student), with GPUs4,5,6,7 and both original F pairs.
Its scientific cell bindings and train/eval lists match the prior queue exactly.

Preserved prior code: `ROOT/integration_code_b87f34e`.
Preserved prior queue: `ROOT/release_manifests_b87f34e`,180 train/360 eval
(180 EMA +180 five-port Student), zero Student training. Detector-only and
Student-only metadata are `mixed_detector_manifests_b87f34e` and
`port_student_manifests_b87f34e`. These are actual prepared bindings, not GPU
execution or result evidence. Its three-GPU capacity binding remains unchanged.
Future metadata restores the already-authorized fourth GPU after the foreign
occupation ended; this does not change any per-cell training budget/topology.

Code-only delivery: do not modify the live a39 producer, its snapshots, any
prepared source queue, or the frozen0f98 training checkout. Parent owns
integration/deployment and producer-only cutover; preserve existing GPU workers.
There is still exactly one finite producer, using shared GPU/cell locks,
canonical sessions and pidfd completion events. No prerequisite scheduler.

## CPU mixed-queue preparation

Run from the **new isolated executor checkout** with the existing native Python.
These four prepared detector queues total180 train +180 EMA-eval cells.
The12 TAM fits are prerequisites, **not** detector rows.

```bash
ROOT=/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908
Q="$ROOT/mixed_detector_manifests_NEW_REVISION"
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
export PYTHONPATH="$RUNNER_CODE"

"$PY" -m experiments.comparison.mixed_queue \
  --queue "$ROOT/b_reg_manifests_0e5f8f7" \
  --queue "$ROOT/f_deletion_manifests_daa3992" \
  --queue "$ROOT/aasfod_manifests_36c2053" \
  --queue "$ROOT/sfyolo_manifests_84a1aeb" \
  --out-dir "$Q"
```

The output must be fresh. It references original queue runtimes and retains
every cell's original source, config, training checkout/SHA, TSD/TAM path,
checkpoint, work, eval and terminal paths. It creates no formal artifacts and
does not copy/modify scientific configs. New wrappers use this executor;
B_REG/F still invoke0f98, and AASFOD retains its prepared36c2053 code.
Additional already-prepared extension detector queues can be explicit
`--queue` inputs; overlapping detector identities are rejected.

### Explicit five-port Student nativeTEST bindings

This is CPU metadata preparation, not training or inference. `IRG_QUEUE`,
`LPLD_QUEUE`, and `SFUT_QUEUE` name the **existing prepared source queue
directories**, each with its original36 cells; they are not new Student
runtimes to be supplied by an owner. AASFOD/SFYOLO use the same prepared
queues as above. Run from the new executor checkout:

```bash
STUDENTS="$ROOT/five_port_student_manifests_NEW_REVISION"
"$PY" -m experiments.comparison.mixed_queue \
  --queue "$IRG_QUEUE" --queue "$LPLD_QUEUE" --queue "$SFUT_QUEUE" \
  --queue "$ROOT/aasfod_manifests_36c2053" \
  --queue "$ROOT/sfyolo_manifests_84a1aeb" \
  --role student \
  --method IRG --method LPLD --method SFUT --method AASFOD --method SFYOLO \
  --out-dir "$STUDENTS"

# Optional single-producer union with the default180 train +180 EMA-eval queue:
COMBINED="$ROOT/mixed_detector_student_manifests_NEW_REVISION"
"$PY" -m experiments.comparison.mixed_queue \
  --queue "$Q" --queue "$STUDENTS" --out-dir "$COMBINED"
```

The Student queue contains **0 train +180 Student eval** rows; `train.list`
is empty. The combined queue contains **180 train +360 eval** rows, not
IRG/LPLD/SFUT retraining. Each Student has width1, its original final
`student_checkpoint` (`iter_531.pth` RSAR / `iter_369.pth` DIOR for SFYOLO;
the original266/185 labels otherwise), native331 `eval_config`, and the
sibling `eval_full_DOMAIN_student_ids_v1` directory used by the generalized
qualitative plan. No B–F Student recollection, control/oracle Student rows,
ROI extraction, or inference occurs during preparation.

Runtime `cells` retains canonical EMA model references unchanged;
`student_cells` contains one explicit Student evaluation binding per model.
`source_queues` is flattened to the original prepared queues for both roles,
including when unions are inputs. Collect from the final union runtime
**instead of additionally passing its component runtimes**: passing both
would duplicate the same collector identities. Canonical references are not
additional finite jobs; only the train/eval lists authorize work.

Use `$COMBINED` in place of `$Q` in the one producer command below (or
`$STUDENTS` for Student-only evaluation). New evaluation wrappers accept
`run_eval_full.sh GPU DS DOMAIN SEED METHOD [ema|student]`; omitted role
remains EMA. At evaluation time the executor reads the original model's
`execution.json` for its actual training code path/SHA, not the prepared
code revision. The native runner still executes the original331 checkout.

EMA training, EMA evaluation and Student evaluation share the canonical
model lock (`DS/domain/seed/method`, irrespective of role) and cannot be
allocated concurrently. Existing `xafS-...` Student session names remain
unchanged. Student-only queues can adopt existing canonical model jobs,
including original source wrappers, without submitting model training.
Missing Student finals stay waiting while their corresponding EMA training
is active or waiting for TSD/TAM. Rechecks use the producer's existing
pidfd exit events only; no event source means a finite blocked exit, not a
new polling scheduler. Completed model outputs do not recheck old TSD/TAM.

New metadata permits only GPUs4,5,6,7; F uses the two original pairs
4,5 /29804 and6,7 /29806. No four-GPU model or alternate pair is introduced:
B_REG and other singles remain1x32; each F cell remains2x16 at the same LR.
The existing executor still checks actual foreign occupancy and GPU/cell locks
before allocation. GPUs0-3 remain outside the allowed set. Historical and
already-prepared metadata are not rewritten.

## Producer and prerequisites

After the parent's producer-only cutover, keeping live canonical GPU jobs:

```bash
"$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$Q" --run-dir "$Q/finite_runs/NEW_ATTEMPT" \
  --train-list "$Q/train.list" --eval-list "$Q/eval.list" \
  --gpus 4,5,6,7 --handoff-confirmed
```

Use only one `launch`, not one per input queue. Runner APIs are unchanged:
`run_train_1gpu.sh GPU DS DOMAIN SEED METHOD`,
`run_train_2gpu.sh GPU_PAIR PORT DS DOMAIN SEED METHOD`, and
`run_eval_full.sh GPU DS DOMAIN SEED METHOD [ema|student]`.

- **AASFOD:** the exact bound `tsd.json` must pass
  `aasfod_protocol.validate_split`: completed real TSD, source/domain/seed/VAL
  identity, frozen choice, and actual ordered full-VAL split.
- **SFYOLO:** the bound TAM checkpoint must pass `load_completed_tam` on CPU:
  matching dataset/domain/fit seed42, completed160k outer iterations, declared
  preprocessing, learned decoder/F1/F2 state, and strict external Oxford
  VGG16 conv1_1..conv4_1 encoder loading. Two detector epochs are unchanged.
- Missing/partial/mismatched prerequisites produce per-cell `train=waiting`
  plus `prerequisite_reason`; they do not prevent independent ready B/F or
  other admitted detector work. Checks repeat on existing tracked exit events.
  With no tracked work left the producer exits `blocked` (2), not success.
  After the existing owner finishes TSD/TAM under actual locks, resume with a
  fresh run directory. There is no timer, retry loop or new prerequisite engine.
- Both finite worker and native training entry enforce admission before
  output creation. Existing completed detector artifacts are adopted without
  requiring their old prerequisites again. Source-queue canonical wrappers
  and terminal evidence remain recognized.

## B_REG / one F-control bounded NON_RESULT smoke

These commands **execute GPUs**; none were run in this code delivery. The
parent can run them using the prepared union after deployment. Each command
acquires the actual shared GPU and formal-cell locks for its entire lifetime.

**Launcher contract:** invoke `smoke_frozen_training` directly from the
supervisor's orchestrator, without `with_gpu_lock.sh`, `flock`, or a
`finite_resumer worker` holding an outer GPU/cell lock around it. This entry
owns both locks itself and sets `IRAOD_GPU_LOCKED=1` only for its native child
after acquisition. An inherited flag never bypasses acquisition. If another
holder exists, the smoke must fail; do not unlink lock files or bypass locks.
Native detector/Student evaluation, TSD, TAM and the LoRA trainer retain their
existing outer-lock contract; do not apply the smoke contract to those entries.

The compute owner confirmed the actual16:07 F failure: its launcher held
`with_gpu_lock.sh` for4,5, while `smoke_frozen_training.run` opened and tried
to acquire the same `xaf_s424344/gpu_locks/gpu4.lock` again. This was nested
acquisition, not evidence that a GPU-use check should override a lock.
The owner's next smoke retry removes the outer wrapper; the smoke remains
the sole shared-lock acquirer. LoRA keeps its outer wrapper because it has
no internal GPU lock. Failed752 logs remain preserved.

Legacy `b_reg_manifests_0e5f8f7` cells do not contain `world_size`. Their
topology comes from the existing finite runner's B_REG single-GPU dispatch
(`Cell.width`), not from the four-GPU source-training checkpoint pathname and
not from an arbitrary safety default. The cell/config are not rewritten;
the smoke still uses1x32 and records the topology's resolution basis.
The corrected harness is deployed as `ROOT/integration_code_adb6ba4`; use it
with the existing `ROOT/release_manifests_752a139`, fresh NON_RESULT output
roots and the direct launcher contract above. No source queue or752 snapshot
was overwritten.

```bash
"$PY" -m experiments.comparison.smoke_frozen_training \
  --queue "$Q" --dataset DIOR --domain clean --seed 42 --method B_REG \
  --gpus 6 --updates 2 --out-dir "$ROOT/NON_RESULT_B_REG_NEW_ATTEMPT"

"$PY" -m experiments.comparison.smoke_frozen_training \
  --queue "$Q" --dataset DIOR --domain clean --seed 42 --method F_text_only \
  --gpus 4,5 --updates 2 --out-dir "$ROOT/NON_RESULT_F_text_only_NEW_ATTEMPT"
```

Select one F deletion, not a new sweep. `F_veto_only` is also supported.
For the original second pair use `--gpus 6,7`; the harness selects port29806.
B_REG may use any one currently free authorized GPU4-7. Updates are limited
to1–4 actual optimizer steps.
The harness runs the real frozen `train.py` and its original model imports,
source, data, resolved controls, LR schedule and iteration-start EMA.
B_REG remains1x32; F remains2x16, with its pair's original port and LR.02. A smoke-only native hook
counts actual optimizer calls and terminates after the requested update;
it does not shorten epochs, alter the sampler or flush EMA.

Outputs are fresh, separate from formal paths: `invocation.json`, native
`train.log`, config/log files in `work/`, and per-rank `rank_N.json` with real
losses, effective updates, student/EMA parameter deltas and `bounded_pass`.
One step can legitimately have zero EMA delta under iteration-start EMA.
Checkpoint saving is disabled; no formal finals, terminal_status, wrapper
success or evaluation artifacts are produced. `NON_RESULT` is not a result.

## Validation and remaining execution inputs

Targeted CPU selectors:
`tools.tests.test_extension_training`, `tools.tests.test_finite_resumer`,
`tools.tests.test_mixed_queue`, `tools.tests.test_nontraining_extensions.StudentFiniteTest`,
and the existing legacy180 preparation test below.
Use `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
Exact local gate for Student binding, native routing, old Student180
preparation, shared-model locks/waiting and existing finite API regressions:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest \
  tools.tests.test_mixed_queue tools.tests.test_extension_training \
  tools.tests.test_finite_resumer \
  tools.tests.test_nontraining_extensions.StudentFiniteTest \
  tools.tests.test_nontraining_extensions.SeededQualitativeTest.test_prepare_emits_only_new_student_and_seeded_non_source_jobs
```

Validation: **53 distinct tests passed** across the initial gate (46 passes)
and the corrected/new selectors (7 passes,6.906s). The initial gate exposed
a missing fixture directory (fixed) and the local environment's absent
`mmrotate` import in unrelated ROI/embedding fixtures. Legacy180 preparation
now uses an existing CPU-only plan fixture; detector-dependent imports remain
local to their two ROI/embedding tests, which were not executed locally.
The initial gate took36.743s. No dependencies were installed, and no GPU smoke
or native detector inference was executed.

Remaining native/GPU requirements: existing mmdet/mmrotate CUDA environment,
unchanged0f98/36c2053 and native331d checkouts, bound source checkpoints and
target VAL images, frozen SARCLIP assets for the chosen F control; completed
per-cell TSD;12 matching160k TAM fits plus the bound Oxford encoder weights.
Parent must execute the real B/F smokes and inspect per-rank evidence, then
operate the approved finite cells and native evaluations. Local CPU tests
prove routing/admission/accounting, not detector loss behavior on real data.
