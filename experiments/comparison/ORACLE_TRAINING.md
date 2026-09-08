# DIOR oracle24: CPU preparation, then owner-controlled execution

This appendix-only **Target-supervised** queue contains exactly `LoRA-CGA` and
`LoRA-CGA+VLST`, DIOR `clean/brightness/contrast/cloudy`, seeds `42/43/44`.
It does not depend on an RSAR adapter, read RSAR images/weights, train a source,
or modify Q752, original queues, source configs, adapters or report aggregates.
Inherited D/F Python recipes may retain RSAR names; these are code, not RSAR
artifact inputs. No detector performance is established by preparation.

## Read-only inputs and CPU command

Run from the delivered checkout on **73F3-5xA6000-221 / endpoint20138**.
The owner supplies the existing completed 192-cell core report, accepted base
plan, core `paths.py` (including `dior_cfg("D")`, `dior_cfg("F")`, source A and
the adjacent shared `with_gpu_lock.sh`), and unchanged `331d2131` evaluation
checkout. Source A checkpoint, its resolved DIOR teacher config, selected
SARCLIP base and DIOR VAL/TEST image directories must already exist. TEST
annotation paths are passed through for later native evaluation, never read
for adaptation. No downloads, reconciliation or installation occur.

The supervisor has reported the following completed DIOR adapter; the command
still calls the existing `inspect_oracle_adapter()` on its **actual payload**.
The owner retains final validation/provenance responsibility. `SARCLIP_BASE`
must identify that payload's base initialization, not a replacement checkpoint.
Existing host-binding aliases are accepted; a genuinely different base, partial
adapter, wrong dataset, invalid factors or wrong recipe fails admission.

Set `BASE_PLAN`, `CORE_REPORT`, `CORE_PATHS`, `EVAL_CODE`, and `SARCLIP_BASE`
to those existing inputs. Set `QUEUE` and `ARTIFACT_ROOT` to **new, separate,
non-nested directories** under `/home/zechuan/iraod_artifacts`; neither may
already exist. These are the complete arguments:

```bash
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
ADAPTER=/home/zechuan/iraod_artifacts/FORMAL_DIOR_LoRA_adapter_0fd539a_20138_20260908T183058Z/lora_dior.pth
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m experiments.comparison.oracle_training \
  --base-plan "$BASE_PLAN" --core-report "$CORE_REPORT" \
  --core-paths "$CORE_PATHS" --eval-code "$EVAL_CODE" --python "$PY" \
  --adapter "$ADAPTER" --sarclip-base "$SARCLIP_BASE" \
  --out-dir "$QUEUE" --artifact-root "$ARTIFACT_ROOT"
```

Only the new metadata directory is written: `runtime.json`, `cells.json`,
two executable resolved configs, `paths.py`, `train.list`, `eval.list` and
existing-pipeline runner/lock wrappers. Each list has 24 rows. The formal
artifact root remains absent. No GPU process is launched.

## Preserved science and routing

The configs resolve the existing core DIOR D/F recipes with custom model imports
disabled during preparation. The only method change is explicit DIOR-trained
LoRA in the existing oracle consumer. A plain source `OrientedRCNN` teacher is
converted to `OrientedRCNN_CGA` by **type only**, with every architecture field
retained; an already-CGA teacher is retained. The oracle consumer then performs
its existing checked wrapping. No source weights are replaced.

Common launch bindings preserve CGA-only **1x32**, CGA+VLST **2x16**, global32,
SGD LR `.02`, one epoch (184 DIOR updates, checkpoint filename label185),
`use_bbox_reg=False`, image-only target VAL, default iteration-start EMA `.998`,
unchanged thresholds/veto/protection and VLST alpha/blend. The resolved config
and launch environment bind the selected SARCLIP base and VLST cache explicitly,
clear ambient CGA/SARCLIP/VLST settings, and never use ambient `SARCLIP_LORA`.
Final `iter_185_ema.pth` is evaluated with the original source architecture;
`iter_185.pth` is retained but has **no Student eval or qualitative queue**.

The existing finite resumer admits these DIOR methods, checks actual adapter
prerequisites, reserves the proper width, and dispatches EMA evaluation only
after successful training and both retained checkpoint files. On the selected
host it uses GPUs0–4, pairs `0,1/29804` and `2,3/29806`, shared host locks,
the native rank bridge and existing numerical-integrity pipeline. Do not bypass
these with direct `train.py` invocations.

The owner may later union this queue using the existing `mixed_queue` CLI
(`--queue EXISTING --queue "$QUEUE" --out-dir NEW_UNION`). Original per-cell
source-queue routing is retained for training and the 24 EMA jobs. **One producer
only**: CPU preparation does not stop, replace, union or launch anything in live
Q752. Producer handoff and any finite `launch/resume --handoff-confirmed` remain
the supervisor's explicit action, preserving existing jobs and ownership.

## Local evidence boundary

`tools.tests.test_oracle_training` uses actual MMCV fixture-config resolution,
synthetic CPU LoRA payload admission, native command construction and mocked
evaluation dispatch. It checks exact grid counts, science/source preservation,
type-only wrapping, no RSAR artifact prerequisite, adapter failures, host aliases
and original-runtime routing through `mixed_queue`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_oracle_training
```

The local environment has no native `mmrotate` stack;
`tools.tests.test_oracle_isolation` cannot import there. Native detector/model
construction and actual target CPU preparation remain owner-side acceptance,
not a local pass or a claimed experiment result. Parent owns commit/push/PR;
the supervisor owns target execution.
