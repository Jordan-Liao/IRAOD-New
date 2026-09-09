# Oracle DIOR24 or RSAR48: CPU preparation, then owner-controlled execution

The existing preparer selects one dataset per invocation: `--dataset RSAR`
produces48 cells; `--dataset DIOR` (also the unchanged default) produces24.
Each selection reads only its own dataset's adapter, source and image inputs.
Both use generated `train.list`/`eval.list`; no second runner or RSAR static
selection layer is needed.

## Integrated code delivery

`work/dior-oracle-consumers` merges accepted Oracle consumer
`0a421f0199ac3b693335f4901467c34e140f2a11` with native nested-EMA fix
`e5089cc8ea34dfb8e8695798503aadbb66c4b1db`, including the accepted finite
Student-role retry fix. The normal merge preserves both histories and all
scientific settings. PR #6 carries the combined deployment snapshot; PR #7
retains the distinct native-fix scope.

Both adapters are complete; do not refit either. The exclusive runtime owner
is Herdr `w6:pG`, session `98b171c5-ce2e-4ad6-b66a-97ab0e4c76d9`.
Only that owner prepares actual inputs and launches the authorized DIOR24 and
RSAR48 work. This code delivery does not perform SSH, remote preparation,
GPU work, payload admission or producer replacement.

Deploy the self-contained bundle as a **new clone**, not a copied worktree
with a dangling `.git` pointer. The parent supplies `BUNDLE` and a new `DEST`:

```bash
git clone --branch work/dior-oracle-consumers "$BUNDLE" "$DEST"
cd "$DEST"
```

Then the runtime owner uses the two complete CPU preparation commands below
with the unchanged frozen inputs. Oracle72 remains final-EMA quantification
plus retained Student only, with no extra Student evaluation or qualitative work.

This appendix-only **Target-supervised** queue contains exactly `LoRA-CGA` and
`LoRA-CGA+VLST`, DIOR `clean/brightness/cloudy/contrast`, seeds `42/43/44`.
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

The following bindings reuse the original paths recorded in
`extensions/protocol.md`, mapped by `HOST_BINDING.md`; they are not newly
generated source data or reports. Run from the owner's deployed checkout of
this delivery. `QUEUE` and `ARTIFACT_ROOT` are **new, separate, non-nested
directories**; neither may already exist. These are the complete arguments:

```bash
CODE="$PWD"
ART=/home/zechuan/iraod_artifacts
ROOT="$ART/comparison/xaf_student_quant_20260908"
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
BASE_PLAN="$ART/comparison/full_test_roi_v3_331d213/completion-plan.json"
CORE_REPORT="$ART/comparison/final_delivery_20260908_4a6bc50/report/report.json"
CORE_PATHS="$ART/comparison/xaf_s424344/paths.py"
EVAL_CODE=/home/zechuan/IRAOD-New-rc331d213
SARCLIP_BASE=/home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors
ADAPTER="$ART/FORMAL_DIOR_LoRA_adapter_0fd539a_20138_20260908T183058Z/lora_dior.pth"
QUEUE="$ROOT/oracle_dior24_manifests_20260909"
ARTIFACT_ROOT="$ROOT/oracle_dior24_20260909"
export PYTHONPATH="$CODE"
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

`dior_oracle24.list` is the checked-in native finite scope: exactly24 training
cells and the same24 final-EMA evaluation cells. Its four-column rows use the
existing default EMA role, not Student training/evaluation. Its ordering matches
the existing generator (cloudy before contrast); no original queue is reordered.
It is a ready consumer **selection**, not an admitted `runtime.json`.

### Owner-only launch handoff

After the CPU command succeeds and the sole compute owner confirms the clean
producer boundary, the existing finite entry point consumes this exact list:

```bash
SCOPE="$CODE/experiments/comparison/dior_oracle24.list"
PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" "$PY" \
  -m experiments.comparison.finite_resumer launch \
  --queue "$QUEUE" --run-dir "$ROOT/finite_runs/oracle-dior24-20260909" \
  --train-list "$SCOPE" --eval-list "$SCOPE" \
  --gpus 0,1,2,3 --handoff-confirmed
```

This is an exact **new DIOR-only scope**, not recovery of Q752. Do not pass
Q752's `--previous-state`: that interface requires an identical finite scope.
Do not launch this producer alongside a release/recovery producer. If the owner
has resumed another producer, retain this queue for its next authorized boundary
instead of starting a competitor. The example retains GPUs0-3; both adapters
are already complete. Parent relays the bundle; only the runtime owner
`98b171c5-ce2e-4ad6-b66a-97ab0e4c76d9` owns remote systems/restoration for this
handoff. No launch, remote check, adapter copy or data write was performed here.

### Actual remaining input boundary

The accepted `d58ee86` code already implements these DIOR consumers; the DIOR
delivery117f607 reused its admission and routing. The RSAR extension below
changes dataset selection, not finite recovery, model code, source configs,
dataset splits or native evaluation. The completed adapter's2971460-byte/final10 record
is owner evidence, not a substitute for `inspect_oracle_adapter` reading it.
The payload and frozen target inputs are not mounted in this local checkout,
so no production `runtime.json` or actual admission pass is claimed.

The owner-side CPU command needs the exact base plan, completed192-cell report,
core resolver and adjacent shared lock wrapper above; the resolver's original
DIOR D/F configs and source teacher; the source42 checkpoint bound by all four
A references; SARCLIP base and final DIOR adapter; unchanged331d evaluator;
and all four original DIOR VAL/TEST directories. No RSAR adapter is an input.
Preparation validates payload/config/source identities, but does not certify
full datasets. The finite input gate additionally requires5863 supported VAL
images per domain and the native TEST list. The accepted target221 boundary now
reports next-stage original inputs present; its readiness receipt records5863
VAL images in each DIOR domain, including brightness, and no missing original
inputs. This supersedes the earlier incomplete-input status, but is not an
Oracle payload-admission or completed-training claim. Actual preparation must
still use those bindings, never synthesize replacements.

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

## RSAR48 actual-input handoff

The owner reported the RSAR fit terminal at2026-09-09T00:47:49Z:
PID3521295 wait exit0, ten CSV epochs and the final file present. This is not
local payload admission or detector compatibility evidence. The same existing
`inspect_oracle_adapter` must read the actual final RSAR payload, including its
six classes, seven corrupted TRAIN domains, final10 selection, rank8/alpha16/
dropout0 and declared SARCLIP initialization. No DIOR adapter is an input.

From the owner's deployed checkout of this delivery on221, the exact CPU
preparation is:

```bash
CODE="$PWD"
ART=/home/zechuan/iraod_artifacts
ROOT="$ART/comparison/xaf_student_quant_20260908"
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
BASE_PLAN="$ART/comparison/full_test_roi_v3_331d213/completion-plan.json"
CORE_REPORT="$ART/comparison/final_delivery_20260908_4a6bc50/report/report.json"
CORE_PATHS="$ART/comparison/xaf_s424344/paths.py"
EVAL_CODE=/home/zechuan/IRAOD-New-rc331d213
SARCLIP_BASE=/home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors
ADAPTER="$ART/FORMAL_RSAR_LoRA_adapter_788a61a_20138_20260908T195742Z/lora_rsar.pth"
QUEUE="$ROOT/oracle_rsar48_manifests_20260909"
ARTIFACT_ROOT="$ROOT/oracle_rsar48_20260909"
export PYTHONPATH="$CODE"
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -m experiments.comparison.oracle_training --dataset RSAR \
  --base-plan "$BASE_PLAN" --core-report "$CORE_REPORT" \
  --core-paths "$CORE_PATHS" --eval-code "$EVAL_CODE" --python "$PY" \
  --adapter "$ADAPTER" --sarclip-base "$SARCLIP_BASE" \
  --out-dir "$QUEUE" --artifact-root "$ARTIFACT_ROOT"
```

Both output roots must be new and non-nested. Only `QUEUE` is created, with
the existing runtime/config/resolver/wrapper outputs and **48 rows in each
generated list**. The native selection is exactly `LoRA-CGA`/`LoRA-CGA+VLST`
x seeds42/43/44 x the original domain order:
`clean/chaff/gaussian_white_noise/point_target/noise_suppression/am_noise_horizontal/smart_suppression/am_noise_vertical`.
Both lists default to EMA role; the Student is retained, not evaluated.

Frozen inputs are the paths above, the core resolver's `rsar_cfg("D")` and
`rsar_cfg("F")` plus their inherited configs and matching six-class source
teacher, and adjacent shared `with_gpu_lock.sh`. All eight A references must
bind the original source42 checkpoint:
`$ART/rsar_source/seed42-orthonet-ddp4-gpu4567-spg16-gbs64/train/epoch_100.pth`.
Retain the actual plan's eight `img_prefix` TEST paths ending in
`test/images`, their corresponding sibling `val/images`, and each native
`ann_file` directory with its original TEST annotations. Do not replace those
with oracle TRAIN crops or rewrite the plan. The finite gate requires8467
supported VAL images/domain; native TEST uses8538 images/domain. The original
evaluation checkout stays at331d2131b84651f0a2930a3d53faeefad8701531.
These are input requirements, not a new claim that target restoration passed.

After actual preparation and the owner's later single-producer boundary,
the exact native launch uses the **generated** lists, with no new static list:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" "$PY" \
  -m experiments.comparison.finite_resumer launch \
  --queue "$QUEUE" --run-dir "$ROOT/finite_runs/oracle-rsar48-20260909" \
  --train-list "$QUEUE/train.list" --eval-list "$QUEUE/eval.list" \
  --gpus 0,1,2,3 --handoff-confirmed
```

Topology remains LoRA-CGA1x32 and LoRA-CGA+VLST2x16 (pairs0,1/29804 and
2,3/29806), global32, SGD LR.02, one epoch/**265 updates**, final filename
label266, iteration-start EMA.998, image-only VAL, unchanged D/F controls.
Only final `iter_266_ema.pth` native quant is queued; `iter_266.pth` is retained.
No extra Student eval, ROI, visualization or source training is authorized here.

**Actual formal RSAR clean42 B_REG failed CUDA OOM on the49140MiB A6000.**
The accepted owner receipt records executor `e5089cc`, 1x32/global32, LR.02
and the full one-epoch/265-update protocol, not a surrogate smoke. This replaces
the historical-memory-only risk assessment. The other23 RSAR B_REG cells are
held, not individually proven failed. This result is not an Oracle run:
Oracle72 actual training remains pending, and neither adapter completion nor
the B_REG result certifies Oracle detector compatibility.

The [target221 boundary](HOST_BINDING.md#verified-target221-execution-boundary)
links the preserved receipts. Native331d's genuine complete clone is preserved.
The first accepted post-repair EVAL success was clean42 B_REG at01:38:33Z,
not brightness; this success is distinct from the earlier environment repair.
The latest full verified recovery aggregate is dated02:33:03Z, not current:
TRAIN38/180 complete,4 actual OOMs and23 unretried policy holds; EMA25/180
complete and3 running,0 failed; Student0 complete/running/failed. The later
F_veto am_noise_horizontal44 TRAIN success at02:44 with both checkpoints
does not silently update that snapshot. The36 preserved TRAIN cohort is not
added again, and retained Student checkpoints do not establish Student EVAL.

Exact source-existing ROI input transfer became terminal at02:55:29Z and
wrapper4b2 is deployed clean, but no ROI inference has started. Restoration
is not Oracle training or a new quantitative result. The01:40 TSD0/36 and
TAM0/12 input receipt remains dated evidence. No aggregate completion, refit, new smoke,
batch/world/budget reduction, extra Student evaluation or qualitative work
is authorized by this documentation.

CPU preparation does not acquire the producer lock or interfere with ongoing
144-phase recovery. The launch body does acquire the shared
`$ART/comparison/xaf_s424344/finite_resumer.producer.lock` and uses canonical
tmux session `xaf-finite-producer`, plus the existing GPU/cell locks.
**Do not run a second producer or interrupt/redeploy the live recovery for
this queue.** Hold the prepared selection until the sole runtime owner
authorizes its clean boundary. Do not attach the144/Q752 `--previous-state`
to this different finite scope. This delivery performs no launch, remote access,
input restoration or actual payload validation; parent relays deployment code and
Herdr `w6:pG`, session `98b171c5-ce2e-4ad6-b66a-97ab0e4c76d9`, controls execution.

## Integration pre-push secrets gate

Before committing a branch update, check whitespace and scan the staged net
delivery from HEAD. The original integration used common accepted base
`d58ee869b52359bd528c2b0767f9fe441f6fa489`; subsequent updates use HEAD to
check only the new delivery. This local credential-pattern scan
prints no matching content; any finding blocks the push for owner resolution.
It is not a claim that pattern matching detects every possible secret.

```bash
git diff --cached HEAD --check
python3 - <<'PY'
import re
import subprocess

diff = subprocess.check_output([
    "git", "diff", "--cached",
    "HEAD", "--",
], text=True)
patterns = (
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
    r"\bsk-[A-Za-z0-9_-]{20,}\b",
    r"""(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*["'][A-Za-z0-9_+/.=-]{16,}["']""",
)
hits = sum(bool(re.search(pattern, diff)) for pattern in patterns)
if hits:
    raise SystemExit(f"BLOCKED: {hits} credential pattern(s); inspect locally without logging values")
print("PASS: staged delivery credential-pattern scan")
PY
```

## Local evidence boundary

`tools.tests.test_oracle_training` uses actual MMCV fixture-config resolution,
synthetic CPU LoRA payload admission, native command construction and mocked
evaluation dispatch. It checks exact grid counts, science/source preservation,
type-only wrapping, no opposite-dataset artifact prerequisite, adapter failures, host aliases
and original-runtime routing through `mixed_queue`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest tools.tests.test_oracle_training
```

The DIOR exact-grid test also compares the checked-in finite selection byte-for-byte
with both generated lists and parses it with the native finite consumer. The
one narrow delivery check is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest \
  tools.tests.test_oracle_training.OracleTrainingTest.test_exact24_executable_configs_preserve_science_and_source
```

The RSAR follow-on's narrow gate is the same module command above: its RSAR
tests check the generated48 selection, actual synthetic payload admission,
source/config preservation,265/266 budget bindings, native rank commands and
original-runtime EMA routing through `mixed_queue`; the DIOR tests protect the
existing default24 behavior. No real detector or remote payload is used.

The local environment has no native `mmrotate` stack;
`tools.tests.test_oracle_isolation` cannot import there. Native detector/model
construction and actual target CPU preparation remain owner-side acceptance,
not a local pass or a claimed experiment result. The code worker owns this
commit/push delivery; the supervisor owns target execution.
