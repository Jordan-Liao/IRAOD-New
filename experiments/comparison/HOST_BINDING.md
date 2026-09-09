# Selected target execution binding

This is an operational overlay for exact hostname `73F3-5xA6000-221`
(owner's endpoint `20138`), not a fleet scheduler.

## Interfaces

- `host_binding.map_path(str | Path) -> str`: maps the four selected old
  prefixes to `/home/zechuan`; canonicalizes the equivalent HDD artifact
  prefix to the same home spelling. Never resolves the Python executable.
- `map_data(value)`: copies dict/list/tuple containers, mapping only path
  strings. Keys, source SHAs, tensors, numeric settings and budgets are unchanged.
- `same_path(a, b)`: compares native artifact identities through the mapped
  filesystem aliases on the target, without resolving the interpreter prefix.
- `is_target_host()`: exact hostname comparison, no environment override.
- `approved_gpus()` / `pair_ports()`: target singles `0..4`, pairs
  `0,1:29804` and `2,3:29806`; historical defaults remain singles `4..7`,
  pairs `4,5:29804` and `6,7:29806`. GPU4 is the target's remaining single.
- `finite_resumer.lock_root(queue)` returns the target's one shared directory:
  `/home/zechuan/iraod_artifacts/comparison/xaf_s424344/gpu_locks`.
- `native_command(argv, entry)` inserts the path-only native shim around
  `train.py` / `test.py` on the target. It leaves the bound model code and
  interpreter unchanged.
- Torch's `--local-rank` / `--local_rank` injected before `native` is accepted
  by the wrapper and forwarded unchanged to the underlying native entry.
  The parser does not alter rank/GPU mapping, world size or remaining arguments.

`load_cell` follows mapped `source_queues` and returns mapped runtime views.
`finite_resumer.load_paths` follows copied resolver symlinks with their targets
mapped before reading. Target workers use the current executor, not copied
Bash launchers. Target worker receipts stay in the fresh run directory; they
do not append wrapper statuses to copied Q752.

## Owner launch contract

Set the current integration checkout as `PYTHONPATH` and use exactly
`/home/zechuan/miniforge3/envs/iraod/bin/python`.

Outer LoRA, TAM and TSD ownership uses:

```bash
python -m experiments.comparison.host_binding lock 4 -- COMMAND ARGUMENTS
```

Use the exact venv executable in place of `python`. A two-card command uses
`lock 0,1` or `lock 2,3`. This entry uses the existing nonblocking `take_lock`
and GPU-idle check and exports `IRAOD_GPU_LOCKED=1` only while holding locks.
New queue wrappers expose the same `LOCKDIR` and call this entry. Never execute
the old copied lock script on the target.

**Smoke is different:** invoke `smoke_frozen_training` directly, without an
outer lock. It acquires both cell and GPU locks itself and does not treat an
inherited `IRAOD_GPU_LOCKED=1` as permission to bypass a holder.

The native shim loads literal-path-only temporary config views, including
absolute/relative `_base_` files and frozen F environment assignments.
Snapshots remain untouched. Native model imports resolve in the bound checkout.
The native shim sets the selected interpreter's existing runtime prefix/library
environment before entering frozen scripts, preventing their bootstrap from
re-execing and dropping the active path overlay. It does not add package paths.
Only the operational TAM artifact reader is supplied by the integration
checkout, so copied completed TAM payloads can find their mapped external VGG
encoder without changing tensors or the native TAM implementation.

### Nested RSAR teacher config resolution

The observed RSAR clean42 B_REG two-update NON_RESULT attempt failed while
constructing its teacher, before any optimizer update. The original relative
`ema_config` points to
`configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthonet.py`.
Its existing sibling `_base_` was incorrectly looked up beneath
`/tmp/iraod-bound-config-*/configs/baseline/ema_config/` instead of the frozen
checkout. This is a host-bound config-path resolution defect, not missing
source data/weights, a finite-state failure, or evidence of detector OOM.

`native_config_paths` now makes the mapped source filename absolute in the
native working directory before producing its temporary view. Thus relative
`_base_` references resolve against the real source config's directory, even
when a detector later calls `Config.fromfile(model.ema_config)`. Only requested
config files get temporary views; no checkout copy, replacement config,
fallback search or frozen-file rewrite is performed. MMCV retains its normal
inheritance, original filename and predefined-variable behavior.

The CPU regression loads the actual frozen RSAR teacher and its sibling base
through this nested `Config.fromfile` seam and compares the complete resolved
config with the unpatched loader. The two files are unchanged from0f98.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/iraod-int-venv/bin/python -m unittest \
  tools.tests.test_host_binding.HostBindingTest.test_nested_rsar_ema_config_resolves_original_relative_base
```

The earlier NON_RESULT config failure remains regression evidence only.
The owner subsequently ran the already-planned formal RSAR/clean/42/B_REG
with executor `e5089cc8ea34dfb8e8695798503aadbb66c4b1db`, frozen0f98,
1x32/global32, LR.02 and the full one-epoch/265-update budget. It failed
with actual CUDA OOM, not a new smoke or a completed epoch. The remaining23
RSAR B_REG cells are held, not individually proven failed. The hold does not
establish a failure of the48 F2x16 cells, DIOR or the distinct Oracle methods.
No new smoke, batch/world/budget reduction, numerical workaround or replay
is authorized by this evidence.

### Verified target221 execution boundary

The unchanged owner receipts are versioned in
[`results/extensions/target221_boundary`](../../results/extensions/target221_boundary):
`RSAR_BREG_COMPATIBILITY_RESULT_20260909T014021Z.json`,
`FROZEN_EVAL_CHECKOUT_REPAIR_20260909T012841Z.json` and
`NEXT_STAGE_INPUT_READINESS_20260909T014005Z.json`. The B_REG receipt's
embedded event time is `2026-09-09T01:41:11Z`; its supplied filename is retained.
Remote paths are provenance references, not locally verified artifact copies.

The evaluator at `/home/zechuan/IRAOD-New-rc331d213` is now a genuine complete
clone with HEAD `331d2131b84651f0a2930a3d53faeefad8701531` and tree
`b82e23dbdb7b826a7dbadf36ea64f486fd74f1a8`. Its broken worktree-pointer
checkout was preserved separately. This is environment repair only: these
receipts do not prove any post-repair EVAL succeeded.

The accepted owner boundary preserves36 completed release TRAIN cells
(31 earlier plus5 brightness), with no replay; this count is the parent's
accepted statement, not a result inferred from the three receipts.
Next-stage original inputs are present, including all12 domain VAL counts.
Formal TSD splits remain0/36 and TAM checkpoints0/12; SFYOLO releases per
completed TAM domain, not behind an all12 barrier. Both adapters are complete,
but Oracle72 actual training remains pending. These facts do not establish
aggregate completion or additional authorization. Execution remains exclusively
with Herdr `w6:pG`, session `98b171c5-ce2e-4ad6-b66a-97ab0e4c76d9`.

## Frozen ROI extraction on target221

Preserved BF43/44 ROI240 and port ROI360 use the single-job
`roi_host_binding` operational overlay, with the original native331/b87
exporter and model roots. It maps the old physical prefixes and actual GPU0-4
ownership without changing stored plans, argv, ROI arithmetic or native
identity checks. Current CPU visualization/embedding/collection readers share
that boundary. See [exact owner-only recipes](GENERALIZED_QUALITATIVE.md#selected221-host-preserved-frozen-exports-and-current-cpu-readers).
This is code delivery, not restored-input or actual ROI completion evidence.

## Staging dependencies and parent integration

The compute owner, not this patch, stages:

- Unchanged copied Q752 at
  `/home/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908/release_manifests_752a139`,
  plus its original metadata/config/resolver `source_queues` dependencies.
- Existing base full-TEST plan and TAM plan referenced by the prerequisite
  commands, copied to their prefix-mapped locations rather than regenerated.
- Frozen `0f98` at `/home/zechuan/IRAOD-New-strict-af`, native `331d` at
  `/home/zechuan/IRAOD-New-rc331d213`, and the original a39/36/84 snapshots at
  their artifact-prefix-mapped locations.
- Selected data, original source/checkpoint artifacts, SARCLIP at
  `/home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors`,
  and original `third_party/vgg16_oxford/encoder...pt` at the mapped artifact root.
- This updated integration checkout, including `host_binding.py` and
  `tam_artifacts.py`, accessible to all native child processes.

Native evaluation is wired through the same boundary: host-approved GPUs,
mapped input/environment paths including `RSAR_ROOT`, and the path-only
`native_command` shim for the unchanged331d evaluator. Native home/HDD
checkpoint/config aliases are compared with `same_path`; no sidecar or Q752
is rewritten to make comparisons pass.

Formal AASFOD now routes the mapped queue through the current stage helper,
while each stage executes `cell.training_code/train.py` with that checkout as
cwd and model import root. Stage budgets, LR and 16/32 stage batching are
unchanged. Completed/partial evidence and no-auto-retry behavior are preserved.
Target worker dispatch and AASFOD helper dispatch use absolute current
entrypoints, so an inherited cwd in an old checkout cannot shadow the new
path-binding helpers through Python's `-m` search order.

## DIOR LoRA can precede detector staging

Owner-confirmed current state supersedes the earlier untranslated-path failure:
the deployed0fd539a DIOR smoke completed2/2 updates at18:30:26Z, and the
approved10-epoch fit started18:30:58Z on221 GPU4 (reported pane3513810).
The owner subsequently validated final_epoch10 completion at19:14:24Z:
`lora_dior.pth`2971460 bytes, epoch10 TRAIN loss.0250/accuracy.9922, not TEST.
Do not rerun that completed fit or trigger another compatibility smoke.
DIOR-only24 detector bindings are prepared through `ORACLE_TRAINING.md`;
they do not wait for an RSAR adapter.

The copied `metadata.csv` is not rewritten. On this exact target host,
`train_sarclip_lora_rsar.load_metadata` maps each old absolute `patch_path` in
memory before reading the existing patch file. Class IDs, TRAIN flags,
corruptions, row count, sampling, batch64,10 epochs and all optimizer/LoRA
settings remain unchanged. Runtime bootstrap uses the selected existing
`/home/zechuan/miniforge3/envs/iraod` prefix; the target default SARCLIP import
and tokenizer/model cache directory is the current orchestration checkout,
as selected by the owner. No pypkgs are added to GPU `PYTHONPATH`.

Historical DIOR fit command (already completed; do not rerun):

```bash
ART=/home/zechuan/iraod_artifacts
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH="$CODE"
cd "$CODE"
"$PY" -m experiments.comparison.host_binding lock 4 -- \
  "$PY" tools/train_sarclip_lora_rsar.py \
  --dataset DIOR --seed 42 \
  --metadata "$ART/oracle_dior_patches_36c2053/metadata.csv" \
  --sarclip-dir "$CODE" --sarclip-cache-dir "$CODE" \
  --sarclip-pretrained /home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors \
  --output "$ART/comparison/xaf_student_quant_20260908/DIOR_LoRA_NEW_ATTEMPT" \
  --epochs 10 --batch-size 64 --num-workers 8 \
  --lr 0.0001 --weight-decay 0.0001 --precision fp32 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0
```

That fit required the copied environment, integration code, SARCLIP weights and
DIOR patches, not the detector checkouts/datasets. The outer lock entry is the
single GPU-lock owner for LoRA. Both adapters are now complete; no refit or new
NON_RESULT smoke is authorized here. The code owner does not launch anything.

For detectors, use the copied Q752 with the current executor and explicit
`--gpus 0,1,2,3,4`. Initially passing only `--train-list "$Q/train.list"`
retains180 training +180 automatic EMA evaluations. Adding the existing
`--eval-list "$Q/eval.list"` also admits180 port Student requests; the owner
must respect the known five NaN-checkpoint diagnosis and preserve its artifacts,
not assume filesize proves numerical health or automatically retrain those
cells. No exclusion or replay is implemented by this host-only patch.

The historical54601-54691MiB risk estimate is superseded by the actual formal
RSAR clean42 B_REG OOM on the49140MiB A6000 target. The failed allocation was
512MiB, with194.31MiB free and46.84GiB reserved. Preserve that failure and
the23 held cells; do not reduce batch/world/budget, add activation checkpointing
or substitute a smoke.

## Ready DIOR F2x16 and native evaluation contract

After the owner stages the bound0f98 checkout, source checkpoint, DIOR VAL/TEST,
F config metadata and SARCLIP assets, use the current integration code with
the unchanged copied Q:

```bash
Q="$ART/comparison/xaf_student_quant_20260908/release_manifests_752a139"
cd "$CODE"
PYTHONNOUSERSITE=1 PYTHONPATH="$CODE" "$PY" \
  -m experiments.comparison.smoke_frozen_training \
  --queue "$Q" --dataset DIOR --domain clean --seed 42 --method F_text_only \
  --gpus 0,1 --updates 2 \
  --out-dir "$ART/comparison/xaf_student_quant_20260908/NON_RESULT_F_text_only_221_NEW_ATTEMPT"
```

No outer lock around the self-locking smoke. Pair0,1 uses29804; pair2,3 uses29806.
The original scientific F config,2x16/global32,source and budget remain intact.
Native evaluation uses the bound331d checkout and the actual final checkpoint,
not a smoke checkpoint (smokes do not publish one). The existing finite worker
owns its locks and calls `extension_training evaluate`; for an owner-run
individual evaluation outside that worker, use exactly one outer lock:

```bash
"$PY" -m experiments.comparison.host_binding lock 2 -- "$PY" \
  -m experiments.comparison.extension_training evaluate \
  --queue "$Q" --gpu 2 --dataset DIOR --domain clean --seed 42 \
  --method F_text_only --role ema
```

Current native wrapper instrumentation rejects nonfinite losses/gradients before
updates, nonfinite checkpoint payloads before writing them, and nonfinite loaded
models before evaluation. It does not change finite updates, repair tensors,
retune settings, or replay the diagnosed failed runs. Frozen snapshots remain
unchanged; the owner deploys a new integration snapshot for future detector work.

The boundary wraps the actual standard MMCV optimizer hook and PyTorch
optimizer/save calls used by frozen native entrypoints. It is not a replacement
optimizer or an FP16/gradient-accumulation implementation. Finite CPU regressions
preserve parameters, optimizer state, EMA, RNG, update counts and checkpoint
bytes. On an invalid EMA save, an earlier finite Student file may remain as a
partial failed run, but the bad EMA and final latest marker are not published.
There is no silent tensor repair, batch skip, clipping change or automatic retry.
