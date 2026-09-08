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

The copied `metadata.csv` is not rewritten. On this exact target host,
`train_sarclip_lora_rsar.load_metadata` maps each old absolute `patch_path` in
memory before reading the existing patch file. Class IDs, TRAIN flags,
corruptions, row count, sampling, batch64,10 epochs and all optimizer/LoRA
settings remain unchanged. Runtime bootstrap uses the selected existing
`/home/zechuan/miniforge3/envs/iraod` prefix; the target default SARCLIP import
and tokenizer/model cache directory is the current orchestration checkout,
as selected by the owner. No pypkgs are added to GPU `PYTHONPATH`.

From the newly deployed integration checkout, the owner can use:

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

This requires the copied environment, integration code, SARCLIP weights and
DIOR patches, not the detector checkouts/datasets. The outer lock entry is the
single GPU-lock owner for LoRA. A requested NON_RESULT smoke uses a separate
fresh output, `--smoke-steps 2` and `--num-workers 0`; it never publishes an
adapter. The code owner does not launch or transfer anything.

For detectors, use the copied Q752 with the current executor and explicit
`--gpus 0,1,2,3,4`. Initially passing only `--train-list "$Q/train.list"`
retains180 training +180 automatic EMA evaluations. Adding the existing
`--eval-list "$Q/eval.list"` also admits180 port Student requests; the owner
must respect the known five NaN-checkpoint diagnosis and preserve its artifacts,
not assume filesize proves numerical health or automatically retrain those
cells. No exclusion or replay is implemented by this host-only patch.

The existing RSAR1x32 logs showed roughly54601-54691MiB peak memory. A6000
49140MiB may expose a real OOM at smoke time; no batch/world-size reduction,
activation-checkpoint change or numerical workaround is introduced here.
