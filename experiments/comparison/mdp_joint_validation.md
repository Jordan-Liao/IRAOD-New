# Declared MDP native TEST and ROI

This extension admits the new `RSAR/chaff/42/MDP` Student and EMA roles to the
existing single-forward joint exporter. It does not modify detector inference,
TEST labels/IDs, feature extraction, model weights, the frozen331d evaluator,
the d668 writer, or historical B-F/PORT provenance.

## Checkpoint loading

MDP Student checkpoints contain the396 detector state entries plus exactly11
training-only `mdp.*` auxiliaries. Their names, shapes, dtypes and finite values
are validated against the detector's C2/C4/classifier dimensions. Only those
auxiliaries are removed from the in-memory load view; all detector keys and
shapes must match and the native loader is called with `strict=True`. EMA
checkpoints require the detector-only key set. The original checkpoint payload
and files are not rewritten. The existing host finite-load boundary remains
active.

`mdp_checkpoint_load.json` records the actual strict load. MDP ROI publication
requires this proof as well as the existing genuine native execution and
same-forward prediction/feature checks. A generic non-strict warning is not an
admission result.

## New provenance, not fabricated history

The metadata preparer consumes the staged runtime rows and transfer/evidence
receipt. It requires the original finite full-budget producer record, exact
case/role/training revision, matching source/destination checkpoint identities,
the declared evaluator config and frozen32 selection. The selected rows carry
their real producer/transfer evidence rather than invented historical B-F
reports or previous native outputs.

```sh
PYTHONPATH=PATCHED_ROI_CODE NATIVE_PY -m experiments.comparison.mdp_joint_inputs \
  --inputs STAGED_RUNTIME_INPUTS --evidence STAGED_INPUT_EVIDENCE \
  --roi-code PATCHED_ROI_CODE --output FRESH_METADATA_DIR \
  --case-root FRESH_CASES_PARENT
```

This writes two `kind=mdp` specs, a new declared plan, and a task-owned resolver.
The resolver delegates unchanged path/environment/finite-check behavior to the
identity-checked d668 resolver, but permits only Host134 GPUs1 and2. It wraps
native commands through itself rather than falling back to the global4-7
resolver. The base resolver is never edited.

Use the generated resolver's existing `lock` command around each joint entry.
The operator assigns GPU1 to Student and GPU2 to EMA. The supervisor owns a
separate control/log directory; `roi_joint_native.py --out` must remain absent
until the entry creates it. Do not precreate that case output or run a separate
native TEST pass: each joint forward produces one TEST and one ROI result for
all8538 images.

The current authorization is only these two roles on Host134, with per-case
500-second and combined1000 GPU-second operational caps based on observed
same-device runtimes. The native MDP candidate remains distinct from a claimed
repair of the historical NaN. TEST scores are not used to choose between
earlier candidate checkpoints.
