# Same-inference prediction image IDs

`test.py --out /new/path/predictions.pkl` now also writes
`/new/path/predictions.pkl.image_ids.json`. It captures each batch's
`img_metas` alongside the actual prediction list, not from a second dataset
build, glob, sorted filename list or post-hoc reconstruction.

Single-process inference appends predictions and their identities in the same
loop. Distributed inference gathers `(prediction, identity)` pairs through the
existing collector, so rank interleaving and sampler-padding removal act on
both together. The saved pickle is still the original list-of-per-class-array
format, numerically unchanged. Empty detections retain their image entry.
Training/evaluation hooks that do not request IDs keep their original return
type and behavior.

Sidecar fields:

```text
schema: iraod-prediction-image-order-v1
origin: inference_batch_img_metas
status: complete                  # prediction saving, NOT entire evaluation/training completion
predictions_file: predictions.pkl
dataset_size, n_images
image_ids: IDs in exactly the saved pickle order
records: [{prediction_index, image_id, ori_filename, filename}, ...]
config, cfg_options, checkpoint, dataset_type
training_code_sha: explicit --training-code-sha value, or null if unknown
evaluation_code_sha: actual evaluation checkout HEAD
```

The sidecar is published only after the pickle has been saved. Existing
prediction, sidecar or interrupted sidecar files are refused. There is no
backfill command: an old pickle without native same-inference evidence remains
unverified and must be re-evaluated to a fresh output location. The report
consumer rejects bare ID lists/text files and checks the native schema,
ordered records, image counts, config/checkpoint and prediction filename.
It records training/evaluation SHAs separately; it never substitutes the eval
SHA for an unknown training SHA.

## Minimal compute-owner wiring (do not change the live training checkout)

Keep the current `0f98...` training checkout/processes/queues untouched. The
owner can prepare a detached evaluation worktree at the delivered commit:

```bash
TRAIN_CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af
EVAL_CODE=/mnt/SSD2_8TB/zechuan/IRAOD-New-eval-image-ids
EVAL_SHA=<delivered-commit>
git -C "$TRAIN_CODE" fetch origin exp/strict-af-integration
git -C "$TRAIN_CODE" worktree add --detach "$EVAL_CODE" "$EVAL_SHA"
```

Use the existing detector Python/environment, original inference config,
exact trained checkpoint, full TEST annotation binding and image directory.
Only the evaluation entry point/PYTHONPATH and fresh output location change:

```bash
# GPU OWNER ONLY. Variables come from that owner's actual ready cell.
# NEW_OUT must be a NEW directory, e.g. eval_full_clean_ids_v1.
mkdir "$NEW_OUT" &&
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="$EVAL_CODE" IRAOD_RUNTIME_READY=1 \
  "$PY" "$EVAL_CODE/test.py" "$CFG" "$CKPT" \
  --eval mAP --out "$NEW_OUT/predictions.pkl" --work-dir "$NEW_OUT" \
  --training-code-sha "$TRAIN_SHA" \
  --cfg-options data.test.ann_file="$FULL_TEST_ANN" \
                data.test.img_prefix="$FULL_TEST_IMAGES" \
  > "$NEW_OUT/eval.log" 2>&1
```

Set `TRAIN_SHA` to the checkpoint's actual producer commit (for the currently
running adaptation code, the known `0f98...` SHA), not `EVAL_SHA`. For unknown
training provenance omit the flag instead of guessing. Source checkpoints may
have a different producer commit from adaptation checkpoints.

Retain the existing wrapper's success-status, class-table and prediction-count
postprocessing for this **new** directory. A separate eval wrapper may point
to this entry point; do not patch/source-overwrite the live training queue.
Old `eval_ok` logic that checks only pickle/count/status is not proof of image
identity and must not skip the required new evaluation. The coordinator/owner
should require the native sidecar in its new evaluation completion check.

In the report cell manifest:

```json
{
  "eval_dir": "/absolute/new/eval_full_clean_ids_v1",
  "eval_json": "/absolute/new/eval_full_clean_ids_v1/eval_TIMESTAMP.json",
  "prediction_image_ids": "/absolute/new/eval_full_clean_ids_v1/predictions.pkl.image_ids.json"
}
```

Keep the other dataset/domain/method/seed/role/checkpoint fields unchanged.
Full-test v3 RoI streaming and its separate visualization subset are unchanged.
The coordinator's separately deployed driver/old-wrapper-root snapshot
`27ce02376f...` remains separate runtime provenance; record its full SHA in the
final deployment snapshot rather than inventing it or replacing training/eval
SHAs. This patch does not modify that live runtime.

CPU regression uses the actual inference APIs, actual pickle writer and
actual CPU Gloo collector (one rank with a padded sampler), including
non-dataset iteration order and empty detections. It does not run a GPU job.
