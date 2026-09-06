# Aligned RoI result completion v2

This is code delivery for missing items 3-6, not evidence that remote extraction
or all eight result requirements have finished. GPU execution and checkpoint
provenance remain with the compute supervisor. Existing `.npy`, pickle, DOCX and
legacy manifests are untouched. In particular, old pre-NMS feature arrays are
not labeled instances and must not be supplied to the new t-SNE consumer.

## Exact mapping

The ordinary detector calls `roi_head.simple_test_bboxes`, which feeds the
proposal-ordered RoIs to `fc_cls`, then calls the original bbox head's
`get_bboxes`. The latter retains its activation, class-specific box decoding,
rescale, score threshold, rotated NMS and `max_per_img`.

`sfod.compat` now returns `valid_flat_indices[keep]` when explicitly asked for
indices. Ordinary two-value NMS calls are unchanged. During export only, a
scoped observer requests these indices from that same NMS invocation and returns
the original two values to the bbox head. For `C` foreground classes:

```text
proposal_index = kept_flat_index // C
predicted_label = kept_flat_index % C
feature_row = fc_cls_input[proposal_index]
```

The class-specific box reshape also uses its explicit class dimension so zero
proposals return empty detections instead of failing an ambiguous `(0, -1, 5)`
reshape. Nonempty inference arithmetic is unchanged.

A proposal can legitimately occur in more than one class, with different
class-specific boxes. This is exact NMS indexing, not repeated guessed labels.
The exporter compares the captured detections with the actual detector result,
class by class, before writing. Empty outputs retain `(0, feature_dimension)`
features, `(0, 5)` boxes and empty integer indices. The capture accepts the
existing single-image, single-test-augmentation inference path only.

Each `<image_id>.npz` contains row-aligned `features`, `labels`, `scores`,
`boxes` (original-image `cx,cy,w,h,angle`), `image_ids`, `proposal_indices`,
`flat_indices`, and `detection_indices` (global post-NMS order). `index.json`
records config/checkpoint, code commit, class order, actual test config,
selection, RoI count, feature count and completion. No target annotation or
prototype is consumed by the mapping or plotting code.

## Compute-owner input

Prepare a JSON **bindings** file on the extraction host. Paths are absolute.
The following is the input shape, not a completed manifest:

```json
{
  "output_root": "/absolute/new/result-completion-attempt",
  "datasets": {
    "RSAR": {
      "config": "/absolute/RSAR/source_inference_config.py",
      "source_checkpoint": "/absolute/RSAR/source/epoch_100.pth",
      "selection_evidence": "/absolute/existing/chaff32/index.json",
      "image_ids": ["the exact 32 existing chaff selection filename stems"],
      "domains": {
        "clean": {
          "ann_file": "/absolute/RSAR/test/annfiles",
          "img_prefix": "/absolute/RSAR/test/images",
          "checkpoint_domain": "chaff"
        },
        "chaff": {
          "ann_file": "/absolute/RSAR/test/annfiles",
          "img_prefix": "/absolute/RSAR/corruptions/chaff/test/images",
          "checkpoint_domain": "chaff"
        }
      },
      "checkpoints": {
        "chaff": {
          "B": {
            "ema": "/absolute/chaff/B/final_iter_ema.pth",
            "student": "/absolute/chaff/B/final_iter.pth"
          }
        }
      }
    },
    "DIOR": {
      "config": "/absolute/DIOR/reproducibility/resolved_config.py",
      "source_checkpoint": "/absolute/DIOR/source/epoch_100.pth",
      "selection_evidence": "results/paper_comparison/dior_visualization_selection.json",
      "image_ids": ["11726", "11727", "11728", "11729", "11730", "11731", "11732", "11733", "11734", "11735", "11736", "11737", "11738", "11739", "11740", "11741"],
      "domains": {
        "clean": {
          "ann_file": "/absolute/DIOR/ImageSets/vis16.txt",
          "img_prefix": "/absolute/DIOR/clean/test/images",
          "checkpoint_domain": "brightness"
        },
        "brightness": {
          "ann_file": "/absolute/DIOR/ImageSets/vis16.txt",
          "img_prefix": "/absolute/DIOR/brightness/test/images",
          "checkpoint_domain": "brightness"
        }
      },
      "checkpoints": {
        "brightness": {
          "B": {
            "ema": "/absolute/brightness/B/iter_185_ema.pth",
            "student": "/absolute/brightness/B/iter_185.pth"
          }
        }
      }
    }
  }
}
```

Expand the shape to **all** RSAR domains `clean`, `chaff`,
`gaussian_white_noise`, `point_target`, `noise_suppression`,
`am_noise_horizontal`, `smart_suppression`, `am_noise_vertical`; DIOR domains
`clean`, `brightness`, `cloudy`, `contrast`; and B-F `ema` + `student` for every
corruption. A is always `source`; there is no A/Student or A/EMA duplicate.
Use the actual final iteration, not `latest`, directory mtime, or best-test
selection. Student files from this repository's semi-runner contain the direct
student detector state dict; EMA files contain the standalone EMA detector.
Use the source inference config, not the adaptation/training wrapper config.

For clean images, explicitly bind the adaptation domain whose checkpoint is
being diagnosed (the example uses chaff / brightness). Clean is not a new
adaptation run. Its checkpoint-domain provenance stays in every output and
plot point. All B-F roles in that clean comparison use that same domain.

The local repository does not contain the existing RSAR 32 IDs: the supervisor
must read them from the completed chaff selection, retain their order, and
populate `image_ids`. Do not pick a new first-32 subset. Selection evidence is
read by the planner and must contain either `{"image_ids": [...]}` (the DIOR
selection format) or `{"records": [{"image_id": "name.png"}, ...]}` (the old
extractor index format). Its ordered filename stems must match `image_ids`.
This proves selection identity, not completion of the new extraction.
DIOR's possibly overwritten source must be resolved by its owner before use;
this code does not certify checkpoint history.

## Commands

From the delivered checkout, with the existing detector environment:

```bash
export IRAOD_PYTHON=/home/zechuan/miniforge3/envs/iraod/bin/python
export PYTHONPATH="$PWD"
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export IRAOD_RUNTIME_READY=1
export PLAN=/absolute/completion-plan.json

# CPU: produces 132 planned runs, not completed extraction records.
"$IRAOD_PYTHON" -m experiments.comparison.result_completion plan \
  --bindings /absolute/completion-bindings.json --out "$PLAN"

# COMPUTE SUPERVISOR ONLY: GPU ordinal assigned by that owner.
# Repeat for each plan run_id; examples include the missing Student roles.
experiments/comparison/dior_recovery/run_roi.sh 4 "$PLAN" RSAR/chaff/C/student
experiments/comparison/dior_recovery/run_roi.sh 4 "$PLAN" DIOR/cloudy/F/ema

# CPU: render the same stored predictions, no second detector or NMS invocation.
experiments/comparison/dior_recovery/run_vis.sh "$PLAN" RSAR/chaff/C/student
experiments/comparison/dior_recovery/run_vis.sh "$PLAN" DIOR/cloudy/F/ema

# CPU: consumes completed v2 files; pending rows remain not_started.
"$IRAOD_PYTHON" -m experiments.comparison.result_completion collect --plan "$PLAN"
```

Run IDs are `<RSAR|DIOR>/<domain>/<A-F>/<source|ema|student>`, enumerated in
`plan["runs"]`. Each export refuses an existing run output directory. A failed
attempt is preserved; create a new output root / plan for retries instead of
overwriting or skipping because an old `index.json` exists.

The changed wrapper signatures intentionally replace the old DIOR-only positional
method/corruption arguments and queue-script references. They run the delivered
checkout, not the stale copy of the extractor in a remote queue directory.

The expected coverage is **132 model/domain/role runs and 3,520 image-role
records**: RSAR 88 runs / 2,816 records and DIOR 44 runs / 704 records.
All are seed42; this diagnostic extraction is independent of quantitative
seeds43/44. `collect` writes only
`<output_root>/iraod-aligned-roi-v2/coverage.csv`. It requires valid producer
completion and aligned files before marking ROI completion, plus the renderer's
completion index and actual PNGs before marking visualization completion.
It does not upgrade the repository's stale legacy `not_started` rows from a
plan. Return the actual v2 coverage CSV, indices and files for later integration.

## Joint t-SNE (CPU, after extraction)

The plotting environment needs NumPy, scikit-learn and matplotlib. Use an
existing environment that has them; no GPU is involved. Limit numerical-library
threads for reproducible CPU execution:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
"$IRAOD_PYTHON" -m experiments.comparison.joint_tsne \
  --plan "$PLAN" --dataset RSAR --domain chaff --comparison ema \
  --out-dir /absolute/new/tsne/RSAR/chaff/ema
"$IRAOD_PYTHON" -m experiments.comparison.joint_tsne \
  --plan "$PLAN" --dataset RSAR --domain chaff --comparison student \
  --out-dir /absolute/new/tsne/RSAR/chaff/student
```

Repeat both comparisons for all 12 dataset/domain combinations (24 joint
embeddings). Each comparison includes A/source plus B-F of only the requested
role. All panels select from **one shared embedding** and use identical axis
limits. No per-method projections, GT class coloring or target prototypes.
Colors use the exact NMS-predicted labels; class/color correspondence is in the
separate legend PDF. Panels have no titles; dataset/domain/comparison/method/role
are in filenames. Qualitative images use the same fixed score threshold `0.3`
and the exported detector NMS settings.

Sampling is fixed-seed uniform without replacement, equal count per method:
`min(1000, smallest eligible method pool)`, after score `> 0.3`; no label-based
or visually selected sampling. A single per-channel z-score transform is fitted
to all eligible **A/source-model features on this same domain**, not to target
labels, training prototypes, or each method independently. Zero-variance source
channels use scale 1, recorded explicitly. These statistics are diagnostic
source-model statistics, not clean-training-distribution estimates.

t-SNE uses seed42, random initialization, perplexity30, learning rate200,
Euclidean distance, Barnes-Hut angle0.5, one job. Fewer than 31 total selected
points fails explicitly; it does not silently change the protocol. CLI `--cap`
and `--perplexity` exist for explicit alternate protocols / regression fixtures,
not post-hoc tuning on the final plots.

Outputs: six method PDFs and a class legend, `embedding.npz` with shared
coordinates, normalized sampled features and source mean/std/scale;
`points.csv` with coordinates plus exact file/row/image/proposal/class/box/score/
checkpoint provenance; `protocol.json` with all estimator parameters, library
versions, sampling rules, source counts and per-method raw feature statistics.
Different independently trained representation bases may remain misaligned even
under shared normalization: interpret t-SNE descriptively, not as quantitative
cross-method distance evidence.

## CPU regression

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$IRAOD_PYTHON" -m unittest tools.tests.test_aligned_roi_completion -v
```

This uses the real MMRotate bbox head and MMCV CPU rotated-NMS kernel through the
production capture/consumer, including multiclass suppression, class-specific
decoding, max-count truncation, rescale on/off, score-factor indexing, no
proposals and no surviving detections. It also exercises v2 loading, same-image
rendering, evidence collection, and deterministic joint plotting. It does not
load source/EMA/Student checkpoints or certify their remote provenance.
