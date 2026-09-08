# Full TEST aligned RoI export (v3)

**Deploy the current checkout, not the old queue copy of the extractor.**
Statistics/report expansion is secondary to this extraction delivery. Only the
compute owner launches GPU extraction; these instructions do not imply a new
training run or completed remote artifacts.

## Full versus sampled evidence

| Artifact | Scope |
| --- | --- |
| v3 `image_ids` / per-image NPZ | Every TEST image; every detection after official eval score/NMS/max-per-image processing |
| `visualization_image_ids` / PNG | Frozen RSAR32 / DIOR16 only; separate selection evidence |
| joint t-SNE | Fixed-seed bounded sample drawn from the full aligned exports; not completion evidence for RoIs |
| legacy `.npy`, v1/v2 NPZ/plans | Historical subset/unaligned artifacts; never promoted to `full_test` |

The actual source TEST split counts were read-only confirmed as RSAR **8,538**
annotation files and images, and DIOR **11,738** TEST IDs. The planner reads the
actual split, checks unique IDs/counts, and compares their complete set against
each domain's image directory. No 32/16 limit applies to RoI selection.

The 132 groups are unchanged: RSAR 8 domains x 11 roles and DIOR 4 domains x
11 roles. The derived total is **1,267,816 image-role records**
(`8*11*8538 + 4*11*11738`), not 3,520. Detection-instance counts vary per image
and are recorded, not guessed from the image count. Visualization remains
**3,520 selected image-role records**. A is source only; B-F each have final EMA
and Student. Clean B-F use independently clean-adapted checkpoints.

NPZ export has **no extra 0.3 filter**. It retains all detections after the
original inference score threshold, rotated NMS and `max_per_img`. Score 0.3 is
only for drawing and t-SNE eligibility. All eight arrays have matching row order:
`features`, `labels`, `scores`, `boxes`, `image_ids`, `proposal_indices`,
`flat_indices`, `detection_indices`. Labels are predictions, never inferred from
GT; `boxes` contains the five rotated coordinates.
Empty detections are valid: `(0,D)` features, `(0,5)` boxes, zero rows.

Exact mapping is unchanged: the same actual NMS returns
`valid_flat_indices[keep]`; for C classes, proposal index is `flat // C` and
label is `flat % C`. The exporter observes fc_cls input and compares captured
predictions against the detector result. No nearest-IoU match, label padding,
additional NMS or changed training objective.

## Bindings input

Keep the existing final checkpoint/config bindings. Replace the old subset
selection input with `test_split` and `visualization_selection_evidence`;
remove the old `image_ids` / `selection_evidence` input fields. The planner
derives full `image_ids` itself.

```json
{
  "output_root": "/absolute/new/full-test-roi-attempt",
  "datasets": {
    "RSAR": {
      "config": "/absolute/source_inference_rsar.py",
      "source_checkpoint": "/absolute/rsar_source/train/epoch_100.pth",
      "test_split": "/mnt/shared/zechuan/iraod_data/RSAR/test/annfiles",
      "visualization_selection_evidence": "/absolute/existing/chaff32/index.json",
      "domains": {
        "clean": {
          "ann_file": "/mnt/shared/zechuan/iraod_data/RSAR/test/annfiles",
          "img_prefix": "/mnt/shared/zechuan/iraod_data/RSAR/test/images",
          "checkpoint_domain": "clean"
        },
        "chaff": {
          "ann_file": "/mnt/shared/zechuan/iraod_data/RSAR/test/annfiles",
          "img_prefix": "/mnt/shared/zechuan/iraod_data/RSAR/corruptions/chaff/test/images",
          "checkpoint_domain": "chaff"
        }
      },
      "checkpoints": {
        "clean": {
          "B": {
            "ema": "/absolute/rsar/clean/B/work/iter_266_ema.pth",
            "student": "/absolute/rsar/clean/B/work/iter_266.pth"
          }
        }
      }
    },
    "DIOR": {
      "config": "/absolute/dior_source/reproducibility/resolved_config.py",
      "source_checkpoint": "/absolute/dior_source/train/epoch_100.pth",
      "test_split": "/mnt/shared/zechuan/iraod_data/DIOR/ImageSets/test.txt",
      "visualization_selection_evidence": "/absolute/checkout/results/paper_comparison/dior_visualization_selection.json",
      "domains": {
        "clean": {
          "ann_file": "/mnt/shared/zechuan/iraod_data/DIOR/ImageSets/test.txt",
          "img_prefix": "/absolute/dior_splits/clean/test",
          "checkpoint_domain": "clean"
        }
      },
      "checkpoints": {
        "clean": {
          "B": {
            "ema": "/absolute/dior/clean/B/work/iter_185_ema.pth",
            "student": "/absolute/dior/clean/B/work/iter_185.pth"
          }
        }
      }
    }
  }
}
```

Expand `domains` to all RSAR `clean`, `chaff`, `gaussian_white_noise`,
`point_target`, `noise_suppression`, `am_noise_horizontal`, `smart_suppression`,
`am_noise_vertical`, and DIOR `clean`, `brightness`, `cloudy`, `contrast`.
Expand `checkpoints` to B-F EMA/Student for every domain. Every domain uses
`checkpoint_domain` equal to itself. A always uses its fixed source.
Explicit final checkpoint paths may be planned before training finishes;
the planner does not require all weights to exist. Run each available group
independently instead of waiting for all training.

Every domain's `ann_file` must be the full TEST split, never `vis16.txt`.
RSAR TEST IDs come from sorted annotation stems; DIOR retains `test.txt` order.
The image-directory set must match the whole split with no duplicate stems.
Visualization evidence is the frozen `{"image_ids": [...]}` JSON or the old
chaff `{"records": [{"image_id": "name.png"}, ...]}` index. RSAR must retain its
actual existing 32 IDs; DIOR must be 11726-11741. Both are checked as subsets of
the full split. No new qualitative selection is made.

Output plan/run schema:

```text
schema: iraod-aligned-roi-v3-full-test
scope: full_test
image_ids: every TEST ID
selection_evidence: the actual full TEST split
visualization_image_ids: frozen 32/16 IDs
visualization_selection_evidence: frozen subset evidence
roi_image_roles: derived full total (plan level)
visualization_image_roles: derived selected total (plan level)
```

Old `iraod-aligned-roi-v2` plans are explicitly rejected. Leave old directories
in place, labeled as subset evidence by their old schema. Use a fresh output
root and new plan filename; no binary is rewritten or promoted.

## Commands for the sole compute owner

Use the existing detector environment from the newly deployed checkout:

```bash
export IRAOD_PYTHON=/home/zechuan/miniforge3/envs/iraod/bin/python
export PYTHONPATH="$PWD"
export CONDA_PREFIX=/home/zechuan/miniforge3/envs/iraod
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export IRAOD_RUNTIME_READY=1
export PLAN=/absolute/full-test-roi-plan-v3.json

# CPU: verify actual full split/image directories and materialize all run IDs.
"$IRAOD_PYTHON" -m experiments.comparison.result_completion plan \
  --bindings /absolute/full-test-roi-bindings.json --out "$PLAN"

# GPU OWNER ONLY; examples of individually ready groups, not a training launch.
experiments/comparison/dior_recovery/run_roi.sh 4 "$PLAN" RSAR/chaff/A/source
experiments/comparison/dior_recovery/run_roi.sh 4 "$PLAN" RSAR/chaff/C/student
experiments/comparison/dior_recovery/run_roi.sh 5 "$PLAN" DIOR/cloudy/F/ema

# CPU: draw only the frozen selected images from the full export.
experiments/comparison/dior_recovery/run_vis.sh "$PLAN" RSAR/chaff/C/student

# CPU: stream full coverage; unstarted groups remain not_started.
"$IRAOD_PYTHON" -m experiments.comparison.result_completion collect --plan "$PLAN"

# CPU, only once all six full exports for this comparison exist.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" \
"$IRAOD_PYTHON" -m experiments.comparison.joint_tsne \
  --plan "$PLAN" --dataset RSAR --domain chaff --comparison student \
  --cap 1000 --perplexity 30 --out-dir /absolute/new/tsne-v3/RSAR/chaff/student
```

Run IDs remain `<dataset>/<domain>/<method>/<source|ema|student>`.
Repeat embeddings for each domain and EMA/Student separately: 24 joint fits.
No full feature arrays are loaded to make the sample.

## Streaming and completion

- Extraction reuses the original `image_ids` loop and batch size one. Its
  dataset must contain exactly the full split; its ordered `Subset` now reorders
  **all** TEST images rather than reducing them. It checks final record count
  before publishing `index.json`.
- `load_export` checks complete ordered image metadata eagerly, then yields
  one validated NPZ at a time. Missing any TEST ID, NPZ or aligned row field
  prevents completion. Empty NPZs still produce their image row with zero
  detections. Rendering reads only its selected NPZs.
- `collect` yields per-image rows. `coverage.csv` is written through a
  `.partial` file and published only after validation completes. A
  `vis_status=not_selected` row is not a missing RoI: all non-visual TEST images
  still require `roi_status=complete`. A partial/unstarted run has no completed
  run index and is never counted as full.
- t-SNE streams all NPZs once per method. A fixed-seed bottom-k random-priority
  reservoir holds at most `cap` feature vectors per method, then all six methods
  use the same `min(cap, smallest eligible pool)` size. Streaming batch-Welford
  statistics use **all eligible full-TEST A/source points**, not its reservoir.
  Source zero-variance channels use scale 1, recorded explicitly.
- The joint feature matrix is bounded by `6*cap*D`, plus one image batch and
  bounded reservoirs. Shared source normalization, one t-SNE fit, common panel
  axes, exact point provenance and title-free filenames are retained.
  Sampling/normalization parameters and source population counts are in
  `protocol.json`; this is sampling evidence, not a full-RoI completion marker.

The separate report consumer was only wired to these new scopes and streams;
statistics were not expanded in this production correction. It streams the
full RoI CSV and validates sampled embedding points without keeping all source
NPZ arrays in memory. Its full-TEST RoI and selected visualization totals are
separate.

CPU tests use the real bbox head/rotated-NMS consumer, full-size 8538/11738
split fixtures for the planner CLI, and small split fixtures for streaming
tests. The small fixtures change only expected dataset cardinality in tests,
not the production mapping or NMS implementation.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  IRAOD_DOCX_PYTHON=/path/to/existing/docx/python \
  /path/to/existing/cpu/python -m unittest \
  tools.tests.test_aligned_roi_completion tools.tests.test_comparison_report -v
```
