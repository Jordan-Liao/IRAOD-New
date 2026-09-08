# Authorized extensions to the frozen A-F delivery

Core delivery `32e325442bba54ea11f2179f6c69c6f76fbe995d` is complete and remains
unchanged. Extension results use a new artifact root and separate report tables.
No extension improves or replaces a negative core score by selection or tuning.

## Released nontraining interfaces

- Final Student quantitative: 180 B-F cells, seeds42/43/44, all 12 clean/corrupted
  domains. Use exact RSAR `iter_266.pth` / DIOR `iter_185.pth`, not EMA filenames.
  Reuse the 12 fixed source42 measurements. Student and EMA statistics are separate.
- Seeds43/44 qualitative: 240 new B-F model/domain/role groups, both final EMA and
  Student, 2,305,120 new full-TEST image-role records. Each plan reuses the exact
  12 A/source42 run objects and export directories; do not re-extract A.
- Frozen 32 RSAR / 16 DIOR same-image selection yields 6,400 new visualizations.
  There are 48 new joint embeddings, each comparing A/source42 with one
  adaptation seed and one role. The source points retain seed42 provenance.
  Fixed t-SNE RNG42, cap1000 and perplexity30 remain unchanged.
- Combined unique qualitative scope after completion: 372 groups, 3,572,936
  image-role records, 9,920 visualizations and 72 embeddings. Source references
  appearing in multiple seeded plans are not counted as new exports.

All counts above describe the authorized matrix, **not completed work**.
Detection-row totals must be measured, not extrapolated from seed42.

Prepare on the artifact host, using a new checkout of this implementation and
the existing scientific Python. Preparation is CPU metadata-only:

```bash
ART=/mnt/shared/zechuan/iraod_artifacts/comparison
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
EXT=/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908
META="$EXT/repo_manifests_NEW_COMMIT"
"$PY" -m experiments.comparison.extension_manifest prepare \
  --base-plan "$ART/full_test_roi_v3_331d213/completion-plan.json" \
  --core-report "$ART/final_delivery_20260908_4a6bc50/report/report.json" \
  --core-paths "$ART/xaf_s424344/paths.py" \
  --out-dir "$META" --artifact-root "$EXT" \
  --eval-code /mnt/SSD2_8TB/zechuan/IRAOD-New-rc331d213 \
  --python "$PY"
```

This binds actual retained Student files to the completed EMA identities.
The compute-owned root already exists; only the new metadata directory is
created. Student outputs follow the live owner's mapping: replace the EMA
directory basename `eval_full_<domain>_ids_v1` with
`eval_full_<domain>_student_ids_v1`, retaining DDP2/preflight topology parents.
These are separate directories; no completed EMA output is overwritten.
It creates `student_queue/{runtime.json,paths.py,student.list,run_eval_full.sh}`,
two seeded qualitative plans, `roi_jobs.json`, `embedding_jobs.json`, and
`extension_scope.json`. Nothing is written to the core queue or source trees.

The sole live Student producer is owned by the compute supervisor:
`xaf-student-producer`, with `xafS-DS-domain-seed-method` jobs running the
existing `run_eval_student.sh GPU DS DOMAIN SEED METHOD`. Do not replace or
duplicate it. The command below is ONLY a future coordinated cutover/recovery
entry after the owner authorizes it; it is not a request to launch now:

```bash
"$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$META/student_queue" --run-dir "$EXT/finite_student_NEW_ATTEMPT" \
  --eval-list "$META/student_queue/student.list" --gpus 4,5,6,7 \
  --handoff-confirmed
```

The optional fifth list field is `student`; it is evaluation-only. Session and
receipt identities use `xafS` for Student evaluations, and evidence validation selects
`student_path` / `eval_student_dir`. Original four-field EMA lists and native
331d evidence remain supported. The known legacy Student wrapper/session
can be adopted from the declared compute-owned extension root without renaming
jobs. Native `role=student student=<path>` status is supported directly.
The new wrapper invokes unchanged `test.py` in
the recorded evaluation checkout; it does not pretend its orchestration SHA is
the inference SHA. Shared GPU locks and actual occupancy checks remain active.
Do not stop foreign applications.

The export/visualize argv records in `roi_jobs.json` can use the unchanged331d
extractor with `PYTHONPATH` set to their `export_code`. Only non-A jobs are emitted.
The t-SNE jobs require this newer integration checkout because source42 and
adaptation43/44 are explicitly validated separately. These job manifests are
not themselves a scheduler: the compute owner retains finite execution,
dependency ordering and terminal artifact collection.

## Remaining implementation decisions

Official IRG/LPLD and DRU/AASFOD/SF-YOLO specifications and licensing/source-stage
constraints are being established by separate research owners. No config-only
alias qualifies as a faithful port.

The two oracle variants are Target-supervised SARCLIP-LoRA-CGA and
SARCLIP-LoRA-CGA+VLST, appendix only. Adapter dataset/split/recipe provenance must
be established before release; an RSAR-trained adapter is not a DIOR-target
adapter. Strict methods never inherit these weights or labels.

Additional ablations are not released until concrete existing controls and a
small claim-driven matrix are identified. The completed B/D/E/F component
comparison is reused, not rerun. No open-ended parameter sweep is authorized.

## Identified additional ablation matrix (not yet released for training)

The two deletion controls below exist in the implemented F mechanism. They
test whether its two extra forms of semantic feedback help or hurt; the
negative core F result is a valid outcome, not a reason to search thresholds.

| ID | Single change from F | Mechanism question |
| --- | --- | --- |
| F_text_only | `model.cfg.vlst_text_visual_alpha=1.0` (from0.5) | Does online visual-prototype fusion help beyond frozen text anchors? |
| F_veto_only | `CGA_BLEND_DET_WEIGHT=1.0` (from0.7), retaining `veto_soft` | Does soft disagreement downweighting help beyond the unchanged hard veto? |

`SemanticPrototypeTeacher.get_fused_prototypes` implements
`alpha*text+(1-alpha)*visual`, so alpha1 removes visual fusion without changing
loss weight0.1, temperature0.07 or pseudo-label thresholds. It may still
compute unused visual updates; do not mislabel this as a speed optimization.

In `TestMixins`'s `veto_soft` branch, weight1 makes both soft-downweight
expressions the identity on detector scores, while the hard-veto condition,
drop score0, protection threshold0.9, semantic thresholds0.7/0.1 and VLST remain
unchanged. This is not CGA-off (which is already E).

Proposed scope: each variant on all 12 clean/corrupted domains, seeds42/43/44:
**36 training +36 final-EMA evaluations per variant;72+72 total**. Reuse A and F
references, fixed dataset sources, one image-only VAL epoch, DDP2x16/global32,
LR0.02 and final266/185 selection. Do not launch until the concrete resolved
configuration/environment overrides and a remote scientific smoke are accepted.
The current strict F config assigns its CGA environment values unconditionally;
setting weight1 in an outer shell alone is not a valid implementation.

Compare seed-paired mPC/clean differences to frozen F and report delta_A and
historical rPC as well. Correct the planned two ablation comparisons separately
per dataset/test. No additional values, corruption cherry-picking or
test-selected reruns are part of this matrix.

## Oracle recipe evidence and outstanding provenance

`tools/train_sarclip_lora_rsar.py` currently fixes six RSAR classes and the
template `A SAR image of a {}`. It learns visual LoRA plus logit scale against
frozen text classifiers using class-balanced sampling and AdamW. The concrete
`scripts/run_sarclip_rsar_train_corrupt_exp.sh` trains on labeled RSAR TRAIN
patches from all seven corruptions: AABB expansion0.4,10 epochs,batch64,
LR1e-4,weight decay1e-4,rank8,alpha16,dropout0. Its VAL/TEST calls are separate
evaluation, not adapter training.

The other convenience wrapper specifies3 epochs and incorrectly passes a
`.pth` filename to a directory-valued `--output`; it is not interchangeable
evidence of the executed recipe. The trainer also permits an explicit
visual-projection mode and a legacy LoRA-error fallback. An artifact's actual
`adapter_type`, `classes`, `metadata`, base-model identity, config and log must
be inspected before calling it a LoRA oracle.

No adapter was found in the specifically checked standard work-directory/model
candidates on67. This is not a claim that no adapter exists anywhere; actual
paths/provenance are requested from the compute owner. A valid DIOR adapter has
not been established. Reusing the six-class RSAR adapter cannot be labeled
DIOR-target-supervised. Training a DIOR adapter requires an explicit20-class,
non-TEST labeled-patch recipe and recorded prompt/split/budget; neither labels
nor adapters may enter strict methods.

## LPLD implementation: available for owner scientific smoke, not a completed run

The independent implementation is in `sfod/extensions/lpld_losses.py`,
`lpld.py`, and `proposal_teacher.py`; no unlicensed upstream source is vendored.
Algorithm reference: official commit
`ebdc813805870ec20de91ecfb03705c24e4d51cf`, `student_sfda_rcnn.py:145-221`.

`configs/unbiased_teacher/sfod/extensions/lpld_{rsar,dior}.py` binds the
existing OrthoNet/Oriented R-CNN source architecture, six/twenty classes,
global32 on one GPU, LR0.02 and one target-image-only VAL epoch. The launcher
must supply the actual fixed `load_from`, `model.ema_ckpt`, domain image root,
seed and a NEW work directory. No new source training is required.

Both RPN/ROI classification AND regression pseudo losses are enabled. The
teacher remains frozen throughout the epoch and updates with retention0.75
at epoch end, through a HIGH-priority hook before the NORMAL checkpoint hook.
This preserves the method cadence, not the core B-F iteration-start EMA.
The one-epoch common budget is shorter than the official ten-epoch recipe;
do not claim a published-budget reproduction.

Full common HPL/NMS is retained. Only the auxiliary branch takes the first300
objectness-ordered raw teacher RPN proposals, maps the same indices between
shared-flip weak/strong views, and extracts actual1024-D pre-classifier vectors.
Teacher boxes support class-agnostic Nx5 and class-specific Nx5C decode;
background uses the original proposal. Empty-HPL images have zero LPLD loss.
Other candidates require IoU-to-HPL<=0.4, full-distribution background
probability<=0.99 and foreground-only softmax maximum>=0.9. Detached
`1-cos(teacher,student)` weights multiply foreground-target KL (background
mass1e-10); reduction is weighted sum / kept count /10. `lpld_kept` is a
diagnostic, not a new acceptance threshold.

The owner may use a32-image, one-iteration smoke in a clearly NON_RESULT
directory to exercise the actual production batch and epoch/checkpoint hook:
override `data.train.unlabeled_epoch_size=32`, while retaining global32,
LR0.02 and both fixed source paths. The resulting iter2 checkpoint is NOT a
formal266/185 completion. Check finite losses, real proposal alignment,
kept-count/empty-HPL behavior, frozen teacher during the iteration and correct
final teacher mixing. No Student producer cutover or new formal training is
implied by this smoke interface.

## IRG implementation and pinned canonical choices

`sfod/extensions/irg_losses.py` and `irg.py` now implement and connect the
code-defined IRG mechanism. The source classifier ambiguity is resolved at
the pinned author call site: the Student graph uses the current Student
classifier; the Teacher graph uses the frozen Teacher classifier, retaining
input gradients into the shared graph. The latter is not a no-grad branch.
Graph/MLP parameters are registered before optimizer construction and excluded
from the plain-detector EMA state.

`configs/unbiased_teacher/sfod/extensions/irg_{rsar,dior}.py` reuse the
same source architecture, image-only loader, four pseudo detection losses,
auxiliary indexed proposal interface and HIGH-priority epoch-final hook.
IRG's retention is0.9, so the one-epoch final Teacher is
`0.9*source + 0.1*finalStudent`. These configs are available for scientific
smoke, not a claim that any IRG run has finished.

Author graphs use squared dot-product affinities normalized row-L1, not the
paper softmax adjacency. Contrast uses the detached raw-affinity mask,
two MLPs, an all-column denominator and forced-positive diagonal, not the
paper's anchor-excluding expression. All three KL terms include background;
the teacher and self-distillation targets are detached. Graphs/mining are
per image; batch reduction averages per-image normalized losses, including
zero-loss images.

The deliberate common-base choices must accompany every report:
1024-D OBB/FPN pre-classifier vectors instead of author2048-D vectors;
common HPL>=0.7 instead of IRG author HPL>0.9 (LPLD author uses>0.7);
one epoch/global32/LR0.02 rather than author ten epochs/batch1/LR0.001.
This is an independent code-defined OBB/common-base reimplementation, not
a bit-identical published benchmark reproduction.

The existing weak `RResize` fixes `keep_ratio=True`; strong views share the
same flip and have photometric changes only. Proposal mapping follows the
same component-wise OBB rescale convention as native
`RotatedBBoxHead.get_bboxes`; integer image resizing can introduce small
x/y scale differences through pixel rounding. No arbitrary shear or
perspective transform is supported or implied.

Pinned paper/code attribution, canonical discrepancies, licensing evidence and
implementation status are recorded in `official_ports.json`. The author
annotation-dependent loader and training evaluator are deliberately NOT
ported: their default empty-image filtering/class histograms can consult
target GT before the mapper drops annotations. Only the existing image-only
VAL loader and pseudo labels are used here. No target GT is used for LPLD
online weighting; the paper's GT-IoU plots are motivation/analysis only.

## Simple-SFOD equivalence correction: B is not SF-UT

The bounded official check found a core objective difference, not just a
different backbone or threshold. Paper2407.07586 Sec3.2 Eq3 retains both
RPN and ROI pseudo-box regression as a distinguishing SF-UT contribution.
Pinned source `ee24594869bde0069947540275a28eebb3d4a540`,
`daod/engine/trainers/source_free_adaptive_teacher.py:541-552`, actively
weights both regression losses. Actual B sets `use_bbox_reg=False` and zeros
both branches in `sfod/rotated_unbiased_teacher.py`.

Therefore B's existing numbers remain classification-only ST evidence, never
an SF-UT result. Earlier equivalence wording is superseded in current docs and
the legacy renderer; frozen historical files are intentionally not rewritten.
A distinct SF-UT arm is required, with its exact paper-defined update/augmentation
semantics frozen before release.

The README-selected source configuration is not automatically that paper
method: it disables weak/strong augmentation and the canonical trainer's
teacher-update call is commented out, whereas the paper describes weak/strong
views and per-iteration EMA0.9996. The source variant must be labeled separately.
The umbrella's AdaBN+Fixed SF-FixMatch strategy is also distinct: target-only
BN statistics, then fixed pseudo labels rather than an updating EMA teacher.
It is not collapsed into B or silently added as another experiment matrix.

## Paper-defined SF-UT implementation (distinct from B)

`sfod/extensions/sfut.py` and
`configs/unbiased_teacher/sfod/extensions/sfut_{rsar,dior}.py` implement the
paper-defined mechanism on the existing source architecture and shared
one-epoch/global32/LR0.02 budget. Teacher weak-view predictions supervise the
strong Student through all four RPN/ROI classification/regression losses.
There is no graph, extra head, low-confidence distillation or Student-based
pseudo-label generation. Both models initialize from the fixed source.

`AfterOptimizerTeacherHook` runs at priority45: after the standard optimizer40
and before checkpoint50. It actually updates the teacher after every optimizer
step with retention0.9996, including the final step. It is not the core B
iteration-start EMA0.998, a frozen-teacher source variant, or the alternate
single_wq trainer that labels with its Student.

The common HPL>=0.7 overrides the paper's original strict threshold>0.8 and
is explicitly disclosed. B's regression-off numbers are not reused. This
distinct arm is available for owner scientific smoke; no formal run or
SF-UT result is claimed. AdaBN+Fixed SF-FixMatch remains a separate, unreleased
strategy, not an implicit addition to this implementation.

## Formal finite metadata for smoke-accepted ports

The execution adapter `experiments.comparison.extension_training` supports
only explicitly selected IRG/LPLD/SFUT methods. Each selected method binds
36 full training cells and36 final-EMA native evaluations across all12 domains
and seeds42/43/44. Source A is reused, never enqueued for retraining.
Preparation creates no model outputs and is not release/completion evidence.

```bash
ART=/mnt/shared/zechuan/iraod_artifacts/comparison
EXT="$ART/xaf_student_quant_20260908"
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
META="$EXT/formal_ports_manifests_NEW_COMMIT"
RUNS="$EXT/formal_ports_runs_NEW_COMMIT"
"$PY" -m experiments.comparison.extension_training prepare \
  --base-plan "$ART/full_test_roi_v3_331d213/completion-plan.json" \
  --core-report "$ART/final_delivery_20260908_4a6bc50/report/report.json" \
  --core-paths "$ART/xaf_s424344/paths.py" \
  --out-dir "$META" --artifact-root "$RUNS" \
  --eval-code /mnt/SSD2_8TB/zechuan/IRAOD-New-rc331d213 \
  --python "$PY" --method IRG --method LPLD --method SFUT
```

Both metadata and output roots must be new and separate. The generated
`runtime.json`, `cells.json`, `train.list`, `eval.list`, `paths.py` and wrappers
bind fixed source identities, actual target VAL directories, current training
code/config, the unchanged331d evaluation checkout and exact266/185 finals.
Single-GPU global32/LR0.02 execution removes ambient VLM variables and explicitly
keeps regression enabled. Failed processes or missing final Student/EMA files
receive nonzero terminal status; iter2 smoke files cannot count as formal finals.
The evaluator reuses the common native binding helper and records the actual
new training producer, not the old core producer.

Only after scientific smokes and explicit compute-owner control coordination,
the owner can use these four-field lists with the existing finite event-driven
runner. IRG/LPLD/SFUT are admitted as training/EMA methods; the separate Student
extension remains restricted to the original B-F cells. No automatic model retry,
source retraining, new scheduler, or Student producer cutover is performed by
preparation. The sole live Student submitter and its GPU workers must remain
untouched until the current owner explicitly coordinates formal control.
