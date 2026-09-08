# Authorized extensions to the frozen A-F delivery

Core delivery `32e325442bba54ea11f2179f6c69c6f76fbe995d` is complete and remains
unchanged. Extension results use a new artifact root and separate report tables.
No extension improves or replaces a negative core score by selection or tuning.

## Current integration delivery

`delivery_state.json` tracks the approved parent gate, ready interfaces and
technical blockers. Approval is not pending; DRU remains explicitly excluded.
The existing IRG/LPLD/SFUT producer remains exclusively supervisor-owned.

Student180 is now actually collected: **180/180**, failed0, plus12 reused fixed
source evaluations. Native ordered predictions cover1,728,840 Student TEST
image records and23,844,023 post-NMS detections. Numeric/seed-block statistics,
CSV, LaTeX, plots and DOCX are under `results/extensions/student180/`; large
remote coverage CSV provenance is in its `artifact_manifest.json`.
All five Student methods have lower corrupted-domain mean mPC than A on both
datasets. These negative results are retained, not selected away.

F-deletion deployment was adopted complete at `f_deletion_manifests_daa3992`:
72 bindings,36 per method, frozen0f98,2x16; no formal artifact root had been
created. Committed code84a1aeb is deployed as `integration_code_84a1aeb` under
the same extension root. `tam_manifests_84a1aeb` binds the actual Oxford16-tensor
encoder to12 domain seed42 fits; `sfyolo_manifests_84a1aeb` binds36 two-epoch
detectors to those fits. Neither formal root was created by preparation.
Native CPU SFYOLO/TAM/preparation checks passed; real-weight GPU smokes and
training remain the GPU owner's work.

The bounded LoRA smoke CLI is documented in `tools/ORACLE_PREPARATION.md`.
AASFOD's actual aligned TSD, local/global GRL, four-image geometric FNS and
stage/reset/cadence implementation are now available in `AASFOD.md`; the34-test
native CPU mechanism/queue/cloudy gate passed. TSD and full-detector GPU smokes
remain unrun. DIOR cloudy's author-bound stateful CPU implementation is in
`tools/dataset/DIOR_CLOUDY.md`, with materialization pending. Generalized
new-method report/qualitative consumers remain unfinished. Student quantitative
collection is not completion of these scopes or full qualitative extensions.

CPU collection from the existing compatibility queue (new output directories):

```bash
"$PY" -m experiments.comparison.collect_student_report \
  --queue "$EXT/repo_manifests_37c65b6/student_queue" \
  --core-report "$ART/final_delivery_20260908_4a6bc50/report/report.json" \
  --out-dir "$EXT/student_collection_NEW"
"$PY" -m experiments.comparison.final_report \
  --manifest "$EXT/student_collection_NEW/report-manifest.json" \
  --out-dir "$EXT/student_report_NEW" --metadata-only
```

After copying the compact report, render without reopening remote predictions:
`python -m experiments.comparison.collect_student_report --render-dir REPORT
--docx-python PYTHON`. This adds the Student table and explicitly labels the
DOCX quantitative-only; it does not rewrite the completed core report.

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
export PYTHONNOUSERSITE=1
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

The original Student producer was owned by the compute supervisor and is now
terminal180/180; do not relaunch any Student inference:
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

## Explicitly approved F deletion controls

The two controls below were initially candidates and were held, not run.
The coordinator subsequently explicitly approved exactly these two additions,
each36 training/EMA cells, alongside the separate B_REG36 control. This is
the authority for their active matrix, not a claim that the original plan
contained an unspecified grid. No other F values or deletion variants are added.

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

Approved scope: each variant on all 12 clean/corrupted domains, seeds42/43/44:
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
only explicitly selected IRG/LPLD/SFUT methods, B_REG, F_text_only and F_veto_only. Each selected method binds
36 full training cells and36 final-EMA native evaluations across all12 domains
and seeds42/43/44. Source A is reused, never enqueued for retraining.
Preparation creates no model outputs and is not release/completion evidence.

```bash
ART=/mnt/shared/zechuan/iraod_artifacts/comparison
EXT="$ART/xaf_student_quant_20260908"
PY=/home/zechuan/miniforge3/envs/iraod/bin/python
export PYTHONNOUSERSITE=1
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
Single-GPU IRG/LPLD/SF-UT/B_REG execution keeps global32/LR0.02 and regression
enabled. The F deletions instead retain frozen F's DDP2x16, regression-off
objective, original model environment and iteration-start EMA. Failed processes or missing final Student/EMA files
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

## Confirmed common freeze and scope decisions

`common_base_evidence.json` records literal values read from the actual saved
RSAR/DIOR clean seed42 B training configs. Both explicitly contain confidence0.7,
batch32, LR0.02, one epoch, weight_l0/weight_u1 and regression-off, together
with the unchanged dataset-specific source paths. This is observed evidence,
not an assumed0.7 default; the audit is limited to the two named base records.
All extension configs retain the unchanged strict B augmentation pipelines.

The coordinator permits method-inherent training-only IRG modules, both
pseudo regression branches for IRG/LPLD/SF-UT, and their proper EMA cadences.
Source weights/detector are not replaced. IRG/LPLD perform one actual
epoch-final update after the last optimizer step and before the checkpoint;
SF-UT performs its paper-defined after-step0.9996 updates.

The distinct36-cell SF-UT arm is explicitly in scope. Its comparison with
regression-off B is NOT single-factor because EMA retention/timing also differ.
The coordinator therefore authorized a distinct36-cell B_REG control, not a
duplicate SF-UT alias. The later explicit approval also adds F_text_only and
F_veto_only, each36 cells. AdaBN+Fixed SF-FixMatch is not automatically requested.

## Authorized minimal extra ablation: B_REG

`B_REG` enables `model.cfg.use_bbox_reg=True` for both RPN/ROI pseudo regression,
with every other B setting unchanged. It runs the frozen training checkout
`0f98a48b539f9260055fc305784ffbc924f50451`, uses the original dataset-specific
B configs/source paths/augmentations/hooks, and retains B's .998
iteration-start EMA, not SF-UT's .9996 after-step EMA.

`b_regression.build_b_regression_spec` resolves the original B config with the
same launch options, compares every semantic configuration value, and requires
the exact scientific diff:

```json
[{"path":"model.cfg.use_bbox_reg","before":false,"after":true}]
```

Preparation emits minimal overlays referencing the original B configs and a
per-cell `b_regression_config_diff.json`. Operational experiment/work paths
are new; numerical settings, source, budget and EMA are not changed.
Training records the true frozen0f98 producer and separately the new
orchestration commit. It never modifies the frozen training checkout.

Prepare only `--method B_REG` into NEW metadata/output roots using the same
formal preparation interface above. This binds exactly36 training and36 EMA
evaluation cells (all12 domains, seeds42/43/44), distinct from the already
prepared108 IRG/LPLD/SF-UT cells. The existing B/D/E/F factorial is reused.
No other explicit additional ablation control was found in the original plan;
the later coordinator approval adds only the two named F deletions, not a wider
grid or other F tuning.

## Executable F deletion bindings

`f_deletion.build_f_deletion_spec` reads the original dataset-specific F config
from the unchanged0f98 checkout, reconstructs the exact core DDP model environment,
and verifies original alpha0.5/blend0.7. It emits a fully resolved executable
config, not an inheritance overlay whose parent could reset the environment.
The config restores its effective model environment after loading; a poisoned
outer environment is used in the CPU round-trip check.

The exact allowed differences are:

| Control | Resolved config diff | Effective environment diff |
| --- | --- | --- |
| F_text_only | `model.cfg.vlst_text_visual_alpha:0.5->1.0` | none |
| F_veto_only | none | `CGA_BLEND_DET_WEIGHT:'0.7'->'1.0'` |

All other F loss weights, thresholds, source, augmentations, hooks, and base VLM
bindings remain unchanged. Actual source-CGA scalar construction is covered by
CPU tests with only the VLM constructor mocked; native scientific smoke remains
required before formal execution.

Use the formal preparation CLI with explicit `--method F_text_only --method
F_veto_only` and `--sarclip-base` set to the verified existing base SARCLIP.
Fresh metadata contains72 train/72 EMA bindings and
`f_deletion_config_diff.json`. The six-argument `run_train_2gpu.sh` matches the
existing finite API and supports only4,5/port29804 or6,7/port29806. It executes
frozen0f98 via torch.distributed.launch, samples16/rank, global32, LR0.02,
find_unused_parameters=True and regression-off. It does not overwrite core F,
claim a smoke as formal, or launch/control GPUs during preparation.

Actual B_REG metadata and36 executable-overlay audits are available at
`/mnt/shared/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908/b_reg_manifests_0e5f8f7/`.
Every actual overlay was loaded against its original B config and compared
with identical launch options; the only difference was regression False->True.
The formal output root was not created and no model/producer was launched.

Use `PYTHONNOUSERSITE=1` before starting the metadata/worker Python, matching
the frozen training wrappers. On67, user-site NumPy2 shadows the existing
conda NumPy1.26.4 and breaks OpenCV/MMCV ABI if isolation is omitted. The
actual audit succeeded with the existing conda environment and isolation;
no package was installed, upgraded or downgraded. Generated launchers now set
the flag before Python starts; previously prepared0e5 launchers require the
same flag in the owner/finite-parent environment. Do not modify live workers
or global packages to resolve this import-path issue.

## SF-YOLO independent implementation and auxiliary budget

The actual TAM implementation is `sfod/extensions/tam.py`; its finite trainer
is `experiments/comparison/train_tam.py`. The detector wrapper
`sfod/extensions/sfyolo.py` and `sfyolo_{rsar,dior}.py` configs keep the same
detector/source, hard pseudo OBB losses and common base transforms, adding the
real fitted/frozen TAM to the strong view. No identity/jitter substitute or
YOLO-specific head is used. Pinned formulas/corrections and remaining executable
dependencies are in `remaining_ports.json`. Current approved policy supersedes
the older36-fit/one-detector-epoch proposal.

TAM uses the released code formula (style moments first), unconstrained F2
scale, a frozen VGG encoder, learned decoder/F1/F2, the general aerial objective,
and decoder-first then recomputed F1/F2 Adam updates. Both objectives are
checked for finite values. Preprocessing is consistently centered BGR for
content/style/generated images. Working BGR mean is
`[102.9801,115.9465,122.7717]`. Every frozen Oxford encoder forward adds
`[-.9589,-.8325,-.9083]`; decoder output remains in working space. Inverse
conversion adds the working mean, clips pixels to0..255, reverses channels for
RGB detector input and restores detector normalization/spatial dimensions,
without changing OBB coordinates or padding.

Use Oxford's author-released VGG16 configurationD, conv1_1..conv4_1 only:
16 plain numeric Sequential-key tensors,2,915,648 parameters, three ceil-mode
pools. The remote compute owner supplied
`/mnt/shared/zechuan/iraod_artifacts/third_party/vgg16_oxford/encoder_conv1_1_to_conv4_1.pt`,
SHA256 `158e3923122cc9cccbb26b36d1fa852b91eecd7019a9386c51d415547889d18c`.
This worker has not loaded that remote artifact. Synthetic CPU tests do not
certify its bytes or a real-data smoke. Oxford models are CC BY4.0:
retain Simonyan/Zisserman attribution, author URL, license and tensor-conversion
notice beside the external artifact. No pretrained tensors enter Git or TAM
trained-component payloads.

Exactly12 domain-isolated fits, all seed42, are approved. Each is reused by
detector seeds42/43/44. Each fit has160,000 **outer** iterations: decoder Adam,
then recomputed joint F1/F2 Adam on the same8 content+8 style slots. Totals:
1,920,000 outer loops and3,840,000 optimizer updates. No cross-domain pooling,
seed multiplier, halving or extra iterations. Sorted image IDs, independent
seeded shuffled content/style streams and no entropy reseeding are disclosed
reproducibility corrections, not bitwise author replay.

CPU preparation commands (run from the parent's deployed integration checkout;
`BASE_PLAN`, `CORE_REPORT`, `CORE_PATHS`, `EVAL_CODE`, `PY` are the existing
accepted bindings; `TAM_META`, `TAM_ROOT`, `SF_META`, `SF_ROOT` must be new,
disjoint paths):

```bash
export PYTHONNOUSERSITE=1
VGG=/mnt/shared/zechuan/iraod_artifacts/third_party/vgg16_oxford/encoder_conv1_1_to_conv4_1.pt
"$PY" -m experiments.comparison.prepare_tam \
  --base-plan "$BASE_PLAN" --vgg-weights "$VGG" \
  --out-dir "$TAM_META" --artifact-root "$TAM_ROOT" --python "$PY"
"$PY" -m experiments.comparison.extension_training prepare \
  --base-plan "$BASE_PLAN" --core-report "$CORE_REPORT" --core-paths "$CORE_PATHS" \
  --out-dir "$SF_META" --artifact-root "$SF_ROOT" \
  --eval-code "$EVAL_CODE" --python "$PY" \
  --method SFYOLO --tam-plan "$TAM_META/runtime.json"
```

Preparation writes metadata only, no formal directories, CUDA calls or model
loads. TAM `runtime.json`/`fit_commands.txt` contain12 fixed commands without
`--gpu`; the existing GPU owner appends that argument under its shared lock.
SFYOLO preparation may precede fitting; actual detector loading requires a
completed domain fit with the selected Oxford normalization contract.

Only the parent GPU owner executes the following NON_RESULT smoke, then the
formal commands after validating real weights, finite losses/input gradients,
and frozen-encoder behavior. This worker launches none:

```bash
export PYTHONNOUSERSITE=1
"$PY" -m experiments.comparison.train_tam \
  --base-plan "$BASE_PLAN" \
  --dataset RSAR --domain chaff --seed 42 \
  --vgg-weights "$VGG" \
  --out-dir /absolute/new/tam-smoke --gpu 4 --smoke-steps 1
```

Formal fitting omits `--smoke-steps` and has fixed160k iterations. It reads only
the exact domain's VAL image directory, with no annotations, source images or
GT filtering. New output contains image/training manifests, flushed loss CSV,
terminal state and atomic `tam.pth`. Failed/nonfinite or smoke-only runs cannot
pass the detector's exact dataset/domain/fit-seed42/160k admission. Formal
artifacts from the previous encoder/normalization contract are not reusable.

Detector EMA is parameter-only retention0.999 after optimizer; fixed source
normalization buffers are retained. SSM moves Student halfway toward Teacher
at the next epoch start, skipping epoch0 and retaining optimizer momentum.
The approved two detector epochs activate SSM before epoch2. The image-only
single-group sampler yields `ceil(N/32)` updates/epoch: RSAR265, DIOR184.
Thus there are530/368 optimizer updates; `SemiEpochBasedRunner` saves at
epoch end with `iter+1`, giving **RSAR531, DIOR369** filename labels. Do not
double the old266/185 labels. Both final Student and EMA paths are prepared.
Report as `SF-YOLO (our OBB reimplementation; extended two-epoch + TAM budget)`,
separate from the common one-epoch ranking. Before formal detector execution,
the GPU owner still needs the actual TAM-load/forward/backward smoke and the
two-epoch SSM/EMA/checkpoint interaction confirmed in the installed stack.
No controller or frozen live producer is modified by this preparation.

Targeted CPU checks use the existing unittest runner:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 \
  "$PY" -m unittest tools.tests.test_tam tools.tests.test_tam_training \
  tools.tests.test_extension_training tools.tests.test_sfyolo
```

The local temporary environment passed28 TAM/loop tests and18 preparation/F
consumer tests (`test_extension_training` + `test_f_deletion_execution`), plus
resolved SFYOLO config assertions. The native `test_sfyolo` suite is still
unrun: import stops at missing `mmdet`. Run it in the existing complete IRAOD
environment; it covers pixel inversion/clipping, fitted-TAM admission,
SSM/EMA ordering and the actual sampler/runner checkpoint naming. No replacement
native-op stubs or GPU execution are used to claim those tests passed.

### AASFOD executable boundary

Post-hoc Dropout and a disclosed budget-rescaled EMA are **already approved**.
There is no pending approval gate and no source-retraining requirement.
The executable implementation and pre-result freeze are in `AASFOD.md`.
It uses20 distinct aligned dropout draws, top20% TSD, paired target subsets,
local/global GRL and teacher-before-composition four-image FNS. Stage4 resets
the optimizer and initializes both detectors from stage3 Student. The frozen
alignment/FNS budgets are RSAR159/106 and DIOR110/74 updates, preserving one
common padded epoch. Momentum.99 EMA runs post-optimizer at the disclosed
rescaled interval1. `extension_training prepare --method AASFOD` now emits36
bindings and explicit TSD prerequisite commands. Native CPU geometry, gradient,
dataset, registry and schedule tests passed; no TSD or GPU smoke has run.

## Oracle prerequisite recipe and budgets (before GPU launch)

Absent proven adapters does not stop the oracle objective: the user authorized
constructing the necessary dataset-specific Target-supervised TRAIN adapters.
The concrete valid existing recipe is
`scripts/run_sarclip_rsar_train_corrupt_exp.sh`: pooled seven-corruption RSAR
TRAIN labels, RGB AABB crops with expansion0.4 per side, base SARCLIP ViT-B-32,
10 epochs, batch64, AdamW LR1e-4/weight decay1e-4, rank8/alpha16/dropout0,
FP32, class-balanced replacement sampling and final epoch only. The separate
3-epoch convenience wrapper is not silently substituted; its output-directory
argument has been corrected.

`oracle_recipe.json` freezes the two adapters: RSAR6 classes with
`A SAR image of a {}`, and DIOR20 classes with the existing optical-domain
wording `an aerial image of a {}`. Each uses adapter seed42 and is shared by
both original oracle variants and detector seeds42/43/44. Result uncertainty
is therefore conditional on a fixed adapter, as well as a fixed source.
No adapter-seed sweep or alternate LoRA algorithm is added.

For each dataset, let N be the actual completed crop-manifest row count:
the exact optimizer budget is `10*ceil(N/64)` updates and `10*N` sampled
patch draws. There is no patch cap or dropped tail batch. Counts are obtained
with `tools/build_oracle_patches.py --count-only` before launch; no historical
hand estimate is presented as an observed N. Full construction streams actual
RGB patches and metadata, and training rejects non-TRAIN rows or class/dataset
mismatches.

The existing trainer is parameterized, not replaced:
`tools/train_sarclip_lora_rsar.py --dataset RSAR|DIOR --seed 42`.
LoRA injection must produce actual factor pairs; errors no longer fall back
silently to visual-projection training. The frozen text classifier is detached,
sampling is seeded, nonfinite losses/weights fail, and only the final complete
epoch is atomically saved with full dataset/split/class/prompt/base/optimizer/
count/code provenance. Explicit legacy visual-projection mode remains labeled
as such and is not a LoRA oracle.

The original VAL/TEST corruption folders contained no TRAIN IDs. Dedicated
brightness and contrast TRAIN directories now contain5862 each. The separate
author-bound cloudy generator and CPU command are documented in
`tools/dataset/DIOR_CLOUDY.md`; it preserves DOTA-C equations, lexicographic
texture/image order, MATLAB-compatible resize conventions and cross-image A
state. Materialization remains pending until its terminal summary proves5862
TRAIN outputs. Neither VAL/TEST data nor clean images are relabeled as
corrupted oracle inputs, and cloudy is never substituted with fog.

Both adapters are appendix-only Target-supervised resources. The detector
source stays unchanged, and no strict A-F/baseline config or environment is
permitted to inherit them. The two oracle variants remain LoRA-CGA and
LoRA-CGA+VLST; adapter construction is their prerequisite, not a source-model
replacement.

The actual remote RSAR count audit is now complete:
**1,047,844 valid crops**,149,692 per corruption, zero rejected crops.
The frozen10-epoch/batch64 budget is therefore **163,730 optimizer updates**
and10,478,440 patch draws. This count used the single expansion0.4; it must
not be replaced by an old estimate based on multiple crop expansions.
`oracle_rsar_crop_count.json` preserves the compact class/domain counts and
links the full remote TRAIN-ID/source summary. Count-only mode created no
patch images or training metadataCSV: materialize the crops and reconcile
the final row count before training.

The explicit oracle consumers are in `sfod/extensions/oracle.py`, imported
only by `oracle_cga_*` / `oracle_cga_vlst_*` configs. They preserve the original
D/F detector and inference settings, attach admitted visual LoRA explicitly
after strict base-SARCLIP initialization, and keep separate frozen CGA/VLST
encoder instances. No `SARCLIP_LORA` environment mutation or strict-guard
weakening is used, and no RAW-teacher fallback is allowed for oracle failures.
`oracle_adapters.py` admits only the completed matching dataset/class/TRAIN/
corruption coverage/final-epoch recipe and complete finite LoRA factors.
Training prompts are recorded separately from the preserved D/F inference
prompt ensemble. None of these interfaces claims that an adapter or oracle
detector run has completed.
