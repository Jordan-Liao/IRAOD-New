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
EXT=/absolute/new/extension-root
"$PY" -m experiments.comparison.extension_manifest prepare \
  --base-plan "$ART/full_test_roi_v3_331d213/completion-plan.json" \
  --core-report "$ART/final_delivery_20260908_4a6bc50/report/report.json" \
  --core-paths "$ART/xaf_s424344/paths.py" \
  --out-dir "$EXT" \
  --eval-code /mnt/SSD2_8TB/zechuan/IRAOD-New-rc331d213 \
  --python "$PY"
```

This binds actual retained Student files to the completed EMA identities.
It creates `student_queue/{runtime.json,paths.py,student.list,run_eval_full.sh}`,
two seeded qualitative plans, `roi_jobs.json`, `embedding_jobs.json`, and
`extension_scope.json`. Nothing is written to the core queue or source trees.

The compute owner first executes a scientific smoke, then releases the finite
Student list using the new runner checkout:

```bash
"$PY" -m experiments.comparison.finite_resumer launch \
  --queue "$EXT/student_queue" --run-dir "$EXT/finite_student_attempt1" \
  --eval-list "$EXT/student_queue/student.list" --gpus 4,5,6,7 \
  --handoff-confirmed
```

The optional fifth list field is `student`; it is evaluation-only. Session and
receipt identities include `-student`, and evidence validation selects
`student_path` / `eval_student_dir`. Original four-field EMA lists and native
331d evidence remain supported. The new wrapper invokes unchanged `test.py` in
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
