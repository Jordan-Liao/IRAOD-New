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
