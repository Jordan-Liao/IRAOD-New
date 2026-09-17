# MDP RSAR/chaff/42 native TEST and ROI

Both authorized joint cases completed. Each covered8538 TEST images and8538
ROI feature files. All17,076 exports have finite FP32 features/boxes/scores and
exact same-forward detection/image-ID alignment. Native sidecar order is the
data-loader order; the ROI index uses the declared sorted order. Their ID sets,
not their incidental ordering, match exactly.

## Measured results

| Method / role | Native mAP50 | mAP50 (%) | Delta vs source (pp) | Relative vs source (%) | Detections |
|---|---:|---:|---:|---:|---:|
| Source A / source | 0.4719308912754059 | 47.193089 | 0 | 0 | Reference only |
| MDP / Student | 0.02022515796124935 | 2.022516 | -45.170573 | -95.714381 | 791231 |
| MDP / EMA | 0.08438440412282944 | 8.438440 | -38.754649 | -82.119330 | 64853 |

The reference is the already-published `RSAR/chaff/42/A/source` row in
`results/recovered_20260915/raw_metrics.csv` at
`b38f419cfa132b8ffd774aebb33f4b8778a283b6`. It has the same8538-image TEST scope,
source identity `e489f015cc2b67926a415436c97532ff814f826c4861b659cbc712fb9413b98c`,
and frozen evaluator revision `331d2131b84651f0a2930a3d53faeefad8701531`.
No baseline was rerun.

Class AP50 values below retain the native text table's three-decimal ratio
precision, displayed as percentages. They are not higher-precision estimates.

| Class | Source A (%) | MDP Student (%) | MDP EMA (%) |
|---|---:|---:|---:|
| ship | 67.9 | 11.7 | 20.7 |
| aircraft | 60.1 | 0.0 | 0.8 |
| car | 69.3 | 0.1 | 10.9 |
| tank | 16.8 | 0.0 | 9.1 |
| bridge | 36.8 | 0.2 | 9.1 |
| harbor | 32.3 | 0.0 | 0.0 |

## Interpretation and limits

1. Execution and artifact validity passed, but both MDP roles regress sharply
   against the matching source reference on this one domain/seed.
2. Student emits92.6717 detections/image versus EMA7.5958, a12.2004-fold count
   difference. Student emits no aircraft or tank detections. This is evidence
   of severe predictive degradation, not an identified training root cause.
3. These are valid low-scoring results. They are retained rather than excluded,
   rerun for a better TEST score, or replaced by an earlier candidate. One case
   does not establish all-domain/multi-seed behavior, and no significance claim
   is made. Any further diagnosis should use training/VAL evidence under a new
   decision, not tune this frozen TEST outcome.

The selected checkpoints are the native final pair from the CPU-checked
265-update diagnostic, not the earlier recovered pair. Numerical model code
remains `d8e2cc8e07043d966b1e8a8ca77921eb46531545`. The exact11 Student training
auxiliaries were admitted explicitly, while all396 detector keys loaded
strictly; EMA loaded396 detector keys with no auxiliaries. Native epoch/iter
metadata is1/266. No claim is made that the historical NaN root cause is fixed.

## Artifacts and execution

`joint_result.json` is the unmodified operator's terminal packet. `student/`
and `ema/` retain the original metric JSON, class table, execution record and
strict-load proof. The large native predictions and ROI files remain on
Host134 under:

`/home/zechuan/iraod_scratch/mdp134_validation_20260916/cases/{student,ema}/`

Student used GPU1,365.68 elapsed seconds /366 allocated GPU seconds.
EMA used GPU2,266.02 /267 seconds. The combined633 GPU seconds remain below
the1000-second validation cap. Two joint forwards produced two native TEST
and two ROI results; no independent duplicate TEST forward was run.

The first attempt failed before locks/model loading because the supervisor
omitted the writer `PYTHONPATH` and used unavailable `os.pidfd_open`.
The corrected task supplied the selected writer namespace and used portable
owned-child waiting. Both failures and correction records remain preserved;
no scientific parameter, checkpoint or frozen runtime was changed.

## Verified incremental ByPy delivery

The complete TEST/ROI case directories and a shared metadata bundle were
delivered to:

`/apps/bypy/IRAOD-New/results/mdp_chaff42_test_roi_20260917/`

| Archive | Bytes | Provider verification |
|---|---:|---|
| `mdp-chaff42-student.tar` | 1241026560 | Size and whole-file MD5 match |
| `mdp-chaff42-ema.tar` | 152524800 | Size and whole-file MD5 match |
| `mdp-chaff42-metadata.tar` | 747520 | Size and whole-file MD5 match |
| Total | 1394298880 | 3 delivered,0 pending,0 active |

The unmodified `bypy_delivery.json` receipt binds each selected role and source
directory to its archive prefix and provider object. Header-only membership
proof covers8552 regular members per role, including all8538 ROI NPZ files,
the six required native files, indices/plans and strict-load/provenance records.
The metadata bundle has17 regular members. No external-link substitutes or
payload reads were used for that membership proof.

This is a new increment, not a rewrite of earlier deliveries. Model checkpoint
weights, raw datasets, repositories, environments and credentials are not part
of this TEST/ROI archive scope.
