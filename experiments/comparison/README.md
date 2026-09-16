# Comparison evidence index

Use the [recovered results](../../results/recovered_20260915/README.md) for the
current complete native TEST view. Historical snapshots and early local planning
files are not live queues.

| Family | Canonical detector TRAIN cells | Required native TEST roles | Required ROI roles |
|---|---:|---:|---:|
| A / fixed source | Not adaptation training | 12 | 12 |
| B-F | 180 | 360 | 360 |
| IRG, LPLD, SFUT | 108 | 216 | 216 |
| B_REG, F_text_only, F_veto_only | 108 | 108 EMA | Not planned |
| AASFOD | 36 | 72 | 72 |
| SFYOLO | 36 | 72 | 72 |
| LoRA-CGA, LoRA-CGA+VLST Oracle | 72 | 72 EMA | Not planned |
| Total | 540 | 912 | 732 |

All listed training budgets, required TEST roles and ROI computation are now
complete. The two fixed source models, 36 TSD jobs, 12 TAM jobs and two LoRA
adapters are prerequisites, not additional adaptation seeds. Six formerly
nonfinite training cells have genuine corrected full-budget final pairs and
their twelve new TEST/ROI results.
[Archival delivery](../../results/recovered_20260915/delivery_manifest.json)
is also complete: 732 unique ROI keys, with training-repair assets counted
separately.

## Method and budget boundaries

- IRG, LPLD, SFUT, AASFOD and SFYOLO are executed Oriented R-CNN OBB ports, not
  upstream HBB/YOLO architectural reproductions. SFUT is a distinct implemented
  method, not an alias for B/ST.
- SFYOLO has a disclosed two-detector-epoch plus TAM budget and is not ranked
  as equal-cost to the common one-epoch methods.
- LoRA-CGA and LoRA-CGA+VLST are supervised Oracle appendix results, not strict
  source-free competitors.
- DRU was explicitly excluded from the accepted matrix because the fixed
  source head lacks source-trained decoder-depth outputs. It was not run.
  The conditional Multi-level Domain Perturbation candidate was not admitted
  to the accepted matrix either.
- Host134's pause and Host68's deferred placement do not represent additional
  unstarted canonical experiments. Freed GPUs are not assigned filler work.

## Early local files

Early copies of `method_matrix.csv`, `ports_executability.md` and
`results/paper_comparison/` may still describe September 6 executability,
unported IRG/LPLD, an unrun Oracle, SFUT as B/ST, or seed42-only statistics.
Those statements are historical and superseded by the keyed coverage above
and the current result tables. They must not be used to exclude completed
methods or launch duplicate work. User-owned uncommitted historical files are
preserved rather than overwritten wholesale.

The September 12 and September 13 published snapshots also retain their
original cutoffs and incomplete statuses intentionally. Later completion is
represented by a new report, not by rewriting those historical metrics.
