# Final delivery leakage/provenance audit

The frozen strict A-F method and training policy remain unchanged: source-free image-only VAL, no source forward or target-label prototypes, frozen base CLIP/SARCLIP, no LoRA, final EMA rather than TEST selection. TEST labels are used only in offline AP evaluation. The report consumer does not train or select models.

Exact per-cell checkpoint, source ID, native inference ID sidecar, recorded/effective training SHA and evaluation SHA are in raw_results.csv. The RSAR source producer correction b474aaa versus recorded adaptation 0f98 is explicit in producer_metadata_audit.csv; raw sidecars are preserved. Source-available and target-supervised groups are not run.

The earlier executed-config audit is preserved in results/paper_comparison/historical/experiment_docs/leakage_audit.md. This delivery adds provenance evidence, not a new training-code audit.
