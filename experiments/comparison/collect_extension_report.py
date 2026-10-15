"""Collect explicitly approved extension TEST bindings, CPU only; never launch models."""

import argparse
from pathlib import Path

from experiments.comparison.final_report import SCHEMA
from experiments.comparison.report_inputs import (
    FINAL_ITERATION, SFYOLO_FINAL_ITERATION, key, keyed)
from experiments.comparison.report_statistics import (
    SEEDS, comparison_groups, declared_methods, method_group)
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


def collect(scope, runtimes, core_report, out_dir):
    """A scope declares the full intended matrix even if no bindings exist yet."""
    scope = read_json(scope)
    methods = declared_methods(scope["methods"])
    roles = scope["roles"]
    if (not roles or len(set(roles)) != len(roles)
            or set(roles) - {"ema", "student"} or len(methods) < 2):
        raise ValueError("Declare at least one adaptation method and ema and/or student roles")
    core = read_json(core_report)
    if (core["status"] != "declared_scopes_complete"
            or core["quantitative_complete_cells"] != 192):
        raise ValueError("Reuse the accepted 192-cell core report, not an extension publication")
    expected_source = {(ds, domain, "A", 42, "source")
                       for ds, domains in DOMAINS.items() for domain in domains}
    source_rows = keyed([r for r in core["raw_results"] if r["method"] == "A"])
    source_checkpoints = keyed([r for r in core["checkpoints"] if r["method"] == "A"])
    if (set(source_rows) != expected_source or set(source_checkpoints) != expected_source
            or any(r["status"] != "complete" for r in source_rows.values())):
        raise ValueError("Core must contain exactly twelve accepted fixed source cells")
    cells = [{k: r[k] for k in (
        "dataset", "domain", "method", "seed", "role", "eval_dir", "eval_json",
        "prediction_image_ids", "config")} for r in source_rows.values()]
    checkpoints = list(source_checkpoints.values())
    bindings = {}
    for filename in runtimes:
        runtime = read_json(filename)
        # These are the actual EMA training and explicit native Student emitters.
        # In particular student_checkpoint is NOT a Student evaluation binding.
        for field in ("cells", "student_cells"):
            for original in runtime.get(field, {}).values():
                if original["method"] not in methods or original["role"] not in roles:
                    continue
                binding = dict(original)
                identity = key(binding)
                keyed([binding], methods)
                if identity in bindings:
                    raise ValueError(f"Ambiguous duplicate evaluation binding: {identity}")
                binding["binding_file"] = str(Path(filename).resolve())
                binding["evaluation_code_sha"] = runtime["evaluation_code_sha"]
                bindings[identity] = binding
    inventory = []
    for ds, domains in DOMAINS.items():
        for domain in domains:
            for method in methods[1:]:
                for seed in SEEDS:
                    for role in roles:
                        identity = (ds, domain, method, seed, role)
                        base = dict(zip(("dataset", "domain", "method", "seed", "role"), identity))
                        binding = bindings.get(identity)
                        if binding is None:
                            inventory.append({**base, "status": "incomplete",
                                              "missing": ["missing_explicit_evaluation_binding"]})
                            continue
                        source = source_checkpoints[ds, domain, "A", 42, "source"]
                        if (binding["source_id"] != core["source_ids"][ds]
                                or binding["source_checkpoint"] != source["path"]):
                            raise ValueError(f"Binding differs from the frozen source: {identity}")
                        iteration = (SFYOLO_FINAL_ITERATION if method == "SFYOLO"
                                     else FINAL_ITERATION)[ds]
                        if method == "SFYOLO" and binding["final_checkpoint_iteration"] != iteration:
                            raise ValueError(f"SFYOLO needs the actual two-epoch final label: {identity}")
                        checkpoint = Path(binding["checkpoint"])
                        suffix = "_ema" if role == "ema" else ""
                        if checkpoint.name != f"iter_{iteration}{suffix}.pth":
                            raise ValueError(f"Not the declared final {role} checkpoint: {identity}")
                        verified = checkpoint.is_file() and checkpoint.stat().st_size > 0
                        if verified and "checkpoint_bytes" in binding:
                            verified = checkpoint.stat().st_size == binding["checkpoint_bytes"]
                        directory = Path(binding["eval_dir"])
                        # No resolver fallback, retry choice by mtime, or core eval substitution.
                        if any(directory.resolve() == Path(r["eval_dir"]).resolve()
                               for r in core["raw_results"]):
                            raise ValueError("Extension binding must not reuse a core evaluation directory")
                        metrics = sorted(directory.glob("eval_*.json"))
                        missing = []
                        if len(metrics) != 1:
                            missing.append("ambiguous_eval_json" if metrics else "missing_eval_json")
                        if not verified:
                            missing.append("checkpoint_missing_or_size_changed")
                        config = binding.get("eval_config", binding["config"])
                        training_sha = binding["training_code_sha"]
                        execution_path = directory / "execution.json"
                        if not execution_path.is_file():
                            missing.append("missing_native_execution_binding")
                        else:
                            execution = read_json(execution_path)
                            wanted = {**base, "checkpoint": str(checkpoint), "config": config,
                                      "source_id": binding["source_id"],
                                      "evaluation_code_sha": binding["evaluation_code_sha"]}
                            if any(execution.get(k) != v for k, v in wanted.items()):
                                missing.append("native_execution_binding_mismatch")
                            # extension_training.evaluate records the actual training execution SHA,
                            # which may differ from the earlier prepared runtime SHA.
                            training_sha = execution.get("training_code_sha")
                            if not training_sha:
                                missing.append("missing_training_code_sha")
                        checkpoints.append({
                            **base, "path": str(checkpoint), "selection": "final",
                            "iteration": iteration, "verified": verified,
                            "source_id": binding["source_id"],
                            "source_checkpoint": binding["source_checkpoint"],
                            "training_code_record": training_sha,
                            "binding_file": binding["binding_file"],
                            "verification_policy": "exact native binding; existing final file; native IDs inspected",
                        })
                        cells.append({
                            **base, "eval_dir": str(directory),
                            "eval_json": str(metrics[0]) if len(metrics) == 1 else None,
                            "prediction_image_ids": str(directory / "predictions.pkl.image_ids.json"),
                            "config": config, "training_code_sha": training_sha,
                            "evaluation_code_sha": binding["evaluation_code_sha"],
                            "binding_file": binding["binding_file"],
                            "native_execution": str(execution_path),
                            "comparison_group": method_group(method),
                            "collection_problems": missing,
                        })
                        inventory.append({**base, "eval_dir": str(directory), "missing": missing,
                                          "status": "incomplete" if missing else "bound_not_inspected"})
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "cells.json", cells)
    write_json(out / "checkpoints.json", checkpoints)
    write_json(out / "collection_inventory.json", inventory)
    manifest = {
        "schema": SCHEMA, "methods": methods, "roles": roles, "quantitative_only": True,
        "source_ids": core["source_ids"], "source_provenance": core.get("source_provenance", {}),
        "cells": str(out / "cells.json"), "checkpoints": str(out / "checkpoints.json"),
        "comparison_groups": comparison_groups(methods),
        "collection": {
            "core_report": str(Path(core_report).resolve()),
            "runtime_files": [str(Path(p).resolve()) for p in runtimes],
            "reused_source_cells": 12,
            "export_scope": "Explicit-method quantitative TEST collection only; parent extensions incomplete. "
                            "Full TEST RoI, fixed visualizations and joint t-SNE remain pending.",
            "student_policy": "Only explicit existing Student evaluation bindings; no inferred queue paths.",
        },
    }
    write_json(out / "report-manifest.json", manifest)
    return out / "report-manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, help="JSON with approved methods (A first) and roles")
    parser.add_argument("--runtime", action="append", default=[], help="Existing runtime.json; repeatable")
    parser.add_argument("--core-report", required=True)
    parser.add_argument("--out-dir", required=True, help="New consumer directory, never an existing report")
    args = parser.parse_args()
    print(collect(args.scope, args.runtime, args.core_report, args.out_dir))


if __name__ == "__main__":
    main()
