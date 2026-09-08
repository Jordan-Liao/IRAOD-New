"""Collect existing B-F Student artifacts without inference or producer changes."""

import argparse
from pathlib import Path
import shutil

from experiments.comparison.final_report import SCHEMA, write_csv
from experiments.comparison.report_inputs import FINAL_ITERATION, key
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


def collect(queue, core_report, out_dir):
    runtime = read_json(Path(queue) / "runtime.json")
    core = read_json(core_report)
    if core["status"] != "declared_scopes_complete":
        raise ValueError("Student collection requires the accepted complete core report")
    expected = {(ds, domain, method, seed, "student")
                for ds, domains in DOMAINS.items() for domain in domains
                for method in "BCDEF" for seed in (42, 43, 44)}
    bindings = list(runtime["student_cells"].values())
    if len(bindings) != 180 or {key(c) for c in bindings} != expected:
        raise ValueError("Expected exactly180 distinct B-F Student bindings")
    core_rows = {key(row): row for row in core["raw_results"]}
    cells = [{k: row[k] for k in (
        "dataset", "domain", "method", "seed", "role", "eval_dir", "eval_json",
        "prediction_image_ids", "config")}
        for row in core["raw_results"] if row["method"] == "A"]
    checkpoints = [row for row in core["checkpoints"] if row["method"] == "A"]
    inventory = []
    for binding in bindings:
        ds, domain, method, seed, role = key(binding)
        base = dict(dataset=ds, domain=domain, method=method, seed=seed, role=role)
        ema = core_rows[ds, domain, method, seed, "ema"]
        checkpoint = Path(binding["checkpoint"])
        if (ema["status"] != "complete"
                or binding["source_id"] != core["source_ids"][ds]
                or binding["source_id"] != ema["source_id"]
                or checkpoint != Path(ema["checkpoint"]).with_name(
                    f"iter_{FINAL_ITERATION[ds]}.pth")):
            raise ValueError(f"Student binding differs from accepted final training run: {base}")
        verified = checkpoint.is_file() and checkpoint.stat().st_size == binding["checkpoint_bytes"]
        checkpoints.append({
            **base, "path": str(checkpoint), "selection": "final",
            "iteration": FINAL_ITERATION[ds], "verified": verified,
            "source_id": binding["source_id"], "training_code_record": binding["training_code_sha"],
            "verification_policy": "same accepted final EMA training run; exact Student path and bytes",
            "accepted_ema_checkpoint": ema["checkpoint"],
        })
        directory = Path(binding["eval_dir"])
        metrics = sorted(directory.glob("eval_*.json"))
        missing = [name for name in (
            "eval_status", "class_ap.txt", "pred_count.txt", "predictions.pkl",
            "predictions.pkl.image_ids.json") if not (directory / name).is_file()]
        if len(metrics) != 1:
            missing.append("ambiguous_eval_json" if metrics else "missing_eval_json")
        if not verified:
            missing.append("checkpoint_missing_or_size_changed")
        if len(metrics) == 1:
            cells.append({
                **base, "eval_dir": str(directory), "eval_json": str(metrics[0]),
                "config": binding["config"],
                "prediction_image_ids": str(directory / "predictions.pkl.image_ids.json"),
            })
        inventory.append({**base, "eval_dir": str(directory), "missing": missing,
                          "status": "metadata_collected" if not missing else "incomplete"})
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(queue) / "runtime.json", out / "student_runtime.json")
    write_json(out / "cells.json", cells)
    write_json(out / "checkpoints.json", checkpoints)
    write_json(out / "collection_inventory.json", inventory)
    write_csv(out / "collection_inventory.csv", inventory, ("dataset", "domain", "status"))
    manifest = {
        "schema": SCHEMA, "roles": ["student"], "source_ids": core["source_ids"],
        "source_provenance": core["source_provenance"],
        "cells": str(out / "cells.json"), "checkpoints": str(out / "checkpoints.json"),
        "embeddings": [],
        "collection": {
            "queue": str(Path(queue).resolve()), "core_report": str(Path(core_report).resolve()),
            "student_bindings": len(bindings), "reused_source_cells": len(cells) - sum(
                c["role"] == "student" for c in cells),
            "read_scope": "Student metadata; final_report inspects actual predictions and native IDs",
            "qualitative_scope": "not collected here; approved extension qualitative work remains pending",
        },
    }
    write_json(out / "report-manifest.json", manifest)
    return out / "report-manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "core-report", "out-dir"):
        parser.add_argument("--" + name, required=True)
    print(collect(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
