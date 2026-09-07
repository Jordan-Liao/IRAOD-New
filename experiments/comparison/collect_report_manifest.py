"""Read owner artifact metadata into reusable report inputs; never run models."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import re
import shutil
import sys

from experiments.comparison.final_report import SCHEMA, write_csv
from experiments.comparison.report_inputs import FINAL_ITERATION
from experiments.comparison.result_completion import DOMAINS, read_json, write_json
from experiments.comparison.source_provenance import read_source_producer


ROOT = Path(__file__).resolve().parents[2]


def load_resolver(path):
    """Import the inspected owner path-only helper without writing its pycache."""
    spec = importlib.util.spec_from_file_location("owner_artifact_paths", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def read_identity(path):
    path = Path(path)
    return path.read_text().strip() if path.is_file() else None


def choose_evaluation(paths, owner_format, dataset, domain, seed, method):
    configured = Path(paths.eval_full_dir(dataset, domain, str(seed), method))
    full = configured.with_name(f"eval_full_{domain}")
    method_dir = Path(paths.method_dir(dataset, domain, str(seed), method))
    directories = ([] if configured == full else [configured]) + [
        full.with_name(full.name + "_ids_v1"), full,
        method_dir / f"eval_full_{domain}", full.parent / f"eval_{domain}",
        method_dir / f"eval_{domain}",
    ]
    candidates = [(directory, None) for directory in dict.fromkeys(directories)]
    if method == "A" and domain == "clean":
        source = owner_format.get("A_source_clean", {}).get(dataset)
        if source:
            exact = Path(source["json"])
            candidates.append((exact.parent, exact))
    inspected = []
    for directory, exact in candidates:
        files = ([exact] if exact is not None and exact.is_file() else
                 sorted(directory.glob("eval_*.json")) if exact is None else [])
        inspected.append({"directory": str(directory), "eval_jsons": [str(f) for f in files]})
        if len(files) > 1:
            return directory, None, inspected, "ambiguous_eval_json"
        if len(files) == 1:
            return directory, files[0], inspected, None
    return full, None, inspected, "missing_eval_json"


def collect_manifest(paths, owner_format, roi_plan, out_dir, quant_seeds=(42,),
                     inspect_roi_run_ids=None, paths_file=None, owner_format_file=None,
                     qualitative_evidence=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    rsar_source = read_json(ROOT / "experiments/comparison/checkpoint_manifest.json")["source"]
    dior_source = read_json(ROOT / "results/paper_comparison/dior_checkpoint_manifest.json")
    source_ids = {"RSAR": rsar_source["sha256"], "DIOR": dior_source["source_sha256"]}
    source_checkpoints = {"RSAR": rsar_source["path"], "DIOR": dior_source["source_ckpt"]}
    source_provenance = {
        ds: read_source_producer(
            checkpoint, source_ids[ds],
            Path(checkpoint).parent.parent / "reproducibility/git_commit.txt",
            out / "source_records" / f"{ds}_producer_git_commit.txt")
        for ds, checkpoint in source_checkpoints.items()
    }
    if not set(quant_seeds).issubset({42, 43, 44}):
        raise ValueError("Only declared adaptation seeds 42,43,44 may be collected")
    if paths_file:
        shutil.copyfile(paths_file, out / "paths_snapshot.py")
    if owner_format_file:
        shutil.copyfile(owner_format_file, out / "owner_format_snapshot.json")
    # Freeze selection metadata, not the ongoing export files.
    shutil.copyfile(roi_plan, out / "qualitative-plan.json")
    cells, checkpoints, inventory = [], [], []
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            identities = [("A", 42, "source")] + [
                (method, seed, "ema") for method in "BCDEF" for seed in (42, 43, 44)]
            for method, seed, role in identities:
                base = {"dataset": dataset, "domain": domain, "method": method,
                        "seed": seed, "role": role}
                if method != "A" and seed not in quant_seeds:
                    inventory.append({**base, "status": "not_inspected", "missing": []})
                    continue
                root = Path(paths.root(dataset, domain, str(seed)))
                method_dir = Path(paths.method_dir(dataset, domain, str(seed), method))
                checkpoint = Path(paths.ema_path(dataset, domain, str(seed), method))
                source_record = read_identity(root / "source_ckpt.sha256")
                # The dataset source identity is already owner-confirmed. A present
                # per-run marker must agree, but its absence is not a wrong source.
                matches_source = not source_record or source_record.split()[0] == source_ids[dataset]
                terminals = [method_dir / "terminal_status",
                             method_dir / "ddp2/terminal_status",
                             method_dir / "preflight_ddp2/terminal_status"]
                terminal_records = [
                    {"path": str(p), "record": read_identity(p)} for p in terminals if p.is_file()]
                successful_terminal = False
                for terminal in terminal_records:
                    exits = re.findall(r"\b(?:tmux_wrap_exit|launcher_exit)=(-?\d+)\b",
                                       terminal["record"])
                    successful_terminal |= bool(exits and int(exits[-1]) == 0)
                verified = (checkpoint.is_file() and checkpoint.stat().st_size > 0
                            and matches_source and (method == "A" or successful_terminal))
                checkpoints.append({
                    **base, "path": str(checkpoint), "selection": "source" if method == "A" else "final",
                    "iteration": None if method == "A" else FINAL_ITERATION[dataset],
                    "verified": verified, "source_id": source_ids[dataset],
                    "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
                    "source_identity_file": str(root / "source_ckpt.sha256"),
                    "source_identity_record": source_record,
                    "source_run_marker_present": source_record is not None,
                    "source_identity_basis": "owner-confirmed dataset source manifest",
                    "training_code_record": read_identity(root / "git_commit.txt"),
                    "terminal_evidence": terminal_records,
                    "verification_policy": (
                        "exact checkpoint file; final terminal tmux_wrap_exit/launcher_exit=0; "
                        "owner-confirmed dataset source, rejecting contradictory per-run markers"),
                })
                directory, metric_file, searched, problem = choose_evaluation(
                    paths, owner_format, dataset, domain, seed, method)
                names = ("eval_status", "class_ap.txt", "pred_count.txt", "predictions.pkl",
                         "predictions.pkl.image_ids.json")
                missing = [name for name in names if not (directory / name).is_file()]
                if problem:
                    missing.append(problem)
                if not verified:
                    missing.append("checkpoint_or_source_not_verified")
                metric = None
                if metric_file is not None:
                    payload = read_json(metric_file)
                    metric = float(payload["metric"]["mAP"])
                    cells.append({
                        **base, "eval_dir": str(directory), "eval_json": str(metric_file),
                        "config": payload["config"],
                        "prediction_image_ids": str(directory / "predictions.pkl.image_ids.json"),
                    })
                inventory.append({
                    **base, "status": "metadata_collected" if metric_file else "incomplete",
                    "mAP50": metric, "eval_dir": str(directory),
                    "eval_json": str(metric_file) if metric_file else "",
                    "checkpoint_verified": verified, "missing": missing,
                    "searched": searched,
                    "native_sidecar_present": (directory / "predictions.pkl.image_ids.json").is_file(),
                })
    write_json(out / "cells.json", cells)
    write_json(out / "checkpoints.json", checkpoints)
    write_json(out / "collection_inventory.json", inventory)
    write_csv(out / "collection_inventory.csv", inventory, ("dataset", "domain", "status"))
    manifest = {
        "schema": SCHEMA, "roles": ["ema"], "source_ids": source_ids,
        "source_provenance": source_provenance,
        "inspect_quant_seeds": list(quant_seeds),
        "cells": str((out / "cells.json").resolve()),
        "checkpoints": str((out / "checkpoints.json").resolve()),
        "qualitative_plan": str((out / "qualitative-plan.json").resolve()),
        "embeddings": [],
        "collection": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "quant_seeds_inspected": list(quant_seeds),
            "inventory": str((out / "collection_inventory.json").resolve()),
            "metric_files_found": len(cells),
            "native_sidecars_present": sum(bool(r.get("native_sidecar_present")) for r in inventory),
            "missing_counts": dict(Counter(m for r in inventory for m in r["missing"])),
            "read_scope": "metadata only; no predictions pickle or ROI NPZ loaded by collection",
            "legacy_subset_reference_only": owner_format.get("result_completion", {}),
        },
    }
    if inspect_roi_run_ids is not None:
        manifest["inspect_roi_run_ids"] = list(inspect_roi_run_ids)
    if qualitative_evidence:
        manifest["qualitative_evidence"] = str(Path(qualitative_evidence).resolve())
    write_json(out / "report-manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-module", required=True, help="Inspected owner path-only Python helper")
    parser.add_argument("--owner-format", required=True)
    parser.add_argument("--roi-plan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quant-seed", type=int, action="append",
                        help="Repeat for selected seeds; default inspects 42,43,44")
    parser.add_argument("--roi-run-id", action="append",
                        help="Repeat to freeze the exact ROI groups allowed for this inspection")
    parser.add_argument("--qualitative-evidence",
                        help="Completed streaming evidence directory; reuse without NPZ rescanning")
    args = parser.parse_args()
    manifest = collect_manifest(
        load_resolver(args.paths_module), read_json(args.owner_format),
        args.roi_plan, args.out_dir, tuple(args.quant_seed or (42, 43, 44)),
        args.roi_run_id, args.paths_module, args.owner_format, args.qualitative_evidence)
    print(f"{Path(args.out_dir) / 'report-manifest.json'}: "
          f"{manifest['collection']['metric_files_found']} actual metric JSONs; metadata only")


if __name__ == "__main__":
    main()
