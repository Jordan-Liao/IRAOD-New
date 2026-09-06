"""Seed42 same-image coverage planning, rendering and evidence collection."""

import argparse
import csv
import json
from pathlib import Path


SCHEMA = "iraod-aligned-roi-v2"
DOMAINS = {
    "RSAR": ("clean", "chaff", "gaussian_white_noise", "point_target",
             "noise_suppression", "am_noise_horizontal", "smart_suppression",
             "am_noise_vertical"),
    "DIOR": ("clean", "brightness", "cloudy", "contrast"),
}
ROLES = [("A", "source")] + [
    (method, role) for method in "BCDEF" for role in ("ema", "student")]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def build_plan(bindings):
    """Bindings name final checkpoints explicitly, never mtime/latest selection."""
    runs = []
    for dataset, domains in DOMAINS.items():
        spec = bindings["datasets"][dataset]
        ids = spec["image_ids"]
        expected = 32 if dataset == "RSAR" else 16
        if len(ids) != expected or len(set(ids)) != expected:
            raise ValueError(f"{dataset} requires {expected} unique fixed image IDs")
        if any(not isinstance(i, str) or Path(i).name != i or "." in i for i in ids):
            raise ValueError("image_ids must be filename stems, without extensions")
        if dataset == "DIOR" and ids != [str(i) for i in range(11726, 11742)]:
            raise ValueError("DIOR selection must be 11726-11741 in order")
        if not spec["selection_evidence"]:
            raise ValueError("Existing same-image selection evidence is required")
        selection = read_json(spec["selection_evidence"])
        selected_ids = (selection["image_ids"] if "image_ids" in selection else
                        [record["image_id"] for record in selection["records"]])
        if ids != [Path(image_id).stem for image_id in selected_ids]:
            raise ValueError("image_ids differ from the existing selection evidence")
        if set(spec["domains"]) != set(domains):
            raise ValueError(f"{dataset} must bind exactly {domains}")
        for domain in domains:
            binding = spec["domains"][domain]
            checkpoint_domain = binding["checkpoint_domain"]
            if checkpoint_domain != domain:
                raise ValueError(f"{domain} must use its own adapted checkpoint")
            for method, role in ROLES:
                checkpoint = (spec["source_checkpoint"] if method == "A" else
                              spec["checkpoints"][checkpoint_domain][method][role])
                if not checkpoint:
                    raise ValueError("All requested final checkpoint bindings are required")
                run_id = f"{dataset}/{domain}/{method}/{role}"
                runs.append({
                    "run_id": run_id, "dataset": dataset, "domain": domain,
                    "method": method, "role": role, "seed": 42,
                    "checkpoint_domain": "source" if method == "A" else checkpoint_domain,
                    "checkpoint": checkpoint, "config": spec["config"],
                    "ann_file": binding["ann_file"], "img_prefix": binding["img_prefix"],
                    "image_ids": ids, "selection_evidence": spec["selection_evidence"],
                    "show_score_thr": 0.3,
                    "out_dir": str(Path(bindings["output_root"]) / SCHEMA / run_id),
                })
    return {"schema": SCHEMA, "runs": runs}


def load_run(plan_path, run_id):
    plan = read_json(plan_path)
    if plan["schema"] != SCHEMA:
        raise ValueError("Legacy, unaligned exports are not supported")
    return next(run for run in plan["runs"] if run["run_id"] == run_id)


def load_export(run):
    """Validate the real producer's rows before rendering, plotting or collecting."""
    import numpy as np

    root = Path(run["out_dir"])
    index = read_json(root / "index.json")
    if index["schema"] != SCHEMA or index["status"] != "complete" or index["run"] != run:
        raise ValueError(f"Export identity/completion mismatch: {root}")
    if [r["image_id"] for r in index["records"]] != run["image_ids"]:
        raise ValueError("Export does not cover the exact ordered same-image selection")
    records = []
    for record in index["records"]:
        with np.load(root / record["feature_file"], allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        n = record["n_detections"]
        if (arrays["features"].ndim != 2 or arrays["boxes"].shape != (n, 5)
                or arrays["scores"].shape != (n,) or arrays["image_ids"].shape != (n,)
                or any(len(value) != n for value in arrays.values())):
            raise ValueError("Export row count/shape mismatch")
        for name in ("labels", "proposal_indices", "flat_indices", "detection_indices"):
            if arrays[name].shape != (n,) or arrays[name].dtype.kind not in "iu":
                raise ValueError(f"Invalid integer provenance: {name}")
        classes = len(index["classes"])
        if (not np.array_equal(arrays["image_ids"], np.full(n, record["image_id"]))
                or not np.array_equal(arrays["detection_indices"], np.arange(n))
                or not np.array_equal(arrays["flat_indices"],
                                      arrays["proposal_indices"] * classes + arrays["labels"])
                or np.any(arrays["labels"] < 0) or np.any(arrays["labels"] >= classes)
                or np.any(arrays["proposal_indices"] < 0)
                or np.any(arrays["proposal_indices"] >= record["n_roi"])):
            raise ValueError("Invalid per-point provenance")
        if any(not np.isfinite(arrays[k]).all() for k in ("features", "scores", "boxes")):
            raise ValueError("Nonfinite exported values")
        records.append((record, arrays))
    return index, records


def visualize(run):
    from mmrotate.core import imshow_det_rbboxes
    import numpy as np

    index, records = load_export(run)
    out = Path(run["out_dir"]) / "visualizations"
    out.mkdir(exist_ok=False)
    images = []
    for record, arrays in records:
        filename = (f"{run['dataset']}_{run['domain']}_{run['method']}_"
                    f"{run['role']}_{record['image_id']}.png")
        imshow_det_rbboxes(
            record["image_path"],
            np.column_stack((arrays["boxes"], arrays["scores"])),
            arrays["labels"], class_names=index["classes"],
            score_thr=run["show_score_thr"], show=False, out_file=str(out / filename))
        images.append({"image_id": record["image_id"], "file": filename})
    write_json(out / "index.json", {
        "schema": SCHEMA, "status": "complete", "run": run, "images": images})


def collect(plan):
    """Only actual complete exports can produce completed per-image rows."""
    rows = []
    for run in plan["runs"]:
        root = Path(run["out_dir"])
        exported = {}
        if (root / "index.json").exists():
            _, records = load_export(run)
            exported = {record["image_id"]: record for record, _ in records}
        rendered = {}
        if (root / "visualizations/index.json").exists():
            vis = read_json(root / "visualizations/index.json")
            if vis["run"] != run or vis["status"] != "complete" or vis["schema"] != SCHEMA:
                raise ValueError("Visualization identity/completion mismatch")
            if [i["image_id"] for i in vis["images"]] != run["image_ids"]:
                raise ValueError("Incomplete visualization selection")
            for image in vis["images"]:
                path = root / "visualizations" / image["file"]
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing rendered image: {path}")
                rendered[image["image_id"]] = str(path)
        for image_id in run["image_ids"]:
            record = exported.get(image_id)
            rows.append({
                **{k: run[k] for k in ("dataset", "domain", "method", "role", "seed",
                                       "checkpoint_domain", "checkpoint", "config")},
                "image_id": image_id,
                "roi_status": "complete" if record else "not_started",
                "vis_status": "complete" if image_id in rendered and record else "not_started",
                "feature_file": str(root / record["feature_file"]) if record else "",
                "n_detections": record["n_detections"] if record else "",
                "visualization_file": rendered.get(image_id, ""),
                "evidence": str(root / "index.json") if record else "",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("--bindings", required=True)
    plan_parser.add_argument("--out", required=True)
    for name in ("visualize", "collect"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", required=True)
        if name == "visualize":
            sub.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        plan = build_plan(read_json(args.bindings))
        with Path(args.out).open("x") as stream:
            json.dump(plan, stream, indent=2)
        print(f"Planned {len(plan['runs'])} runs; no completion evidence written")
    elif args.command == "visualize":
        visualize(load_run(args.plan, args.run_id))
    else:
        plan = read_json(args.plan)
        if plan["schema"] != SCHEMA:
            raise ValueError("Expected a v2 plan")
        rows = collect(plan)
        # Write only the new versioned manifest, never overwrite legacy evidence.
        out = Path(plan["runs"][0]["out_dir"]).parents[3] / "coverage.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{out}: {sum(r['roi_status'] == 'complete' for r in rows)}"
              f"/{len(rows)} ROI rows complete")


if __name__ == "__main__":
    main()
