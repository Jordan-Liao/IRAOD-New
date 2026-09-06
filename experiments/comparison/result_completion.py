"""Full-TEST RoI coverage with an independent frozen visualization subset."""

import argparse
import csv
import json
from pathlib import Path


SCHEMA = "iraod-aligned-roi-v3-full-test"
EXPECTED_TEST_IMAGES = {"RSAR": 8538, "DIOR": 11738}
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


def test_image_ids(path):
    path = Path(path)
    if path.is_dir():
        return sorted(p.stem for p in path.glob("*.txt") if p.is_file())
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def validate_run(run):
    ids = run["image_ids"]
    ds = run["dataset"]
    if (run.get("scope") != "full_test" or len(ids) != EXPECTED_TEST_IMAGES[ds]
            or len(set(ids)) != len(ids)):
        raise ValueError(f"{ds}: full TEST requires {EXPECTED_TEST_IMAGES[ds]} unique IDs")
    vis = run["visualization_image_ids"]
    if (len(vis) != (32 if ds == "RSAR" else 16) or len(set(vis)) != len(vis)
            or not set(vis).issubset(ids)):
        raise ValueError("Frozen visualization selection must be a 32/16-image TEST subset")
    if ds == "DIOR" and vis != [str(i) for i in range(11726, 11742)]:
        raise ValueError("DIOR visualization selection must be 11726-11741")


def build_plan(bindings):
    """Bindings name final checkpoints explicitly, never mtime/latest selection."""
    runs = []
    image_directories = {}
    for dataset, domains in DOMAINS.items():
        spec = bindings["datasets"][dataset]
        split = Path(spec["test_split"])
        ids = test_image_ids(split)
        if any(not isinstance(i, str) or Path(i).name != i or "." in i for i in ids):
            raise ValueError("image_ids must be filename stems, without extensions")
        selection = read_json(spec["visualization_selection_evidence"])
        selected_ids = (selection["image_ids"] if "image_ids" in selection else
                        [record["image_id"] for record in selection["records"]])
        vis = [Path(image_id).stem for image_id in selected_ids]
        validate_run({"dataset": dataset, "scope": "full_test", "image_ids": ids,
                      "visualization_image_ids": vis})
        if set(spec["domains"]) != set(domains):
            raise ValueError(f"{dataset} must bind exactly {domains}")
        for domain in domains:
            binding = spec["domains"][domain]
            checkpoint_domain = binding["checkpoint_domain"]
            if checkpoint_domain != domain:
                raise ValueError(f"{domain} must use its own adapted checkpoint")
            if Path(binding["ann_file"]).resolve() != split.resolve():
                raise ValueError("RoI ann_file must be the full TEST split, not the visual subset")
            prefix = Path(binding["img_prefix"])
            if str(prefix) not in image_directories:
                suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                stems = [p.stem for p in prefix.rglob("*")
                         if p.is_file() and p.suffix.lower() in suffixes]
                if len(set(stems)) != len(stems):
                    raise ValueError(f"Ambiguous duplicate image stems: {prefix}")
                image_directories[str(prefix)] = set(stems)
            if image_directories[str(prefix)] != set(ids):
                raise ValueError(f"Image directory does not match the complete TEST split: {prefix}")
            for method, role in ROLES:
                checkpoint = (spec["source_checkpoint"] if method == "A" else
                              spec["checkpoints"][checkpoint_domain][method][role])
                if not checkpoint:
                    raise ValueError("All requested final checkpoint bindings are required")
                run_id = f"{dataset}/{domain}/{method}/{role}"
                runs.append({
                    "run_id": run_id, "dataset": dataset, "domain": domain,
                    "method": method, "role": role, "seed": 42,
                    "scope": "full_test",
                    "checkpoint_domain": "source" if method == "A" else checkpoint_domain,
                    "checkpoint": checkpoint, "config": spec["config"],
                    "ann_file": binding["ann_file"], "img_prefix": binding["img_prefix"],
                    "image_ids": ids, "selection_evidence": str(split),
                    "visualization_image_ids": vis,
                    "visualization_selection_evidence": spec["visualization_selection_evidence"],
                    "show_score_thr": 0.3,
                    "out_dir": str(Path(bindings["output_root"]) / SCHEMA / run_id),
                })
    return {"schema": SCHEMA, "scope": "full_test", "runs": runs,
            "roi_image_roles": sum(len(r["image_ids"]) for r in runs),
            "visualization_image_roles": sum(len(r["visualization_image_ids"]) for r in runs)}


def load_run(plan_path, run_id):
    plan = read_json(plan_path)
    if plan["schema"] != SCHEMA:
        raise ValueError("Expected v3 full-test plan; old v2/subset plans cannot be upgraded")
    run = next(run for run in plan["runs"] if run["run_id"] == run_id)
    validate_run(run)
    return run


def load_export(run):
    """Validate complete image metadata eagerly; yield at most one NPZ at a time."""
    validate_run(run)
    root = Path(run["out_dir"])
    index = read_json(root / "index.json")
    if index["schema"] != SCHEMA or index["status"] != "complete" or index["run"] != run:
        raise ValueError(f"Export identity/completion mismatch: {root}")
    if [r["image_id"] for r in index["records"]] != run["image_ids"]:
        raise ValueError("Export does not cover the exact ordered full TEST same-image selection")
    return index, iter_export_records(run, index)


def iter_export_records(run, index, image_ids=None):
    import numpy as np

    root = Path(run["out_dir"])
    for record in index["records"]:
        if image_ids is not None and record["image_id"] not in image_ids:
            continue
        if record["feature_file"] != record["image_id"] + ".npz":
            raise ValueError("NPZ filename must identify its TEST image")
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
        yield record, arrays


def visualize(run):
    from mmrotate.core import imshow_det_rbboxes
    import numpy as np

    index, _ = load_export(run)
    out = Path(run["out_dir"]) / "visualizations"
    out.mkdir(exist_ok=False)
    images = []
    selected = set(run["visualization_image_ids"])
    for record, arrays in iter_export_records(run, index, selected):
        filename = (f"{run['dataset']}_{run['domain']}_{run['method']}_"
                    f"{run['role']}_{record['image_id']}.png")
        imshow_det_rbboxes(
            record["image_path"],
            np.column_stack((arrays["boxes"], arrays["scores"])),
            arrays["labels"], class_names=index["classes"],
            score_thr=run["show_score_thr"], show=False, out_file=str(out / filename))
        images.append({"image_id": record["image_id"], "file": filename})
    by_id = {image["image_id"]: image for image in images}
    write_json(out / "index.json", {
        "schema": SCHEMA, "status": "complete", "run": run,
        "images": [by_id[image_id] for image_id in run["visualization_image_ids"]]})


def collect(plan):
    """Stream full-TEST coverage rows, validating each NPZ before emitting it."""
    if plan["schema"] != SCHEMA:
        raise ValueError("Only v3 full-test plans can produce full TEST completion")
    for run in plan["runs"]:
        validate_run(run)
        root = Path(run["out_dir"])
        exported = None
        if (root / "index.json").exists():
            _, exported = load_export(run)
        rendered = {}
        if (root / "visualizations/index.json").exists():
            vis = read_json(root / "visualizations/index.json")
            if vis["run"] != run or vis["status"] != "complete" or vis["schema"] != SCHEMA:
                raise ValueError("Visualization identity/completion mismatch")
            if [i["image_id"] for i in vis["images"]] != run["visualization_image_ids"]:
                raise ValueError("Incomplete visualization selection")
            for image in vis["images"]:
                path = root / "visualizations" / image["file"]
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"Missing rendered image: {path}")
                rendered[image["image_id"]] = str(path)
        visual_ids = set(run["visualization_image_ids"])
        records = (record for record, _ in exported) if exported is not None else (
            {"image_id": image_id} for image_id in run["image_ids"])
        for record in records:
            image_id = record["image_id"]
            complete = exported is not None
            yield {
                **{k: run[k] for k in ("dataset", "domain", "method", "role", "seed",
                                       "checkpoint_domain", "checkpoint", "config")},
                "image_id": image_id,
                "scope": "full_test",
                "roi_status": "complete" if complete else "not_started",
                "vis_status": ("not_selected" if image_id not in visual_ids else
                               "complete" if image_id in rendered and complete else "not_started"),
                "feature_file": str(root / record["feature_file"]) if complete else "",
                "n_detections": record["n_detections"] if complete else "",
                "visualization_file": rendered.get(image_id, ""),
                "evidence": str(root / "index.json") if complete else "",
            }


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
            raise ValueError("Expected a v3 full-test plan")
        rows = collect(plan)
        # Write only the new versioned manifest, never overwrite legacy evidence.
        out = Path(plan["runs"][0]["out_dir"]).parents[3] / "coverage.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        first = next(rows)
        complete = total = 0
        import itertools
        # Publish only after every required ID/NPZ has been validated.
        temporary = out.with_suffix(".csv.partial")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(first))
            writer.writeheader()
            for row in itertools.chain((first,), rows):
                writer.writerow(row)
                total += 1
                complete += row["roi_status"] == "complete"
        temporary.replace(out)
        print(f"{out}: {complete}/{total} full TEST ROI rows complete")


if __name__ == "__main__":
    main()
