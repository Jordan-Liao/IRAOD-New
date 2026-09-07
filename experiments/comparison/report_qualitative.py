"""Streaming evidence indices for full-TEST RoIs and sampled embeddings."""

from pathlib import Path

import numpy as np

from experiments.comparison.report_inputs import read_rows
from experiments.comparison.result_completion import (
    DOMAINS, ROLES, SCHEMA, EXPECTED_TEST_IMAGES, collect, iter_export_records,
    load_export, read_json, validate_run)


def validate_plan(plan):
    if plan["schema"] != SCHEMA:
        raise ValueError("Only v3 full-test qualitative evidence is accepted")
    expected = {f"{ds}/{domain}/{method}/{role}" for ds, domains in DOMAINS.items()
                for domain in domains for method, role in ROLES}
    if (len(plan["runs"]) != len(expected)
            or {r["run_id"] for r in plan["runs"]} != expected):
        raise ValueError("Qualitative plan must cover exactly the 132 full-test groups")
    selections = {}
    for run in plan["runs"]:
        validate_run(run)
        ds = run["dataset"]
        if (run["run_id"] != f"{ds}/{run['domain']}/{run['method']}/{run['role']}"
                or run["seed"] != 42):
            raise ValueError("Qualitative run identity differs from the fixed matrix")
        domain = "source" if run["method"] == "A" else run["domain"]
        if run["checkpoint_domain"] != domain:
            raise ValueError("Qualitative evidence uses a checkpoint from another domain")
        ids = (run["image_ids"], run["visualization_image_ids"])
        if ds in selections and selections[ds] != ids:
            raise ValueError("Qualitative image selection differs across methods/domains")
        selections[ds] = ids


def inspect_embedding(entry, plan):
    root = Path(entry["directory"])
    ds, domain, role = entry["dataset"], entry["domain"], entry["comparison"]
    runs = sorted((r for r in plan["runs"] if r["dataset"] == ds and r["domain"] == domain
                   and (r["role"] == role or r["method"] == "A")), key=lambda r: r["method"])
    files = ["protocol.json", "embedding.npz", "points.csv", "predicted_class_legend.pdf"]
    files += [f"{ds}_{domain}_{role}_{r['method']}_{r['role']}.pdf" for r in runs]
    missing = [name for name in files
               if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        return {"status": "incomplete", "problems": missing, "n_points": None}
    protocol = read_json(root / "protocol.json")
    if (protocol["schema"] != SCHEMA or protocol["runs"] != runs
            or protocol["dataset"] != ds or protocol["domain"] != domain
            or protocol["comparison"] != role):
        raise ValueError("Embedding identity differs from the current qualitative plan")
    if protocol["sampling_seed"] != 42 or protocol["tsne_parameters"]["random_state"] != 42:
        raise ValueError("Embedding must use the fixed sampling/t-SNE seed")
    points = read_rows(root / "points.csv")
    with np.load(root / "embedding.npz", allow_pickle=False) as saved:
        coords, normalized = saved["coordinates"], saved["normalized_features"]
        mean, scale = saved["source_mean"], saved["source_scale"]
    count = protocol["points_per_method"]
    if coords.shape != (6 * count, 2) or len(points) != len(coords):
        raise ValueError("Joint embedding does not contain all six equally sampled panels")
    if normalized.shape != (len(points), len(mean)) or not np.isfinite(coords).all():
        raise ValueError("Invalid joint embedding shape/values")
    for i, run in enumerate(runs):
        export_index, _ = load_export(run)
        wanted = {}
        for j in range(i * count, (i + 1) * count):
            wanted.setdefault(Path(points[j]["feature_file"]).name, []).append(j)
        # Full ROI coverage is checked separately. Audit only the NPZs actually
        # referenced by these sampled points, not the entire dataset again.
        records = iter_export_records(run, export_index, {Path(name).stem for name in wanted})
        matches = ((j, data) for record, data in records
                   for j in wanted.get(record["feature_file"], ()))
        validated = 0
        for j, data in matches:
            validated += 1
            point = points[j]
            feature_file = Path(point["feature_file"])
            if (point["method"] != run["method"] or point["role"] != run["role"]
                    or point["checkpoint"] != run["checkpoint"]
                    or point["dataset"] != ds or point["domain"] != domain
                    or int(point["seed"]) != 42 or point["config"] != run["config"]
                    or point["checkpoint_domain"] != run["checkpoint_domain"]
                    or feature_file.parent != Path(run["out_dir"])
                    or int(point["point_index"]) != j):
                raise ValueError("Embedding point provenance differs from its export")
            row = int(point["feature_row"])
            if row < 0 or row >= len(data["features"]):
                raise ValueError("Embedding feature row is outside the export")
            if (int(point["predicted_label"]) != data["labels"][row]
                    or int(point["proposal_index"]) != data["proposal_indices"][row]
                    or int(point["flat_index"]) != data["flat_indices"][row]
                    or int(point["detection_index"]) != data["detection_indices"][row]
                    or point["image_id"] != data["image_ids"][row]
                    or float(point["score"]) != float(data["scores"][row])
                    or float(point["score"]) <= run["show_score_thr"]
                    or point["predicted_class"] != export_index["classes"][data["labels"][row]]
                    or any(float(point[name]) != float(value) for name, value in
                           zip(("cx", "cy", "w", "h", "angle"), data["boxes"][row]))):
                raise ValueError("Embedding point is not the recorded predicted instance")
            if not np.array_equal([float(point["x"]), float(point["y"])], coords[j]):
                raise ValueError("Point coordinates differ from the shared embedding")
            expected = (data["features"][row].astype(np.float64) - mean) / scale
            if not np.allclose(normalized[j], expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Point feature differs from shared source normalization")
        if validated != count:
            raise ValueError("Embedding points reference missing exported images")
    return {"status": "complete", "problems": [], "n_points": len(points)}


def qualitative_evidence(manifest, roi_sink=None):
    plan_path = manifest.get("qualitative_plan")
    plan = read_json(plan_path) if plan_path else None
    rows = []
    identified = roi_complete = vis_complete = 0
    group_index = []
    if plan is not None:
        validate_plan(plan)
        chosen = manifest.get("inspect_roi_run_ids")
        all_ids = {run["run_id"] for run in plan["runs"]}
        if chosen is not None and (len(set(chosen)) != len(chosen) or set(chosen) - all_ids):
            raise ValueError("Scoped ROI inspection contains duplicate or unknown run IDs")
        selected_ids = all_ids if chosen is None else set(chosen)
        groups = {}
        for run in plan["runs"]:
            selected = run["run_id"] in selected_ids
            group = {
                "run_id": run["run_id"], "scope": "full_test",
                "expected_images": len(run["image_ids"]),
                "status": "pending" if selected else "not_inspected",
                "inspected": selected, "validated_images": 0, "detection_rows": 0,
                "visualizations_complete": 0, "out_dir": run["out_dir"],
            }
            group_index.append(group)
            groups[run["run_id"]] = group
        selected_plan = {**plan, "runs": [r for r in plan["runs"] if r["run_id"] in selected_ids]}
        for row in collect(selected_plan):
            identified += 1
            roi_complete += row["roi_status"] == "complete"
            vis_complete += row["vis_status"] == "complete"
            group = groups[f"{row['dataset']}/{row['domain']}/{row['method']}/{row['role']}"]
            if row["roi_status"] == "complete":
                group["validated_images"] += 1
                group["detection_rows"] += row["n_detections"]
            group["visualizations_complete"] += row["vis_status"] == "complete"
            if roi_sink is None:
                rows.append(row)
            else:
                roi_sink(row)
        for group in group_index:
            if group["inspected"] and group["validated_images"] == group["expected_images"]:
                group["status"] = "complete"
    provided = {}
    for entry in manifest.get("embeddings", []):
        identity = (entry["dataset"], entry["domain"], entry["comparison"])
        if identity in provided:
            raise ValueError("Duplicate embedding entry")
        provided[identity] = entry
    indices = []
    expected = {(ds, d, r) for ds, domains in DOMAINS.items() for d in domains
                for r in ("ema", "student")}
    if set(provided) - expected:
        raise ValueError("Embedding entry outside the 24-comparison matrix")
    for ds, domain, comparison in sorted(expected):
        entry = provided.get((ds, domain, comparison))
        result = {"status": "not_started", "problems": ["missing_embedding_manifest"],
                  "n_points": None}
        if entry and plan is not None:
            result = inspect_embedding(entry, plan)
        elif entry:
            result["problems"] = ["missing_qualitative_plan"]
        indices.append({
            "dataset": ds, "domain": domain, "comparison": comparison,
            "scope": "sampled_joint_embedding_not_full_TEST",
            "directory": entry["directory"] if entry else "", **result})
    roi_expected = (sum(len(r["image_ids"]) for r in plan["runs"]) if plan is not None else
                    sum(len(domains) * len(ROLES) * EXPECTED_TEST_IMAGES[ds]
                        for ds, domains in DOMAINS.items()))
    coverage = {
        "roi_scope": "full_test",
        "roi_expected_image_roles": roi_expected,
        "roi_identified_image_roles": identified,
        "roi_complete": roi_complete,
        "roi_detection_rows": sum(g["detection_rows"] for g in group_index),
        "roi_inspected_groups": sum(g["inspected"] for g in group_index),
        "roi_complete_groups": sum(g["status"] == "complete" for g in group_index),
        "roi_group_index": group_index,
        "vis_expected_image_roles": 3520,
        "vis_complete": vis_complete,
        "embeddings_expected": 24,
        "embeddings_complete": sum(r["status"] == "complete" for r in indices),
        "full_test_roi": "complete" if roi_complete == roi_expected else "incomplete",
    }
    return rows, indices, coverage
