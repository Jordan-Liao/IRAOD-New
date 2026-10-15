"""Streaming evidence indices for full-TEST RoIs and sampled embeddings."""

from pathlib import Path

import numpy as np

from experiments.comparison import host_binding as host
from experiments.comparison.report_inputs import read_rows
from experiments.comparison.result_completion import (
    DOMAINS, ROLES, SCHEMA, EXPECTED_TEST_IMAGES, collect, iter_export_records,
    load_export, read_json, validate_run, plan_methods, comparison_runs)


def validate_plan(plan):
    if plan["schema"] != SCHEMA:
        raise ValueError("Only v3 full-test qualitative evidence is accepted")
    methods = plan_methods(plan)
    roles = [("A", "source")] + [(m, r) for m in methods if m != "A"
                                for r in ("ema", "student")]
    expected = {(ds, domain, method, role) for ds, domains in DOMAINS.items()
                for domain in domains for method, role in roles}
    if (len(plan["runs"]) != len(expected)
            or {(r["dataset"], r["domain"], r["method"], r["role"])
                for r in plan["runs"]} != expected
            or len({r["run_id"] for r in plan["runs"]}) != len(expected)):
        raise ValueError("Qualitative plan must cover exactly the declared full-test groups")
    adaptation_seed = plan.get("adaptation_seed", 42)
    if adaptation_seed not in (42, 43, 44):
        raise ValueError("Qualitative adaptation seed must be 42,43,44")
    selections = {}
    for run in plan["runs"]:
        validate_run(run)
        ds = run["dataset"]
        legacy_id = f"{ds}/{run['domain']}/{run['method']}/{run['role']}"
        seeded_id = f"{ds}/{run['domain']}/seed_{run['seed']}/{run['method']}/{run['role']}"
        allowed = {legacy_id} if "methods" not in plan else {legacy_id, seeded_id}
        if run["method"] not in "ABCDEF":
            allowed = {seeded_id}
            iteration = ({"RSAR": 531, "DIOR": 369} if run["method"] == "SFYOLO"
                         else {"RSAR": 266, "DIOR": 185})[ds]
            if (not run.get("native_prediction") or not run.get("export_code_sha")
                    or run.get("allowed_gpus") not in ([4, 5, 6], [4, 5, 6, 7])
                    or run.get("final_checkpoint_iteration") != iteration
                    or run.get("detector_epochs") != (2 if run["method"] == "SFYOLO" else 1)
                    or (run["method"] == "SFYOLO"
                        and (run.get("budget_group") != "extended_two_epoch_TAM"
                             or run.get("rank_with_common_one_epoch") is not False))):
                raise ValueError("Port ROI requires native bindings and method-specific final budget")
        if (run["run_id"] not in allowed
                or run["seed"] != (42 if run["method"] == "A" else adaptation_seed)):
            raise ValueError("Qualitative run identity differs from the fixed matrix")
        domain = "source" if run["method"] == "A" else run["domain"]
        if run["checkpoint_domain"] != domain:
            raise ValueError("Qualitative evidence uses a checkpoint from another domain")
        ids = (run["image_ids"], run["visualization_image_ids"])
        if ds in selections and selections[ds] != ids:
            raise ValueError("Qualitative image selection differs across methods/domains")
        selections[ds] = ids
        if run["show_score_thr"] != 0.3:
            raise ValueError("Only the frozen .3 visualization/embedding threshold is supported")


def inspect_embedding(entry, plan):
    root = host.read_path(entry["directory"])
    ds, domain, role = entry["dataset"], entry["domain"], entry["comparison"]
    runs = comparison_runs(plan, ds, domain, role)
    files = ["protocol.json", "embedding.npz", "points.csv", "predicted_class_legend.pdf"]
    files += [f"{ds}_{domain}_{role}_{r['method']}_{r['role']}.pdf" for r in runs]
    missing = [name for name in files
               if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        return {"status": "incomplete", "problems": missing, "n_points": None}
    protocol = read_json(root / "protocol.json")
    if (protocol["schema"] != SCHEMA or not host.same_data(protocol["runs"], runs)
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
    if not 0 < count <= protocol["sample_cap"]:
        raise ValueError("Invalid equal sample count")
    if coords.shape != (len(runs) * count, 2) or len(points) != len(coords):
        raise ValueError("Joint embedding does not contain all declared equally sampled panels")
    if normalized.shape != (len(points), len(mean)) or not np.isfinite(coords).all():
        raise ValueError("Invalid joint embedding shape/values")
    for i, run in enumerate(runs):
        export_index, _ = load_export(run)
        if protocol["feature_stats"][i]["code_commit"] != export_index["code_commit"]:
            raise ValueError("Embedding feature producer differs from its export")
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
                    or not host.same_data(point["checkpoint"], run["checkpoint"])
                    or point["dataset"] != ds or point["domain"] != domain
                    or int(point["seed"]) != run["seed"]
                    or not host.same_data(point["config"], run["config"])
                    or point["checkpoint_domain"] != run["checkpoint_domain"]
                    or not host.same_data(str(feature_file.parent), run["out_dir"])
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
    if manifest.get("qualitative_evidence"):
        return reuse_completed_evidence(manifest)
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
        identities = {}
        for run in plan["runs"]:
            selected = run["run_id"] in selected_ids
            group = {
                "run_id": run["run_id"], "scope": "full_test",
                "expected_images": len(run["image_ids"]),
                "status": "pending" if selected else "not_inspected",
                "inspected": selected, "validated_images": 0, "detection_rows": 0,
                "visualizations_complete": 0, "out_dir": run["out_dir"],
                **{k: run[k] for k in ("dataset", "domain", "seed", "method", "role")},
                "budget_group": run.get("budget_group", "source" if run["method"] == "A"
                                        else "common_one_epoch"),
                "checkpoint": run["checkpoint"],
            }
            group_index.append(group)
            groups[run["run_id"]] = group
            identities[tuple(run[k] for k in ("dataset", "domain", "seed", "method", "role"))] = group
        selected_plan = {**plan, "runs": [r for r in plan["runs"] if r["run_id"] in selected_ids]}
        for row in collect(selected_plan):
            identified += 1
            roi_complete += row["roi_status"] == "complete"
            vis_complete += row["vis_status"] == "complete"
            group = identities[tuple(row[k] for k in ("dataset", "domain", "seed", "method", "role"))]
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
            "adaptation_seed": plan.get("adaptation_seed", 42) if plan else 42,
            "methods": ",".join(plan_methods(plan)) if plan else "A,B,C,D,E,F",
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
        "vis_expected_image_roles": (sum(len(r["visualization_image_ids"]) for r in plan["runs"])
                                     if plan else 3520),
        "vis_complete": vis_complete,
        "embeddings_expected": 24,
        "embeddings_complete": sum(r["status"] == "complete" for r in indices),
        "full_test_roi": "complete" if roi_complete == roi_expected else "incomplete",
        "methods": plan_methods(plan) if plan else list("ABCDEF"),
        "adaptation_seed": plan.get("adaptation_seed", 42) if plan else 42,
    }
    return rows, indices, coverage


def reuse_completed_evidence(manifest):
    """Reuse the accepted streaming audit; read indices, never feature NPZs."""
    if manifest.get("inspect_roi_run_ids") is not None:
        raise ValueError("Completed qualitative reuse cannot be combined with scoped inspection")
    root = host.read_path(manifest["qualitative_evidence"])
    summary = read_json(root / "qualitative_summary.json")
    fragment = read_json(root / "report_input_fragment.json")
    plan = read_json(manifest["qualitative_plan"])
    validate_plan(plan)
    if (summary["schema"] != "iraod-qualitative-completion-summary-v1"
            or summary["qualitative_status"] != "complete"
            or summary["full_test_roi"] != "complete"
            or not host.same_data(read_json(summary["plan"]), plan)
            or not host.same_data(read_json(fragment["qualitative_plan"]), plan)):
        raise ValueError("Accepted qualitative evidence does not bind this full-test plan")
    for field, name in (("roi_image_manifest", "roi_vis_coverage.csv"),
                        ("roi_group_summary", "roi_group_summary.csv"),
                        ("embedding_index", "embedding_index.csv"),
                        ("report_input_fragment", "report_input_fragment.json")):
        path = root / name
        if host.read_path(summary[field]).resolve() != path.resolve() or not path.stat().st_size:
            raise ValueError(f"Missing or mismatched accepted qualitative artifact: {field}")
    groups = read_rows(root / "roi_group_summary.csv")
    by_id = {g["run_id"]: g for g in groups}
    if len(groups) != len(plan["runs"]) or set(by_id) != {r["run_id"] for r in plan["runs"]}:
        raise ValueError("Accepted qualitative groups differ from the declared plan")
    visualizations = []
    for run in plan["runs"]:
        group = by_id[run["run_id"]]
        for field in ("expected_images", "validated_images", "detection_rows",
                      "visualizations_complete"):
            group[field] = int(group[field])
        if (group["status"] != "complete" or group["scope"] != "full_test"
                or group["inspected"] != "True" or not host.same_data(group["out_dir"], run["out_dir"])
                or group["expected_images"] != len(run["image_ids"])
                or group["validated_images"] != len(run["image_ids"])
                or group["visualizations_complete"] != len(run["visualization_image_ids"])):
            raise ValueError("Accepted group binding/count mismatch")
        group["inspected"] = True
        if "methods" in plan:
            expected_binding = {k: run[k] for k in ("dataset", "domain", "seed", "method",
                                                    "role", "checkpoint")}
            expected_binding["budget_group"] = run.get(
                "budget_group", "source" if run["method"] == "A" else "common_one_epoch")
            if any(not host.same_data(str(group[k]), str(value))
                   for k, value in expected_binding.items()):
                raise ValueError("Accepted group method/seed/role/budget binding mismatch")
        index, _ = load_export(run)  # The returned NPZ iterator is deliberately not consumed.
        if sum(r["n_detections"] for r in index["records"]) != group["detection_rows"]:
            raise ValueError("Accepted detection count differs from export index")
        vis = read_json(Path(run["out_dir"]) / "visualizations/index.json")
        if (vis["schema"] != SCHEMA or vis["status"] != "complete"
                or not host.same_data(vis["run"], run)
                or [r["image_id"] for r in vis["images"]] != run["visualization_image_ids"]):
            raise ValueError("Accepted visualization binding differs from the frozen selection")
        for image in vis["images"]:
            visualizations.append({
                **{k: run[k] for k in ("dataset", "domain", "method", "role", "seed",
                                       "checkpoint", "checkpoint_domain", "config")},
                "corruption": run["domain"], "image": image["image_id"],
                "show_dir": str(host.read_path(run["out_dir"]) / "visualizations"),
                "visualization_file": str(host.read_path(run["out_dir"]) / "visualizations" / image["file"]),
                "status": "complete", "evidence": str(root / "qualitative_summary.json"),
            })
    embeddings = read_rows(root / "embedding_index.csv")
    expected = {(ds, d, role) for ds, domains in DOMAINS.items()
                for d in domains for role in ("ema", "student")}
    identity = lambda e: (e["dataset"], e["domain"], e["comparison"])
    if len(embeddings) != 24 or {identity(e) for e in embeddings} != expected:
        raise ValueError("Accepted embeddings differ from the 24-comparison matrix")
    provided = manifest.get("embeddings") or fragment["embeddings"]
    if (len(provided) != 24 or not host.same_data(
            {identity(e): e["directory"] for e in provided},
            {identity(e): e["directory"] for e in fragment["embeddings"]})):
        raise ValueError("Requested embeddings differ from the accepted collector")
    for entry in embeddings:
        entry["n_points"] = int(entry["n_points"])
        entry["problems"] = []
        protocol = read_json(Path(entry["directory"]) / "protocol.json")
        runs = comparison_runs(plan, entry["dataset"], entry["domain"], entry["comparison"])
        if (entry["status"] != "complete" or protocol["schema"] != SCHEMA
                or not host.same_data(protocol["runs"], runs) or protocol["sampling_seed"] != 42
                or protocol["tsne_parameters"]["random_state"] != 42
                or entry["n_points"] != len(runs) * protocol["points_per_method"]
                or not 0 < protocol["points_per_method"] <= 1000
                or not host.same_data(entry["directory"], next(
                    e["directory"] for e in provided if identity(e) == identity(entry)))):
            raise ValueError("Accepted embedding binding/count mismatch")
    totals = {
        "roi_expected_image_roles": sum(g["expected_images"] for g in groups),
        "roi_identified_image_roles": sum(g["validated_images"] for g in groups),
        "roi_complete": sum(g["validated_images"] for g in groups),
        "roi_detection_rows": sum(g["detection_rows"] for g in groups),
        "roi_inspected_groups": len(groups), "roi_complete_groups": len(groups),
        "vis_expected_image_roles": len(visualizations), "vis_complete": len(visualizations),
        "embeddings_expected": len(embeddings), "embeddings_complete": len(embeddings),
        "embedding_points": sum(e["n_points"] for e in embeddings),
    }
    if any(summary[k] != v for k, v in totals.items()):
        raise ValueError("Accepted qualitative summary counts disagree with compact indices")
    coverage = {
        **totals, "roi_scope": "full_test", "full_test_roi": "complete",
        "roi_group_index": groups, "visualization_index": visualizations,
        "roi_image_manifest": str(host.read_path(summary["roi_image_manifest"])),
        "evidence_reuse": str(root / "qualitative_summary.json"),
        "validation_scope": "accepted streaming NPZ audit reused; current plan/index bindings and counts",
    }
    return [], embeddings, coverage
