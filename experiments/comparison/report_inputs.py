"""Read owner manifests and the actual run_eval_full.sh outputs, CPU only."""

import ast
import csv
import math
from pathlib import Path
import pickle
import shlex

import numpy as np

from experiments.comparison.result_completion import DOMAINS, read_json
from experiments.comparison import host_binding as host
from experiments.comparison.report_statistics import SEEDS, declared_methods, method_group
from tools.prediction_export import IMAGE_ORDER_ORIGIN, IMAGE_ORDER_SCHEMA
from experiments.comparison.source_provenance import annotate_producer
from experiments.comparison.labels import CLASSES


EXPECTED_IMAGES = {"RSAR": 8538, "DIOR": 11738}
FINAL_ITERATION = {"RSAR": 266, "DIOR": 185}
SFYOLO_FINAL_ITERATION = {"RSAR": 531, "DIOR": 369}
HISTORICAL_ROOT = Path(__file__).resolve().parents[2] / "results/paper_comparison"


def historical_paths(manifest, per_class=False):
    names = (("per_class_summary.csv", "dior_per_class.csv") if per_class else
             ("raw_results.csv", "dior_raw_results.csv"))
    extra = manifest.get("historical_per_class" if per_class else "historical_raw", [])
    frozen = HISTORICAL_ROOT / "historical/seed42"
    root = frozen if frozen.is_dir() else HISTORICAL_ROOT
    return list(dict.fromkeys(str(Path(p).resolve())
                             for p in [*(root / n for n in names), *extra]))


def read_rows(value):
    if isinstance(value, list):
        return value
    path = Path(value)
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array or CSV rows: {path}")
    return rows


def key(row):
    return (row["dataset"], row["domain"], row["method"], int(row["seed"]), row["role"])


def keyed(rows, methods=None):
    methods = declared_methods(methods)
    result = {}
    for row in rows:
        identity = key(row)
        ds, domain, method, seed, role = identity
        if (ds not in DOMAINS or domain not in DOMAINS[ds] or method not in methods
                or seed not in SEEDS or (method == "A" and (role != "source" or seed != 42))
                or (method != "A" and role not in ("ema", "student"))):
            raise ValueError(f"Invalid cell identity: {identity}; A is a single seed42 source")
        if identity in result:
            raise ValueError(f"Duplicate cell: {identity}")
        result[identity] = row
    return result


def historical_values(paths):
    result = {}
    for path in paths:
        for row in read_rows(path):
            if row["corruption"] == "clean_val":
                continue
            identity = (row["dataset"], row["corruption"], row["method"],
                        int(row["seed"]), row["ckpt_role"])
            if identity[3] != 42:
                raise ValueError("Historical locks must be original seed42 rows")
            value = float(row["mAP50"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Historical mAP50 must use 0..1 units")
            if identity in result and result[identity]["value"] != value:
                raise ValueError(f"Conflicting historical rows: {identity}")
            result[identity] = {"value": value, "file": str(path)}
    return result


def class_table(path, dataset):
    rows, columns = [], None
    for line in Path(path).read_text().splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if cells[0].lower() == "class":
            columns = [c.lower() for c in cells]
            continue
        if cells[0] == "mAP":
            continue
        if columns is None:
            continue
        row = dict(zip(columns, cells))
        if row["class"] not in CLASSES[dataset]:
            raise ValueError(f"Unknown class in {path}: {row['class']}")
        ap = float(row["ap"])
        if not math.isfinite(ap) or not 0 <= ap <= 1:
            raise ValueError(f"Invalid AP in {path}")
        rows.append({"class_name": row["class"], "AP50": ap,
                     "precision": "printed class_ap.txt (typically 3 decimals)"})
    if [r["class_name"] for r in rows] != list(CLASSES[dataset]):
        raise ValueError(f"Missing, duplicate or reordered class table: {path}")
    return rows


def prediction_evidence(path, ids_path, dataset):
    """Owner-generated pickles only; never infer image IDs from a file count."""
    order = read_json(ids_path)
    if (order.get("schema") != IMAGE_ORDER_SCHEMA
            or order.get("origin") != IMAGE_ORDER_ORIGIN
            or order.get("status") != "complete"
            or order.get("predictions_file") != Path(path).name):
        raise ValueError("Prediction image order needs a same-inference producer sidecar, not a backfill")
    ids = order["image_ids"]
    if len(ids) != EXPECTED_IMAGES[dataset] or len(set(ids)) != len(ids):
        raise ValueError("Prediction image-order evidence must cover unique full TEST IDs")
    if (order["n_images"] != len(ids) or order["dataset_size"] != len(ids)
            or len(order["records"]) != len(ids)
            or not order["evaluation_code_sha"]):
        raise ValueError("Prediction sidecar count/provenance is incomplete")
    for i, record in enumerate(order["records"]):
        if (record["prediction_index"] != i or record["image_id"] != ids[i]
                or Path(record["ori_filename"]).stem != ids[i]):
            raise ValueError("Prediction sidecar records do not match their saved order")
    with Path(path).open("rb") as stream:
        predictions = pickle.load(stream)
    if len(predictions) != len(ids):
        raise ValueError("Prediction count differs from full TEST image order")
    images = []
    for index, (image_id, per_class) in enumerate(zip(ids, predictions)):
        if len(per_class) != len(CLASSES[dataset]):
            raise ValueError("Prediction class count differs from dataset")
        count = 0
        for boxes in per_class:
            boxes = np.asarray(boxes)
            if boxes.ndim != 2 or boxes.shape[1] != 6 or not np.isfinite(boxes).all():
                raise ValueError("Expected finite per-class post-NMS rotated boxes with scores")
            count += len(boxes)
        images.append({"prediction_image_index": index, "image_id": image_id,
                       "n_post_nms_detections": count})
    return images, order


def inspect_cell(cell, checkpoint, source_id):
    """A successful eval_status alone is insufficient: the script writes it early."""
    result = {**cell, "seed": int(cell["seed"]), "status": "incomplete", "mAP50": None,
              "eval_mAP50": None, "problems": list(cell.get("collection_problems", [])),
              "n_predictions": None,
              "n_post_nms_detections": None}
    issues = result["problems"]
    classes, images = [], []
    if checkpoint is None:
        issues.append("missing_checkpoint_manifest_entry")
    else:
        if (checkpoint.get("verified") not in (True, "true")
                or checkpoint.get("source_id") != source_id):
            issues.append("checkpoint_unverified_or_wrong_source")
        expected_role = "source" if cell["method"] == "A" else "final"
        if checkpoint.get("selection") != expected_role:
            issues.append("checkpoint_not_fixed_source_or_final")
        if cell["method"] != "A":
            iteration = (SFYOLO_FINAL_ITERATION if cell["method"] == "SFYOLO"
                         else FINAL_ITERATION)[cell["dataset"]]
            suffix = "_ema" if cell["role"] == "ema" else ""
            if (int(checkpoint.get("iteration", -1)) != iteration
                    or Path(checkpoint["path"]).name != f"iter_{iteration}{suffix}.pth"):
                issues.append("checkpoint_not_exact_final_iteration")
        result["checkpoint"] = checkpoint["path"]
        result["source_id"] = checkpoint.get("source_id")
        result["gpu_provenance"] = checkpoint.get("gpu_provenance", "")
    out = Path(cell["eval_dir"])
    files = {"eval_status": out / "eval_status", "class_ap": out / "class_ap.txt",
             "predictions": out / "predictions.pkl", "pred_count": out / "pred_count.txt"}
    # Explicit eval JSON selection avoids choosing a stale successful retry by mtime.
    files["eval_json"] = Path(cell["eval_json"]) if cell.get("eval_json") else None
    for name, path in files.items():
        result[name] = str(path) if path is not None else None
        if path is None or not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing_{name}")
    if "missing_eval_status" not in issues:
        matches = [line for line in files["eval_status"].read_text().splitlines()
                   if line.startswith("eval_exit=")]
        fields = dict(token.split("=", 1) for token in shlex.split(matches[-1])
                      if "=" in token) if matches else {}
        wanted = {"eval_exit": "0", "name": cell["method"], "domain": cell["domain"],
                  "seed": str(cell["seed"])}
        if "student" in fields:
            wanted["student"] = result.get("checkpoint")
            wanted["role"] = cell["role"]
        elif "checkpoint" in fields:
            wanted["checkpoint"] = result.get("checkpoint")
            wanted["role"] = cell["role"]
        elif "ema" in fields:
            wanted["ema"] = result.get("checkpoint")
        elif "sidecar" in fields and cell.get("prediction_image_ids"):
            # The native wrapper records the sidecar; its checkpoint is checked below.
            wanted["sidecar"] = Path(cell["prediction_image_ids"]).name
        else:
            issues.append("eval_status_missing_checkpoint_identity")
        if any(not host.same_data(fields.get(k), v) for k, v in wanted.items()):
            issues.append("last_eval_status_failed_or_identity_mismatch")
    if "missing_eval_json" not in issues:
        payload = read_json(files["eval_json"])
        metric = float(payload["metric"]["mAP"])
        if not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError("metric.mAP must be finite 0..1; do not use rounded AP50")
        result.update(mAP50=metric, eval_mAP50=metric, config=payload["config"])
        if cell.get("config") and cell["config"] != payload["config"]:
            issues.append("eval_config_mismatch")
    if "missing_class_ap" not in issues:
        text = files["class_ap"].read_text()
        if "MISSING_CLASS_TABLE" in text:
            issues.append("missing_class_table")
        else:
            classes = class_table(files["class_ap"], cell["dataset"])
            for line in text.splitlines():
                if line.strip().startswith("{'mAP':"):
                    table_map = float(ast.literal_eval(line.strip())["mAP"])
                    if result["eval_mAP50"] is not None and table_map != result["eval_mAP50"]:
                        issues.append("class_table_eval_metric_mismatch")
    if "missing_pred_count" not in issues:
        if int(files["pred_count"].read_text().split()[0]) != EXPECTED_IMAGES[cell["dataset"]]:
            issues.append("prediction_count_not_full_test")
    ids_path = cell.get("prediction_image_ids")
    if not ids_path or not Path(ids_path).is_file():
        issues.append("missing_prediction_image_order")
    elif "missing_predictions" not in issues:
        images, order = prediction_evidence(files["predictions"], ids_path, cell["dataset"])
        if (order["checkpoint"] != result.get("checkpoint")
                or order["config"] != result.get("config")):
            issues.append("prediction_sidecar_checkpoint_or_config_mismatch")
        result["training_code_sha"] = order["training_code_sha"]
        result["evaluation_code_sha"] = order["evaluation_code_sha"]
        for field in ("training_code_sha", "evaluation_code_sha"):
            if cell.get(field) and order.get(field) != cell[field]:
                issues.append(f"prediction_sidecar_{field}_mismatch")
        result["n_predictions"] = len(images)
        result["n_post_nms_detections"] = sum(i["n_post_nms_detections"] for i in images)
    if not issues:
        result["status"] = "complete"
    return result, classes, images


def collect_quantitative(manifest, prediction_sink=None):
    roles = tuple(manifest.get("roles", ["ema"]))
    if not roles or len(set(roles)) != len(roles) or set(roles) - {"ema", "student"}:
        raise ValueError("Quantitative roles must be ema and/or student; no source duplication")
    methods = declared_methods(manifest.get("methods"))
    cells = keyed(read_rows(manifest["cells"]), methods)
    checkpoints = keyed(read_rows(manifest["checkpoints"]), methods)
    if any(k[-1] != "source" and k[-1] not in roles for k in cells):
        raise ValueError("Cell role is not in the declared quantitative roles")
    history = historical_values(historical_paths(manifest))
    class_history = {}
    for path in historical_paths(manifest, per_class=True):
        for old in read_rows(path):
            identity = (old["dataset"], old["corruption"], old["method"],
                        int(old.get("seed", 42)), old["ckpt_role"], old["class_name"])
            value = float(old["AP50"])
            if identity in class_history and class_history[identity] != value:
                raise ValueError(f"Conflicting historical class AP: {identity}")
            class_history[identity] = value
    raw, per_class, predictions = [], [], []
    for ds, domains in DOMAINS.items():
        for domain in domains:
            identities = [(ds, domain, "A", 42, "source")] + [
                (ds, domain, m, s, r) for m in methods[1:] for s in SEEDS for r in roles]
            for identity in identities:
                base = dict(zip(("dataset", "domain", "method", "seed", "role"), identity))
                cell = cells.get(identity)
                if cell:
                    row, classes, images = inspect_cell(
                        cell, checkpoints.get(identity), manifest["source_ids"][ds])
                    if "training_code_sha" in row:
                        row = annotate_producer(row, manifest.get("source_provenance", {}).get(ds))
                else:
                    uninspected = (base["method"] != "A" and base["seed"] not in
                                   manifest.get("inspect_quant_seeds", SEEDS))
                    row = {**base, "status": "not_inspected" if uninspected else "not_started",
                           "mAP50": None, "eval_mAP50": None,
                           "problems": ["outside_quant_inspection_scope" if uninspected
                                        else "missing_cell_manifest"]}
                if identity in history:
                    old = history[identity]
                    row["historical_mAP50"] = old["value"]
                    row["historical_file"] = old["file"]
                    row["mAP50"] = old["value"]
                    if row["eval_mAP50"] is not None and row["eval_mAP50"] != old["value"]:
                        row["status"] = "metric_conflict"
                        row["problems"].append("reevaluation_differs_from_locked_seed42")
                if cell:
                    for c in classes:
                        class_key = (*identity, c["class_name"])
                        if class_key in class_history:
                            old = class_history[class_key]
                            c["historical_AP50"], c["eval_AP50"] = old, c["AP50"]
                            if c["AP50"] != old:
                                row["status"] = "metric_conflict"
                                if "reevaluation_class_AP_differs_from_locked_seed42" not in row["problems"]:
                                    row["problems"].append("reevaluation_class_AP_differs_from_locked_seed42")
                            c["AP50"] = old
                    per_class.extend({**base, **c, "status": row["status"],
                                      "evidence": row["class_ap"]} for c in classes)
                    for image in images:
                        evidence = {
                            **base, **image, "predictions": row["predictions"],
                            "prediction_image_ids": cell["prediction_image_ids"],
                            "scope": "full_TEST_post_NMS_predictions",
                            "cell_status": row["status"],
                        }
                        if prediction_sink is None:
                            predictions.append(evidence)
                        else:
                            prediction_sink(evidence)
                if "methods" in manifest:
                    row["comparison_group"] = method_group(base["method"])
                raw.append(row)
    return raw, per_class, predictions, roles
