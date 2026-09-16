"""Add the twelve corrected-checkpoint results without changing frozen metrics."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "current_20260913"
sys.path.insert(0, str(PREVIOUS))
import build_report as previous  # noqa: E402

PORT_FIX = "e6599be4fab9b2da07ea1abd87cc8473be2e0d60"
SFYOLO_FIX = "82e08b0b5f15278e24f949ae1670f91f86713f8d"
ORIGIN = "corrected_checkpoint_native"
DOMAINS = {
    "DIOR": ("clean", "brightness", "cloudy", "contrast"),
    "RSAR": (
        "clean", "am_noise_horizontal", "am_noise_vertical", "chaff",
        "gaussian_white_noise", "noise_suppression", "point_target", "smart_suppression",
    ),
}
STUDENT_METHODS = ("B", "C", "D", "E", "F", "IRG", "LPLD", "SFUT", "AASFOD", "SFYOLO")
TRAIN_METHODS = STUDENT_METHODS + (
    "B_REG", "F_text_only", "F_veto_only", "LoRA-CGA", "LoRA-CGA+VLST",
)


def recovered_rows(original, native, terminal, post_binder, model, class_text, source, names, cutoff):
    identity = previous.canonical(original)
    execution = native["execution"]
    previous.require(original["mAP50"] == "", f"Cannot replace a valid metric: {identity}")
    previous.require(
        native["canonical_key"] == terminal["canonical_key"] == post_binder["canonical_key"]
        == model["canonical_key"]
        == previous.canonical(execution) == identity,
        f"Recovered case/role mismatch: {identity}",
    )
    previous.require(
        execution["source_id"] == original["source_id"] == source["source_id"],
        f"Recovered source mismatch: {identity}",
    )
    checkpoint = execution["checkpoint"]
    iteration = 531 if original["method"] == "SFYOLO" else 266
    suffix = "_ema" if original["role"] == "ema" else ""
    previous.require(
        checkpoint == model["evaluated_checkpoint"]
        and Path(checkpoint).name == Path(model["source_checkpoint"]).name
        == f"iter_{iteration}{suffix}.pth",
        f"Corrected final checkpoint/role mismatch: {identity}",
    )
    expected_sha = SFYOLO_FIX if original["method"] == "SFYOLO" else PORT_FIX
    previous.require(
        execution["training_code_sha"] == model["training_code_sha"] == expected_sha,
        f"Corrected training revision mismatch: {identity}",
    )
    previous.require(
        native["eval_status"] == post_binder["native_eval_status"] == "complete"
        and terminal["status"] == post_binder["status"] == "COMPLETE_TEST_ROI",
        f"Missing genuine TEST/ROI completion: {identity}",
    )
    previous.require(
        original["dataset"] == "RSAR"
        and native["pred_count"] == post_binder["pred_count"] == post_binder["roi_n_images"] == 8538
        and bool(post_binder["roi_index"]),
        f"Incomplete recovered TEST/ROI coverage: {identity}",
    )
    metric = native["metric"]["metric"]["mAP"]
    previous.require(
        isinstance(metric, (int, float)) and 0 <= metric <= 1 and metric == post_binder["mAP"],
        f"Invalid or inconsistent native mAP: {identity}",
    )
    previous.require(
        native["metric"]["config"] == execution["config"]
        and Path(native["metric_json"]).parent == Path(native["eval_dir"])
        and Path(native["eval_dir"]).parent == Path(terminal["output_root"])
        and native["class_ap_text"] == post_binder["class_ap_remote_path"],
        f"Recovered evaluator/metric binding mismatch: {identity}",
    )
    previous.require(
        datetime.fromisoformat(previous.CUTOFF)
        < datetime.fromisoformat(native["evaluated_at"])
        <= datetime.fromisoformat(cutoff),
        f"Recovered metric outside snapshot interval: {identity}",
    )
    evidence = f"recovered_results.json:records[{identity}]"
    row = {
        **original,
        "status": previous.VALID,
        "eligible": "true",
        "native_mAP50": metric,
        "mAP50": metric,
        "mAP50_percent": 100 * metric,
        "delta_vs_source_pp": 100 * (metric - float(source["mAP50"])),
        "relative_vs_source_percent": 100 * (metric / float(source["mAP50"]) - 1),
        "host": native["host"],
        "checkpoint": checkpoint,
        "eval_json": native["metric_json"],
        "eval_status": str(Path(terminal["output_root"]) / "native_pass.json"),
        "prediction_image_ids": str(Path(native["eval_dir"]) / "predictions.pkl.image_ids.json"),
        "class_ap": native["class_ap_text"],
        "n_images": native["pred_count"],
        "n_post_nms_detections": "",
        "evaluated_at": native["evaluated_at"],
        "collected_at": cutoff,
        "training_code_sha": expected_sha,
        "evaluation_code_sha": execution["evaluation_code_sha"],
        "config": execution["config"],
        "evidence_basis": evidence,
        "note": (
            "Corrected full-budget final; genuine native TEST and accepted complete ROI. "
            "Original held/invalid record remains in current_20260913. "
            "Prediction/ID coverage reuses operator validation; payloads not reopened here."
        ),
        "report_cutoff_utc": cutoff,
        "metric_origin": ORIGIN,
    }
    classes = []
    for line in class_text.splitlines():
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) == 5 and fields[1].isdigit() and fields[2].isdigit():
            name, value = fields[0], fields[4]
            previous.require(0 <= float(value) <= 1, f"Invalid class AP: {identity}/{name}")
            classes.append({
                **{field: row[field] for field in ("dataset", "domain", "method", "seed", "role")},
                "class_name": name,
                "native_AP50": value,
                "AP50": value,
                "AP50_percent": 100 * float(value),
                "status": previous.VALID,
                "precision": "printed class_ap.txt (typically 3 decimals)",
                "evidence": f".{native['host']}:{native['class_ap_text']}",
            })
    actual_names = [row["class_name"] for row in classes]
    previous.require(
        len(actual_names) == len(set(actual_names)) and set(actual_names) == names,
        f"Recovered class coverage mismatch: {identity}",
    )
    return row, classes


def assemble(payload):
    original = previous.load_csv(PREVIOUS, "raw_metrics.csv")
    original_classes = previous.load_csv(PREVIOUS, "per_class.csv")
    by_key = {previous.canonical(row): row for row in original}
    missing = {identity for identity, row in by_key.items() if row["mAP50"] == ""}
    records = payload["records"]
    models = payload["models"]
    record_map = {record["native"]["canonical_key"]: record for record in records}
    model_map = {model["canonical_key"]: model for model in models}
    previous.require(
        len(original) == len(by_key) == 912
        and sum(row["mAP50"] != "" for row in original) == 900,
        "Expected the frozen 900-of-912 TEST snapshot",
    )
    previous.require(
        len(records) == len(record_map) == len(models) == len(model_map) == len(missing) == 12
        and set(record_map) == set(model_map) == missing,
        "Exactly the twelve missing corrected-checkpoint keys are required",
    )
    sources = {
        (row["dataset"], row["domain"]): row for row in original if row["method"] == "A"
    }
    names = {
        row["class_name"] for row in original_classes
        if row["dataset"] == "RSAR" and row["method"] == "A"
    }
    classes = [
        row for row in original_classes if previous.canonical(row) not in missing
    ]
    raw = []
    for original_row in original:
        identity = previous.canonical(original_row)
        if identity in missing:
            record = record_map[identity]
            row, class_rows = recovered_rows(
                original_row, record["native"], record["terminal"], record["post_binder"],
                model_map[identity],
                record["class_ap_text"],
                sources[(original_row["dataset"], original_row["domain"])],
                names, payload["metric_cutoff_utc"],
            )
            raw.append(row)
            classes.extend(class_rows)
        else:
            raw.append(dict(original_row))
    previous.require(
        all(row == by_key[previous.canonical(row)] for row in raw
            if previous.canonical(row) not in missing),
        "A frozen valid row changed",
    )
    previous.require(
        len(classes) == len({previous.key(row) + (row["class_name"],) for row in classes}),
        "Duplicate class rows after recovery",
    )
    previous.require(
        all(row["status"] == previous.VALID and row["mAP50"] != "" for row in raw),
        "The completed TEST view still has missing or invalid metrics",
    )
    return raw, classes


def validate_delivery(delivery):
    old = delivery["canonical_roi720"]
    corrected = delivery["corrected_test_roi12"]
    train = delivery["train_repair6"]
    groups = (old, corrected, train)
    previous.require(
        delivery["status"] == "DELIVERED"
        and all(row["status"] == "DELIVERED" and row["pending"] == row["active"] == 0
                for row in groups),
        "Delivery still has pending, active or non-delivered sets",
    )
    expected = {
        previous.canonical(row) for row in previous.load_csv(previous.BASE, "roi_coverage.csv")
    }
    previous.require(
        len(old["keys"]) == len(set(old["keys"])) == 720
        and len(corrected["keys"]) == len(set(corrected["keys"])) == 12
        and set(old["keys"]).isdisjoint(corrected["keys"])
        and set(old["keys"]) | set(corrected["keys"]) == expected
        and len(expected) == 732,
        "Final delivery does not match the exact original 732 ROI keys",
    )
    historical, rollout, pilot = old["historical377"], old["rollout342"], old["pilot1"]
    previous.require(
        historical["roi_keys"] == 377 and historical["files"] == 338
        and pilot["key"] in old["keys"] and pilot["files"] == 1,
        "Historical/pilot membership or provider-object counts differ",
    )
    all_paths = []
    for group, count, keys in ((rollout, 342, old["keys"]), (corrected, 12, corrected["keys"])):
        entries = group["provider_receipts"]
        names = {f"case-{key.replace('/', '-')}.tar" for key in keys}
        previous.require(
            len(entries) == len({row["name"] for row in entries}) == group["files"] == count
            and sum(row["bytes"] for row in entries) == group["bytes"],
            "Provider receipt count/bytes differ from the delivered set",
        )
        for row in entries:
            previous.require(
                row["name"] in names and Path(row["netdisk_path"]).name == row["name"]
                and row["provider_size"] == row["bytes"]
                and row["provider_md5"] == row["source_md5"]
                and re.fullmatch(r"[0-9a-f]{32}", row["source_md5"]) is not None,
                f"Provider size/whole-MD5 or membership mismatch: {row['name']}",
            )
            all_paths.append(row["netdisk_path"])
    previous.require(len(all_paths) == len(set(all_paths)), "Duplicate provider destinations")
    previous.require(
        old["files"] == historical["files"] + rollout["files"] + pilot["files"] == 681
        and old["bytes"] == historical["bytes"] + rollout["bytes"] + pilot["bytes"],
        "Canonical720 provider totals differ from its partitions",
    )
    cells = {row["cell"] for row in train["cells"]}
    previous.require(
        len(train["cells"]) == len(cells) == train["files"] == 6
        and cells == {key.rsplit("/", 1)[0] for key in corrected["keys"]}
        and sum(row["bytes"] for row in train["cells"]) == train["bytes"]
        and train["provider_size_and_whole_md5_verified"] is True,
        "Training repair archive identity/verification differs",
    )
    totals = delivery["totals"]
    previous.require(
        totals["provider_files"] == sum(row["files"] for row in groups) == 699
        and totals["bytes"] == sum(row["bytes"] for row in groups) == 336958086405
        and totals["pending"] == totals["active"] == 0,
        "Final provider-object totals or closure differ",
    )
    return totals


def validate_coverage(raw, classes):
    train = {
        (dataset, domain, str(seed), method)
        for dataset, domains in DOMAINS.items() for domain in domains
        for seed in (42, 43, 44) for method in TRAIN_METHODS
    }
    recorded_train = previous.load_csv(previous.BASE, "training_status.csv")
    actual_train = {
        tuple(row[field] for field in ("dataset", "domain", "seed", "method"))
        for row in recorded_train
    }
    previous.require(
        len(recorded_train) == len(train) == 540 and actual_train == train,
        "Canonical TRAIN matrix has missing, duplicate or extra cells",
    )
    source = {
        (dataset, domain, "42", "A", "source")
        for dataset, domains in DOMAINS.items() for domain in domains
    }
    ema = {cell + ("ema",) for cell in train}
    student = {cell + ("student",) for cell in train if cell[3] in STUDENT_METHODS}
    expected_test = {"/".join(cell) for cell in source | ema | student}
    actual_test = {previous.canonical(row) for row in raw}
    previous.require(
        len(raw) == 912 and actual_test == expected_test,
        "Canonical TEST matrix has missing, duplicate or extra roles",
    )
    expected_roi = {
        "/".join(cell) for cell in source | student
        | {cell + ("ema",) for cell in train if cell[3] in STUDENT_METHODS}
    }
    roi = previous.load_csv(previous.BASE, "roi_coverage.csv")
    previous.require(
        len(roi) == 732 and {previous.canonical(row) for row in roi} == expected_roi,
        "Canonical ROI matrix differs from required method/role coverage",
    )
    class_counts = Counter(previous.canonical(row) for row in classes)
    previous.require(
        set(class_counts) == expected_test
        and all(class_counts[previous.canonical(row)] == {"DIOR": 20, "RSAR": 6}[row["dataset"]]
                for row in raw),
        "Missing or extra per-class coverage",
    )
    prerequisites = previous.load_csv(previous.BASE, "prerequisites.csv")
    previous.require(
        Counter(row["method"] for row in prerequisites)
        == {"TSD": 36, "TAM": 12, "LoRA_adapter": 2, "source_detector": 2}
        and len({tuple(row[field] for field in ("dataset", "domain", "seed", "method"))
                 for row in prerequisites}) == 52,
        "Prerequisite matrix has missing, duplicate or extra cells",
    )
    return {"training_cells": 540, "prerequisite_cells": 52,
            "test_roles": 912, "roi_roles": 732, "missing_or_extra_keys": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = json.loads((ROOT / "recovered_results.json").read_text())
    delivery = validate_delivery(json.loads((ROOT / "delivery_manifest.json").read_text()))
    raw, classes = assemble(payload)
    coverage_audit = validate_coverage(raw, classes)
    seeds, domains, summary, coverage = previous.aggregate(raw)
    recovered = Counter(
        (row["dataset"], row["method"], row["role"])
        for row in raw if row["metric_origin"] == ORIGIN
    )
    for row in coverage:
        row["report_cutoff_utc"] = payload["metric_cutoff_utc"]
        row["recovered_native_cells"] = recovered[(row["dataset"], row["method"], row["role"])]
    tables = dict(
        raw_metrics=raw, per_class=classes, per_seed=seeds, per_domain=domains,
        summary=summary, coverage=coverage,
    )
    for name, rows in tables.items():
        path = ROOT / f"{name}.csv"
        text = previous.csv_text(rows)
        if args.check:
            previous.require(path.read_text() == text, f"Stale generated output: {path.name}")
        else:
            path.write_text(text)
    print(json.dumps({
        "status": "PASS", "valid_test_keys": len(raw), "recovered_keys": sum(recovered.values()),
        "frozen_valid_rows_preserved": 900, "class_rows": len(classes),
        "summary_rows": len(summary), "metric_cutoff_utc": payload["metric_cutoff_utc"],
        "archive_validation": "PASS", "delivered_roi_keys": 732,
        "provider_files": delivery["provider_files"], "provider_bytes": delivery["bytes"],
        "coverage_audit": coverage_audit,
        "mode": "check" if args.check else "build",
    }))


if __name__ == "__main__":
    main()
