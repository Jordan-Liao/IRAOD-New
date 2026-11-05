"""Import selected scientific metadata, not private session stores or model data."""

import argparse
import json
from pathlib import Path

import build_completed_report as report


def read(path):
    return json.loads(Path(path).read_text())


def select(record, fields):
    return {field: record[field] for field in fields if field in record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator_packet", type=Path)
    args = parser.parse_args()
    packet = read(args.operator_packet)
    completion = read(packet["terminal"])
    mapping = read(packet["checkpoint_mapping"])
    report.previous.require(
        packet["status"] == "READY" and completion["status"] == "COMPLETE",
        "Operator report input is not complete",
    )
    mapped = {row["canonical_key"]: row for row in mapping["records"]}
    results = packet["results"]
    report.previous.require(
        mapping["count"] == len(mapped) == len(results) == 12,
        "Expected twelve actual evaluated-checkpoint bindings",
    )
    counts = completion["counts"]
    report.previous.require(
        counts["native_test_complete"] == counts["roi_complete"] == 12
        and sum(row["starts_used"] for row in results) == counts["actual_starts"]
        and sum(row["gpu_seconds_used"] for row in results) == counts["gpu_seconds"],
        "Completed output counts or charged execution accounting differ",
    )
    payload = {
        "schema": "iraod-corrected-native-results-v1",
        "status": "COMPLETE_12_OF_12",
        "metric_cutoff_utc": completion["completed_utc"],
        "read_scope": (
            "Selected actual native execution/metric/class-AP metadata, accepted complete "
            "terminals and post-binder receipts. Original pre-binder records remain preserved "
            "by the producer. No model, predictions or ROI payload was reopened for this report."
        ),
        "execution_accounting": counts,
        "models": [],
        "records": [],
    }
    for result in results:
        identity = result["canonical_key"]
        native = read(result["native_result_packet_original"])
        terminal = read(result["complete_terminal"])
        bound = read(result["post_binder_packet"])
        model = mapped[identity]
        report.previous.require(
            native["execution"]["checkpoint"] == result["evaluated_checkpoint"]
            == model["declared_evaluated_checkpoint"]
            and native["host"] == result["host"] == model["host"],
            f"Actual-host checkpoint binding differs: {identity}",
        )
        class_body = Path(result["class_ap_body_path"]).read_text()
        report.previous.require(
            bound["class_ap_body"] == class_body,
            f"Class-AP body differs from its accepted packet: {identity}",
        )
        if "exit_code" in terminal:
            report.previous.require(
                terminal["exit_code"] == 0 and terminal["timed_out"] is False,
                f"Non-successful GPU terminal selected: {identity}",
            )
        payload["models"].append({
            "canonical_key": identity,
            "source_checkpoint": model["source_checkpoint_evidence"],
            "evaluated_checkpoint": model["declared_evaluated_checkpoint"],
            "training_code_sha": model["training_code_sha"],
            "host": model["host"],
            "obsolete_staged_checkpoint_not_used": model["obsolete_staged_checkpoint_not_used"],
        })
        native_fields = select(native, (
            "canonical_key", "host", "eval_dir", "metric_json", "metric", "class_ap_text",
            "eval_status", "pred_count", "roi_index", "roi_n_images", "evaluated_at",
        ))
        native_fields["execution"] = select(native["execution"], (
            "dataset", "domain", "seed", "method", "role", "checkpoint", "config",
            "ann_file", "img_prefix", "training_code_sha", "source_id", "evaluation_code_sha",
            "joint_exporter", "command",
        ))
        payload["records"].append({
            "native": native_fields,
            "terminal": select(terminal, (
                "schema", "canonical_key", "status", "host", "output_root", "exit_code",
                "timed_out", "elapsed_seconds", "allocated_gpu_seconds", "salvaged_cpu_binder",
                "checkpoint", "training_code_sha", "metric_json", "mAP", "pred_count",
                "roi_index", "roi_n_images", "completed_utc", "automatic_retry",
            )),
            "post_binder": select(bound, (
                "schema", "canonical_key", "status", "native_eval_status", "mAP",
                "pred_count", "roi_index", "roi_n_images", "class_ap_remote_path", "preservation",
            )),
            "class_ap_text": class_body,
        })
    raw, classes = report.assemble(payload)
    destination = report.ROOT / "recovered_results.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "status": "IMPORTED", "recovered_keys": len(results),
        "valid_test_keys": len(raw), "class_rows": len(classes),
        "frozen_valid_rows_preserved": 900, "output": str(destination),
    }))


if __name__ == "__main__":
    main()
