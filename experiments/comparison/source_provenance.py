"""Explicit source-only producer corrections; never rewrite prediction sidecars."""

from pathlib import Path
import re
import shutil


def read_source_producer(checkpoint, weights_sha256, record_path, snapshot=None):
    record = Path(record_path)
    actual = None
    if record.is_file():
        actual = record.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", actual):
            raise ValueError(f"Invalid source producer commit record: {record}")
        if snapshot is not None:
            Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record, snapshot)
    return {
        "checkpoint": str(checkpoint), "weights_sha256": weights_sha256,
        "producer_record": str(record), "producer_record_snapshot": str(snapshot) if snapshot else None,
        "actual_source_producer_sha": actual,
        "record_status": "present" if actual else "missing",
    }


def annotate_producer(row, source=None):
    result = dict(row)
    recorded = row.get("training_code_sha")
    result["recorded_training_code_sha"] = recorded
    result["effective_training_code_sha"] = recorded
    if row["method"] != "A":
        result["producer_metadata_status"] = "recorded_adaptation_producer"
        return result
    result["effective_training_code_sha"] = None
    result["actual_source_producer_sha"] = None
    result["producer_metadata_status"] = "source_producer_unknown"
    if source is None:
        return result
    if (source["checkpoint"] != row["checkpoint"]
            or source["weights_sha256"] != row["source_id"]):
        raise ValueError("Source producer evidence does not match the checkpoint/source identity")
    actual = source["actual_source_producer_sha"]
    result["actual_source_producer_sha"] = actual
    result["effective_training_code_sha"] = actual
    if actual is None:
        return result
    result["producer_metadata_status"] = "source_record_verified"
    if recorded != actual:
        result["producer_metadata_status"] = "source_producer_corrected_in_report"
        result["producer_metadata_correction"] = {
            **{k: row[k] for k in ("dataset", "domain", "method", "seed", "role")},
            "field": "training_code_sha", "recorded": recorded, "actual": actual,
            "source_checkpoint": source["checkpoint"], "source_weights_sha256": source["weights_sha256"],
            "source_record": source["producer_record"],
            "source_record_snapshot": source["producer_record_snapshot"],
            "sidecar": row["prediction_image_ids"], "sidecar_modified": False,
            "scope": "producer metadata only; weights, predictions, IDs and mAP unchanged",
        }
    return result
