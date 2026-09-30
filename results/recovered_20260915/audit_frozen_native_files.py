"""Match selected frozen TEST files to accepted archive-member metadata."""

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from build_completed_report import previous

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT.parent / "completed_snapshot_20260912T014556Z"
INDEXES = (
    "files-67-numeric.jsonl.gz", "files-221.jsonl.gz",
    "files-183.jsonl.gz", "files-134.jsonl.gz",
)
TEST_ONLY = {"B_REG", "F_text_only", "F_veto_only", "LoRA-CGA", "LoRA-CGA+VLST"}
require = previous.require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((SNAPSHOT / "artifact_manifest.json").read_text())
    assets = {asset["name"]: asset for asset in manifest["assets"]}
    with (ROOT / "raw_metrics.csv").open() as stream:
        selected = [
            row for row in csv.DictReader(stream)
            if row["metric_origin"] == "pr21_frozen_accepted" and row["mAP50"] != ""
        ]
    require(len(selected) == 671, "Expected exactly 671 frozen valid selected TEST keys")
    bound = defaultdict(list)
    metadata_bytes = 0
    for name in INDEXES:
        path = args.index_dir / name
        data = path.read_bytes()
        metadata_bytes += len(data)
        require(
            len(data) == assets[name]["bytes"]
            and hashlib.sha256(data).hexdigest() == assets[name]["sha256"],
            f"Frozen metadata-index identity differs: {name}",
        )
        with gzip.open(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if record["category"] not in (
                    "native_evaluation", "predictions", "other_evaluation_output"
                ):
                    continue
                for binding in record["bindings"]:
                    bound[binding].append(record)
    results, missing, used_assets = [], [], set()
    execution_json_keys = 0
    for row in selected:
        binding = "/".join(row[field] for field in ("dataset", "domain", "method", "seed", "role"))
        entries = bound[binding]
        by_path = {record["source_path"]: record for record in entries}
        sidecar = row["prediction_image_ids"]
        require(sidecar.endswith(".image_ids.json"), f"Unknown ID-sidecar path: {binding}")
        expected = {
            "metric": row["eval_json"],
            "status": row["eval_status"],
            "class_ap": row["class_ap"],
            "image_ids": sidecar,
            "predictions": sidecar.removesuffix(".image_ids.json"),
        }
        absent = [kind for kind, path in expected.items() if path not in by_path]
        if absent:
            missing.append({"canonical_key": row["canonical_key"], "missing_file_kinds": absent})
            continue
        files = []
        for record in entries:
            require(record["source_host"] == row["host"], f"Frozen source-host mismatch: {binding}")
            asset = assets[record["asset_name"]]
            require(
                asset["provider_size"] == asset["bytes"]
                and asset["provider_md5"] == asset["source_md5"],
                f"Frozen archive lacks accepted provider proof: {asset['name']}",
            )
            used_assets.add(asset["name"])
            files.append({
                field: record[field] for field in (
                    "source_path", "category", "bytes", "sha256", "asset_name", "archive_member"
                )
            })
        has_execution = any(Path(record["source_path"]).name == "execution.json" for record in entries)
        execution_json_keys += has_execution
        results.append({
            "canonical_key": row["canonical_key"],
            "source_host": row["host"],
            "test_only_role": row["method"] in TEST_ONLY,
            "required_file_kinds": list(expected),
            "execution_provenance": (
                "archived_per_case_execution_json" if has_execution
                else "archived_native_status_and_ID_sidecar_plus_accepted_shared_report"
            ),
            "files": files,
        })
    proof_fields = (
        "name", "bytes", "sha256", "source_md5", "provider_size", "provider_md5",
        "provider_md5_field", "netdisk_path", "verified_at_utc",
    )
    output = {
        "schema": "iraod-frozen-selected-native-file-coverage-v1",
        "status": "PROVEN" if not missing and len(results) == 671 else "INSUFFICIENT_EVIDENCE",
        "selected_frozen_valid_keys": 671,
        "proven_keys": len(results),
        "test_only_proven_keys": sum(row["test_only_role"] for row in results),
        "per_case_execution_json_keys": execution_json_keys,
        "missing": missing,
        "metadata_bytes_read": metadata_bytes,
        "payloads_read_rehashed_or_retransferred": 0,
        "index_identity_basis": "Exact bytes/SHA256 from the published frozen artifact manifest.",
        "legacy_schema_boundary": (
            "Every selected key has its declared metric, status, class AP, prediction pickle "
            "and ID sidecar mapped to archive members. A per-case execution.json is listed "
            "when it existed; otherwise the native status/sidecar and accepted shared report "
            "are the original provenance, not an invented missing-file requirement."
        ),
        "archive_assets": [
            {field: assets[name][field] for field in proof_fields} for name in sorted(used_assets)
        ],
        "records": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key not in ("records", "archive_assets")}))
    require(output["status"] == "PROVEN", "Selected frozen native file coverage is incomplete")
    require(output["test_only_proven_keys"] == 75, "Expected exactly 75 frozen TEST-only roles")


if __name__ == "__main__":
    main()
