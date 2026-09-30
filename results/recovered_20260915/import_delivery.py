"""Publish the accepted delivery metadata without rereading archived payloads."""

import argparse
import json
from pathlib import Path

import build_completed_report as report


def read(path):
    return json.loads(Path(path).read_text())


def receipts(rows):
    fields = (
        "name", "bytes", "source_md5", "source_sha256", "provider_size",
        "provider_md5", "netdisk_path", "verified_at_utc", "disposition",
    )
    return [{key: row[key] for key in fields if key in row} for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("three_set_terminal", type=Path)
    args = parser.parse_args()
    final = read(args.three_set_terminal)
    old = read(final["sets"]["canonical_roi720"]["receipt"])
    corrected = read(final["sets"]["corrected_test_roi12"]["receipt"])
    train = read(final["sets"]["train_repair6"]["receipt"])
    historical = old["partitions"]["historical377"]
    rollout = old["partitions"]["rollout342"]
    pilot = old["partitions"]["pilot1"]
    indexes = [read(path) for path in rollout["receipt_indexes"]]
    inherited = historical["receipt_references"]
    payload = {
        "schema": "iraod-final-delivery-v1",
        "status": final["status"],
        "completed_utc": final["completed_utc"],
        "verification": (
            "Accepted provider size and whole-file MD5 receipts. Historical377 evidence "
            "is inherited without payload reread, rehash or retransmission. Provider "
            "objects include historical companion assets; file counts are not ROI-key counts."
        ),
        "canonical_roi720": {
            "status": old["status"],
            "keys": old["membership"]["keys"],
            **old["provider_objects"],
            "historical377": {
                "roi_keys": historical["membership_keys"],
                "files": historical["provider_files"],
                "bytes": historical["provider_bytes"],
                "verification_basis": old["success_gate"],
                "frozen_public_proof": "../completed_snapshot_20260912T014556Z/artifact_manifest.json",
                "retained_operator_receipts": [
                    Path(inherited["pr21"]).name,
                    Path(inherited["batch1"]).name,
                    *[Path(path).name for path in inherited["batch2_to_15"]],
                ],
                "destinations": historical["destinations"],
            },
            "rollout342": {
                "files": rollout["provider_files"],
                "bytes": rollout["provider_bytes"],
                "destination_root": rollout["destination_root"],
                "provider_receipts": receipts([
                    row for index in indexes for row in index["receipts"]
                ]),
            },
            "pilot1": {
                "key": old["membership"]["pilot1"],
                "files": pilot["provider_files"],
                "bytes": pilot["provider_bytes"],
                "whole_md5": pilot["provider_md5"],
                "destination": pilot["destination"],
                "verification_basis": old["success_gate"],
            },
        },
        "corrected_test_roi12": {
            key: corrected[key] for key in ("status", "files", "bytes", "pending", "active")
        },
        "train_repair6": {
            "status": train["status"],
            "files": train["archives"],
            "bytes": train["bytes"],
            "cells": train["cells"],
            "provider_size_and_whole_md5_verified": train["provider_size_and_whole_md5_verified"],
            "destination_root": final["sets"]["train_repair6"]["destination_root"],
            "pending": final["sets"]["train_repair6"]["pending"],
            "active": final["sets"]["train_repair6"]["active"],
        },
        "totals": final["totals"],
    }
    payload["corrected_test_roi12"].update(
        keys=corrected["membership"],
        provider_receipts=receipts(corrected["provider_receipts"]),
        destinations=[row["destination"] for row in corrected["partitions"]],
    )
    report.validate_delivery(payload)
    destination = report.ROOT / "delivery_manifest.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "DELIVERY_IMPORTED", **final["totals"], "roi_keys": 732}))


if __name__ == "__main__":
    main()
