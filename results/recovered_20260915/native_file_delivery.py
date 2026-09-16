"""Validate selected native TEST files independently of ROI archive coverage."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import build_completed_report as report

ROOT = Path(__file__).resolve().parent
require = report.previous.require
GROUP_ORIGINS = {
    "frozen671": ("pr21_frozen_accepted", 671),
    "corrected12": (report.ORIGIN, 12),
    "post_pr21_229": ("post_PR21_native_metadata", 229),
}


def read(path):
    return json.loads(Path(path).read_text())


def required_paths(row):
    sidecar = row["prediction_image_ids"]
    require(sidecar.endswith(".image_ids.json"), "Unexpected native ID-sidecar path")
    directory = Path(row["eval_json"]).parent
    return {
        "predictions": sidecar.removesuffix(".image_ids.json"),
        "image_id_sidecar": sidecar,
        "native_metric": row["eval_json"],
        "class_ap": row["class_ap"],
        "execution": str(directory / "execution.json"),
        "eval_status": str(directory / "eval_status"),
    }


def receipt_valid(receipt):
    require(
        receipt["provider_size"] == receipt["bytes"]
        and receipt["provider_md5"] == receipt["source_md5"]
        and Path(receipt["netdisk_path"]).name == receipt["name"],
        f"Native archive provider mismatch: {receipt['name']}",
    )


def validate(payload):
    raw = report.previous.load_csv(ROOT, "raw_metrics.csv")
    by_key = {report.previous.canonical(row): row for row in raw}
    require(payload["status"] == "PROVEN", "Native-file backup is not complete")
    require(
        payload["counts"] == {
            "frozen": 671, "corrected": 12, "post_pr21": 229, "total": 912,
            "missing": 0, "unknown": 0, "pending": 0, "active": 0,
        },
        "Native-file backup has a coverage or live-work gap",
    )
    union = set()
    for group, (origin, count) in GROUP_ORIGINS.items():
        rows = payload[group]["records"]
        keys = {row["canonical_key"] for row in rows}
        require(
            len(rows) == len(keys) == count
            and keys == {key for key, row in by_key.items() if row["metric_origin"] == origin}
            and union.isdisjoint(keys),
            f"Selected native-key coverage differs: {group}",
        )
        union.update(keys)
    require(union == set(by_key) and len(union) == 912, "Incomplete selected-native union")
    assets = {asset["name"]: asset for asset in payload["frozen671"]["archive_assets"]}
    for asset in assets.values():
        receipt_valid(asset)
    for item in payload["frozen671"]["records"]:
        row = by_key[item["canonical_key"]]
        files = {entry["source_path"]: entry for entry in item["files"]}
        paths = required_paths(row)
        paths["eval_status"] = row["eval_status"]
        for kind, path in paths.items():
            if kind == "execution":
                continue
            require(path in files, f"Missing frozen selected file: {item['canonical_key']}/{kind}")
            entry = files[path]
            require(
                entry["asset_name"] in assets and bool(entry["archive_member"])
                and len(entry["sha256"]) == 64 and item["source_host"] == row["host"],
                f"Unbound frozen archive member: {path}",
            )
    prior = read(ROOT / "delivery_manifest.json")
    corrected_receipts = {
        row["netdisk_path"]: row for row in prior["corrected_test_roi12"]["provider_receipts"]
    }
    for item in payload["corrected12"]["records"]:
        row = by_key[item["canonical_key"]]
        require(
            item["selected_checkpoint"] == row["checkpoint"]
            and item["required_source_files"] == required_paths(row)
            and Path(row["eval_json"]).parent.parent == Path(item["source_root"]),
            f"Corrected whole-case native binding differs: {item['canonical_key']}",
        )
        receipt = corrected_receipts[item["netdisk_path"]]
        receipt_valid(receipt)
        require(
            item["archive"] == receipt["name"]
            and item["provider_size"] == receipt["bytes"]
            and item["provider_md5"] == receipt["source_md5"],
            "Corrected native archive receipt differs from accepted ROI delivery",
        )
    post = payload["post_pr21_229"]
    receipts = {row["netdisk_path"]: row for row in post["provider_receipts"]}
    require(len(post["provider_receipts"]) == len(receipts) == 229, "Duplicate native receipts")
    member_checks = 0
    for item in post["records"]:
        row = by_key[item["canonical_key"]]
        require(
            item["selected_checkpoint"] == item["execution_checkpoint"] == row["checkpoint"]
            and Path(item["source_eval_dir"]) == Path(row["eval_json"]).parent,
            f"Selected post-PR21 execution differs: {item['canonical_key']}",
        )
        members = {member["kind"]: member for member in item["required_members"]}
        paths = required_paths(row)
        require(len(item["required_members"]) == 6 and members.keys() == paths.keys(),
                f"Missing native member kinds: {item['canonical_key']}")
        for kind, path in paths.items():
            member = members[kind]
            require(
                member["source_path"] == path
                and member["source_stat_bytes"] == member["stored_bytes"]
                and member["member"] == f"{item['canonical_key']}/{Path(path).name}"
                and member["archive"] == item["archive"]
                and member["regular_file"] is True
                and member["symlink"] is False and member["hardlink"] is False
                and member["match"] is True,
                f"Native TAR member does not back the selected file: {item['canonical_key']}/{kind}",
            )
            member_checks += 1
        receipt = receipts[item["netdisk_path"]]
        receipt_valid(receipt)
        require(
            receipt["name"] == item["archive"] and receipt["bytes"] == item["provider_size"]
            and receipt["source_md5"] == item["provider_md5"]
            and receipt["canonical_key"] == item["canonical_key"],
            f"Native member archive/provider association differs: {item['canonical_key']}",
        )
    new_bytes = sum(row["bytes"] for row in receipts.values())
    native_paths = [row["netdisk_path"] for row in assets.values()]
    native_paths += list(corrected_receipts) + list(receipts)
    require(len(native_paths) == len(set(native_paths)) == 252, "Native container double-counting")
    native_bytes = sum(row["bytes"] for row in assets.values())
    native_bytes += sum(row["bytes"] for row in corrected_receipts.values()) + new_bytes
    require(
        member_checks == 1374 and new_bytes == 3765176320
        and payload["provider_totals"] == {"files": 252, "bytes": native_bytes}
        and native_bytes == 14270551245,
        "Native backup byte/container/member totals differ",
    )
    require(
        payload["combined_delivery_totals"]["provider_files"]
        == prior["totals"]["provider_files"] + len(receipts) == 928
        and payload["combined_delivery_totals"]["bytes"]
        == prior["totals"]["bytes"] + new_bytes == 340723262725,
        "Overlapping native containers were double-counted in overall delivery",
    )
    return {
        "status": "PASS", "native_test_keys": 912, "missing": 0, "pending": 0, "active": 0,
        "post_pr21_member_checks": member_checks,
        "supplemental_provider_files": 229, "supplemental_bytes": new_bytes,
        "native_relevant_provider_files": 252, "native_relevant_bytes": native_bytes,
        "all_delivery_provider_files": prior["totals"]["provider_files"] + len(receipts),
        "all_delivery_bytes": prior["totals"]["bytes"] + new_bytes,
    }


def import_mapping(mapping_path, receipts_path):
    payload = deepcopy(read(mapping_path))
    for group in GROUP_ORIGINS:
        payload[group].pop("proof", None)
    for item in payload["corrected12"]["records"]:
        item.pop("proof_reference", None)
    receipt_index = read(receipts_path)
    require(
        receipt_index["status"] == "COMPLETE" and receipt_index["count"] == 229
        and receipt_index["source_provider_size_matches"] == 229
        and receipt_index["source_provider_md5_matches"] == 229
        and receipt_index["duplicate_names"] == receipt_index["duplicate_keys"] == 0,
        "Supplemental source/provider receipt index is incomplete",
    )
    fields = (
        "canonical_key", "partition", "name", "source_md5", "source_sha256", "provider_size",
        "provider_md5", "netdisk_path", "verified_at_utc", "disposition",
    )
    payload["post_pr21_229"]["provider_receipts"] = [
        {"bytes": row["source_inventory_bytes"],
         **{key: row[key] for key in fields if key in row}}
        for row in receipt_index["records"]
    ]
    payload["scope"] = (
        "Selected native TEST output files. Original model/adapter weights and raw datasets "
        "are outside this backup scope; six training-repair archives were delivered separately."
    )
    prior = read(ROOT / "delivery_manifest.json")
    payload["combined_delivery_totals"] = {
        "provider_files": prior["totals"]["provider_files"] + receipt_index["count"],
        "bytes": prior["totals"]["bytes"] + receipt_index["bytes"],
        "rule": "Add only the 229 new archives to the earlier 699; native252 containers overlap.",
    }
    result = validate(payload)
    (ROOT / "native_test_file_delivery.json").write_text(json.dumps(payload, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--provider-receipts", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = validate(read(ROOT / "native_test_file_delivery.json"))
    else:
        if args.mapping is None or args.provider_receipts is None:
            parser.error("Import requires --mapping and --provider-receipts")
        result = import_mapping(args.mapping, args.provider_receipts)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
