"""Validate this frozen snapshot and its private ByPy file/asset manifests."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import json
import math
from pathlib import Path
import re

CUTOFF = datetime.fromisoformat("2026-09-12T01:45:56+00:00")
KEY = ("dataset", "domain", "method", "seed", "role")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_csv(root, name):
    with (root / name).open() as handle:
        return list(csv.DictReader(handle))


def key(row):
    return tuple(row[field] for field in KEY)


def validate_bypy_assets(manifest, listing):
    remote_root = "/apps/bypy/IRAOD-New/results/completed_snapshot_20260912T014556Z"
    require(manifest["storage"]["netdisk_path"] == remote_root, "ByPy snapshot destination")
    require(manifest["snapshot_cutoff_utc"] == "2026-09-12T01:45:56Z", "Asset cutoff")
    assets = manifest["assets"]
    require(len(assets) == manifest["asset_count"] > 0, "Asset count")
    require(len({asset["name"] for asset in assets}) == len(assets), "Duplicate asset names")
    live = {row["path"]: row for row in listing}
    require(len(live) == len(listing), "Duplicate ByPy listing paths")
    require(set(live) == {remote_root + "/" + asset["name"] for asset in assets},
            "ByPy project listing must match the complete asset inventory")
    require(sum(asset["bytes"] for asset in assets) == manifest["uploaded_asset_bytes"],
            "Uploaded byte total")
    for asset in assets:
        path = remote_root + "/" + asset["name"]
        require(asset["netdisk_path"] == path, f"Asset destination: {asset['name']}")
        require(re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]), f"Source SHA256: {asset['name']}")
        require(re.fullmatch(r"[0-9a-f]{32}", asset["source_md5"]), f"Source MD5: {asset['name']}")
        remote = live[path]
        require(remote["size"] == asset["provider_size"] == asset["bytes"],
                f"ByPy size mismatch: {asset['name']}")
        require(remote["block_list"] == [asset["provider_md5"]] == [asset["source_md5"]],
                f"ByPy MD5 mismatch: {asset['name']}")
    return len(assets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path(__file__).parent)
    parser.add_argument("--file-manifests", type=Path, nargs="*")
    parser.add_argument("--source-host", choices=("67", "221"))
    parser.add_argument("--bypy-assets", type=Path,
                        help="Actual ByPy project listing/meta JSON with path, size and block_list")
    args = parser.parse_args()
    root = args.snapshot
    raw = load_csv(root, "raw_metrics.csv")
    classes = load_csv(root, "per_class.csv")
    roi = load_csv(root, "roi_coverage.csv")
    provenance = json.loads((root / "provenance.json").read_text())
    require(len(raw) == len({key(row) for row in raw}) == 912, "TEST key coverage")
    native = [row for row in raw if row["native_mAP50"]]
    require(len(native) == 676, "Native record count")
    require(sum(row["status"] == "valid_native_test" for row in raw) == 671, "Eligible count")
    require(len(classes) == 7304, "Per-class coverage")
    require(provenance["qualitative"]["approved_roi_expected_image_roles"] ==
            sum(int(row["expected_images"]) for row in roi) == 7030616, "Approved ROI scope")
    require(provenance["qualitative"]["accepted_core_complete_image_roles"] ==
            sum(int(row["completed_images"] or 0) for row in roi) == 1267816, "Core ROI scope")
    evidence = json.loads((root / "ema_numerical_evidence.json").read_text())["checkpoints"]
    invalid = {row["checkpoint"] for row in raw if row["status"] == "native_zero_known_nan_model"}
    require({item["path"] for item in evidence} == invalid and len(invalid) == 5, "EMA evidence binding")
    require(all(item["states"]["state_dict"]["nan_element_count"] > 0 for item in evidence),
            "EMA exclusions require actual nonfinite model state")

    source_checked = 0
    if args.source_host:
        class_by_cell = defaultdict(dict)
        for row in classes:
            class_by_cell[key(row)][row["class_name"]] = float(row["native_AP50"])
        for row in native:
            if row["host"] != args.source_host:
                continue
            path = Path(row["eval_json"])
            metric = json.loads(path.read_text())["metric"]["mAP"]
            require(math.isclose(metric, float(row["native_mAP50"]), abs_tol=1e-12),
                    f"Native mAP transcription: {path}")
            actual_classes = {}
            for line in Path(row["class_ap"]).read_text().splitlines():
                fields = [field.strip() for field in line.strip().strip("|").split("|")]
                if len(fields) == 5 and fields[1].isdigit() and fields[2].isdigit():
                    actual_classes[fields[0]] = float(fields[4])
            require(actual_classes == class_by_cell[key(row)], f"Native class AP: {path}")
            count = (path.parent / "pred_count.txt").read_text().split()[0]
            require(int(count) == int(row["n_images"]), f"Native image count: {path}")
            require(datetime.fromisoformat(row["evaluated_at"]) <= CUTOFF, f"Late metric: {path}")
            source_checked += 1
        require(source_checked == {"67": 480, "221": 196}[args.source_host], "Host native coverage")

    manifest_records = 0
    if args.file_manifests:
        wanted = {(row["host"], row[field]) for row in native
                  for field in ("eval_json", "eval_status", "prediction_image_ids", "class_ap")}
        wanted |= {(row["host"], str(Path(row["eval_json"]).parent / name))
                   for row in native for name in ("predictions.pkl", "pred_count.txt")}
        seen = set()
        archive_members = set()
        categories = Counter()
        roi_files = Counter()
        notices = defaultdict(set)
        visuals = set()
        for path in args.file_manifests:
            with gzip.open(path, "rt") as handle:
                for line in handle:
                    row = json.loads(line)
                    manifest_records += 1
                    require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), f"File SHA256: {path}")
                    require(row["bytes"] >= 0, f"File size: {path}")
                    member = (row["asset_name"], row["archive_member"])
                    require(member not in archive_members, f"Duplicate archive member: {member}")
                    archive_members.add(member)
                    category = row["category"]
                    categories[category] += 1
                    if row.get("publication_auxiliary"):
                        notices[row["asset_name"]].add(Path(row["source_path"]).name)
                        continue
                    identity = (row["source_host"], row["source_path"])
                    require(identity not in seen, f"Duplicate physical result: {identity}")
                    seen.add(identity)
                    require(not row["after_cutoff"], f"Late file in frozen payload: {identity}")
                    require(Path(row["source_path"]).suffix not in (".pth", ".pt", ".ckpt"),
                            f"Model weights in result payload: {identity}")
                    if category == "roi_features" and row["source_path"].endswith(".npz"):
                        require(len(row["bindings"]) == 1, f"Ambiguous ROI group: {identity}")
                        roi_files[row["bindings"][0]] += 1
                    if category == "visualization_with_dataset_pixels":
                        visuals.add(row["asset_name"])
        require(wanted <= seen, f"Native result files absent: {sorted(wanted - seen)[:8]}")
        require(categories["predictions"] == 676, "Prediction payload count")
        require(categories["visualization_with_dataset_pixels"] == 3520, "PNG payload count")
        require(categories["embedding_data"] == 72 and categories["embedding_plot"] == 168,
                "24 completed embedding groups require 72 data files and 168 plots")
        for row in roi:
            if row["status"] == "accepted_core_complete":
                binding = "/".join(row[field] for field in ("dataset", "domain", "method", "role"))
                require(roi_files[binding] == int(row["completed_images"]), f"ROI files: {binding}")
        require(sum(roi_files.values()) == 1267816, "Full TEST ROI feature payload count")
        require(all(notices[asset] == {"RESULT_ASSET_NOTICE.md", "CC-BY-NC-4.0.txt"}
                    for asset in visuals), "Every image-bearing archive needs attribution and full license")

    assets_checked = 0
    if args.bypy_assets:
        manifest = json.loads((root / "artifact_manifest.json").read_text())
        assets_checked = validate_bypy_assets(manifest, json.loads(args.bypy_assets.read_text()))
    print(json.dumps({"status": "PASS", "native_records_checked": source_checked,
                      "manifest_records_checked": manifest_records,
                      "bypy_assets_checked": assets_checked,
                      "snapshot_cutoff_utc": CUTOFF.isoformat()}))


if __name__ == "__main__":
    main()
