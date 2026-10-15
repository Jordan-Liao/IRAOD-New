"""Collect and append the 180 missing seed42 non-chaff B-F AP values."""

import argparse
import csv
from pathlib import Path

from experiments.comparison.collect_report_manifest import choose_evaluation, load_resolver
from experiments.comparison.report_inputs import CLASSES, class_table
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/paper_comparison"
FIELDS = ("dataset", "corruption", "method", "ckpt_role", "class_name", "AP50", "mAP50")
CORRUPTIONS = tuple(d for d in DOMAINS["RSAR"] if d not in ("clean", "chaff"))


def csv_rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def canonical_maps(raw_path):
    return {(r["corruption"], r["method"]): r["mAP50"] for r in csv_rows(raw_path)
            if r["dataset"] == "RSAR" and r["seed"] == "42" and r["ckpt_role"] == "ema"}


def collect_ap(paths, owner_format, raw_path, output):
    canonical = canonical_maps(raw_path)
    records = []
    for domain in CORRUPTIONS:
        for method in "BCDEF":
            directory, metric_file, _, issue = choose_evaluation(
                paths, owner_format, "RSAR", domain, 42, method)
            if issue:
                raise ValueError(f"{domain}/{method}: {issue}; no AP is synthesized")
            metric = read_json(metric_file)["metric"]["mAP"]
            fixed_map = canonical[domain, method]
            if float(metric) != float(fixed_map):
                raise ValueError(f"{domain}/{method}: evaluation mAP differs from fixed seed42 raw value")
            table = directory / "class_ap.txt"
            classes = class_table(table, "RSAR")
            records.append({
                "dataset": "RSAR", "corruption": domain, "method": method, "seed": 42,
                "ckpt_role": "ema", "class_ap_file": str(table), "eval_json": str(metric_file),
                "observed_mAP50": metric, "canonical_mAP50": fixed_map,
                "class_AP_precision": "printed class_ap.txt, three decimals",
                "classes": [{"class_name": r["class_name"], "AP50": f"{r['AP50']:.3f}"}
                            for r in classes],
            })
    evidence = {
        "schema": "iraod-rsar-nonchaff-perclass-evidence-v1", "seed": 42,
        "groups": len(records), "class_rows": sum(len(r["classes"]) for r in records),
        "mAP_source": "results/paper_comparison/raw_results.csv",
        "records": records,
    }
    write_json(output, evidence)
    return evidence


def append_ap(evidence_path, table_path, raw_path):
    evidence = read_json(evidence_path)
    if evidence["schema"] != "iraod-rsar-nonchaff-perclass-evidence-v1":
        raise ValueError("Expected the collected non-chaff per-class evidence")
    canonical = canonical_maps(raw_path)
    wanted = {(d, m) for d in CORRUPTIONS for m in "BCDEF"}
    records = evidence["records"]
    if len(records) != 30 or {(r["corruption"], r["method"]) for r in records} != wanted:
        raise ValueError("All 30 non-chaff B-F groups are required before appending")
    additions = []
    for record in records:
        domain, method = record["corruption"], record["method"]
        if (record["dataset"] != "RSAR" or record["seed"] != 42 or record["ckpt_role"] != "ema"
                or float(record["observed_mAP50"]) != float(canonical[domain, method])):
            raise ValueError("Evidence identity/mAP differs from the frozen seed42 result")
        if [r["class_name"] for r in record["classes"]] != list(CLASSES["RSAR"]):
            raise ValueError("Missing or reordered RSAR class AP rows")
        for item in record["classes"]:
            value = float(item["AP50"])
            if not 0 <= value <= 1 or item["AP50"] != f"{value:.3f}":
                raise ValueError("Class AP must preserve the stated three-decimal precision")
            additions.append({
                "dataset": "RSAR", "corruption": domain, "method": method, "ckpt_role": "ema",
                "class_name": item["class_name"], "AP50": item["AP50"],
                "mAP50": canonical[domain, method],
            })
    existing = csv_rows(table_path)
    key_fields = FIELDS[:5]
    by_key = {tuple(r[k] for k in key_fields): r for r in existing}
    if len(by_key) != len(existing):
        raise ValueError("Existing per-class table contains duplicate rows")
    pending = []
    for row in additions:
        old = by_key.get(tuple(row[k] for k in key_fields))
        if old is None:
            pending.append(row)
        elif float(old["AP50"]) != float(row["AP50"]) or old["mAP50"] != row["mAP50"]:
            raise ValueError("Refusing to overwrite an existing per-class value")
    # Append only after the entire evidence batch passes; preserve all original bytes.
    with Path(table_path).open("a", newline="") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writerows(pending)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--paths-module", required=True)
    collect_parser.add_argument("--owner-format", required=True)
    collect_parser.add_argument("--out", required=True)
    apply_parser = commands.add_parser("append")
    apply_parser.add_argument("--evidence", required=True)
    apply_parser.add_argument("--table", default=str(RESULTS / "per_class_summary.csv"))
    for sub in (collect_parser, apply_parser):
        sub.add_argument("--raw", default=str(RESULTS / "raw_results.csv"))
    args = parser.parse_args()
    if args.command == "collect":
        result = collect_ap(load_resolver(args.paths_module), read_json(args.owner_format),
                            args.raw, args.out)
        print(f"Collected {result['groups']} groups / {result['class_rows']} class AP rows")
    else:
        print(f"Appended {append_ap(args.evidence, args.table, args.raw)} rows")


if __name__ == "__main__":
    main()
