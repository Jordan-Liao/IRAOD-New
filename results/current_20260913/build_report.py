"""Rebuild current descriptive tables from frozen rows and native metadata only."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "completed_snapshot_20260912T014556Z"
sys.path.insert(0, str(BASE))
from validate_artifacts import key, load_csv, require  # noqa: E402

CUTOFF = "2026-09-13T16:59:00+00:00"
VALID = "valid_native_test"


def canonical(row):
    return "/".join(str(row[k]) for k in ("dataset", "domain", "seed", "method", "role"))


def csv_text(rows):
    import csv
    import io
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def mean(values):
    return statistics.mean(values) if values else ""


def std(values):
    return statistics.stdev(values) if len(values) > 1 else ""


def budget(method):
    if method == "A":
        return "fixed_source_reference"
    if method == "SFYOLO":
        return "EXTENDED_2detector_epochs_plus_TAM160k"
    if method.startswith("LoRA-"):
        return "SUPERVISED_ORACLE_1detector_epoch_plus_LoRA10epochs"
    if method == "AASFOD":
        return "1detector_epoch_plus_TSD_disclosed_port"
    return "common_1detector_epoch"


def assemble():
    frozen = load_csv(BASE, "raw_metrics.csv")
    statuses = load_csv(ROOT, "test_completion_status.csv")
    require(len(frozen) == len({key(r) for r in frozen}) == 912, "Frozen canonical keys")
    require(len(statuses) == len({r["canonical_key"] for r in statuses}) == 912, "Current canonical keys")
    status = {r["canonical_key"]: r for r in statuses}
    require(set(status) == {canonical(r) for r in frozen}, "Current/frozen key set equality")
    counts = Counter(r["current_status"] for r in statuses)
    require(counts == {VALID: 900, "invalid_native_ema_nonfinite": 5,
                       "held_nonfinite_student": 5, "held_nonfinite_checkpoint": 2},
            "900 eligible + 7 held + 5 invalid = 912")
    require(all((r["eligible"] == "true") == (r["current_status"] == VALID) for r in statuses),
            "Eligibility and current status agree")
    payload = json.loads((ROOT / "native_metadata.json").read_text())
    records = payload["records"]
    native = {r["canonical_key"]: r for r in records}
    required = {canonical(r) for r in frozen if not r["native_mAP50"]
                and status[canonical(r)]["eligible"] == "true"}
    require(len(records) == len(native) == 229 and set(native) == required,
            "Exactly 229 missing eligible native records; no retries or frozen replacements")
    sources = {(r["dataset"], r["domain"]): r for r in frozen if r["method"] == "A"}
    classes = load_csv(BASE, "per_class.csv")
    raw = []
    for original in frozen:
        row = dict(original)
        identity = canonical(row)
        current = status[identity]
        row.update(canonical_key=identity, pr21_status=row["status"],
                   status=current["current_status"], eligible=current["eligible"],
                   budget_detail=budget(row["method"]), report_cutoff_utc=CUTOFF,
                   metric_origin="pr21_frozen_accepted" if row["native_mAP50"] else "not_available")
        if identity in native:
            record = native[identity]
            require(not record["errors"], f"Native metadata unavailable: {identity}: {record['errors']}")
            execution = record["execution"]
            require(canonical(execution) == identity, f"Native role/case mismatch: {identity}")
            source = sources[(row["dataset"], row["domain"])]
            require(execution["source_id"] == row["source_id"] == source["source_id"],
                    f"Source identity mismatch: {identity}")
            require(str(execution["source_seed"]) == row["source_seed"] == "42",
                    f"Source seed mismatch: {identity}")
            checkpoint = execution["checkpoint"]
            iteration = {"RSAR": 531, "DIOR": 369}[row["dataset"]] if row["method"] == "SFYOLO" else {"RSAR": 266, "DIOR": 185}[row["dataset"]]
            suffix = "_ema" if row["role"] == "ema" else ""
            require(Path(checkpoint).name == f"iter_{iteration}{suffix}.pth",
                    f"Prescribed final checkpoint/role: {identity}")
            sentinel = record["eval_status"].split()
            require(sentinel[0] == "eval_exit=0", f"Native eval failure: {identity}")
            tokens = dict(t.split("=", 1) for t in sentinel if "=" in t)
            require(tokens["checkpoint"] == checkpoint and tokens["role"] == row["role"]
                    and tokens["name"] == row["method"] and tokens["domain"] == row["domain"]
                    and tokens["seed"] == row["seed"], f"Native sentinel binding: {identity}")
            require(datetime.fromisoformat(sentinel[1]) <= datetime.fromisoformat(CUTOFF)
                    and datetime.fromisoformat(record["evaluated_at"]) <= datetime.fromisoformat(CUTOFF),
                    f"Metric after cutoff: {identity}")
            count, expected = record["pred_count"].split()
            expected_images = {"RSAR": 8538, "DIOR": 11738}[row["dataset"]]
            require(int(count) == int(expected.removeprefix("expect=")) == expected_images,
                    f"Native TEST image count: {identity}")
            metric = record["metric_json"]["metric"]["mAP"]
            require(isinstance(metric, (int, float)) and 0 <= metric <= 1,
                    f"Nonfinite or invalid native mAP: {identity}")
            require(record["metric_json"]["config"] == execution["eval_config"],
                    f"Native evaluator config mismatch: {identity}")
            eval_dir = Path(record["eval_dir"])
            row.update(
                native_mAP50=metric, mAP50=metric, mAP50_percent=100 * metric,
                delta_vs_source_pp=100 * (metric - float(source["mAP50"])),
                relative_vs_source_percent=100 * (metric / float(source["mAP50"]) - 1),
                host=record["host"], checkpoint=checkpoint, eval_json=record["eval_json"],
                eval_status=str(eval_dir / "eval_status"),
                prediction_image_ids=str(eval_dir / "predictions.pkl.image_ids.json"),
                class_ap=str(eval_dir / "class_ap.txt"), n_images=count,
                evaluated_at=sentinel[1], collected_at=payload["collected_utc"],
                training_code_sha=execution["training_code_sha"],
                evaluation_code_sha=execution["evaluation_code_sha"], config=execution["eval_config"],
                evidence_basis=f".{record['host']}:{record['execution_json']}; native_metadata.json",
                metric_origin="post_PR21_native_metadata",
                note="Native JSON mAP and printed class AP retained; recorded execution SHA; predictions not reopened in this report; post-NMS total not recollected.",
            )
            class_names = []
            for line in record["class_ap_text"].splitlines():
                fields = [f.strip() for f in line.strip().strip("|").split("|")]
                if len(fields) == 5 and fields[1].isdigit() and fields[2].isdigit():
                    class_name, value = fields[0], fields[4]
                    require(0 <= float(value) <= 1, f"Invalid class AP: {identity}/{class_name}")
                    class_names.append(class_name)
                    classes.append({
                        **{k: row[k] for k in ("dataset", "domain", "method", "seed", "role")},
                        "class_name": class_name, "native_AP50": value, "AP50": value,
                        "AP50_percent": 100 * float(value), "status": VALID,
                        "precision": "printed class_ap.txt (typically 3 decimals)",
                        "evidence": f".{record['host']}:{eval_dir / 'class_ap.txt'}",
                    })
            expected_names = {r["class_name"] for r in classes if r["dataset"] == row["dataset"] and r["method"] == "A"}
            require(len(class_names) == len(set(class_names)) and set(class_names) == expected_names,
                    f"Native class coverage: {identity}")
        if original["native_mAP50"]:
            for field in ("native_mAP50", "mAP50", "mAP50_percent", "checkpoint", "eval_json", "source_id"):
                require(row[field] == original[field], f"Frozen value changed: {identity}/{field}")
        require((row["mAP50"] != "") == (row["status"] == VALID), f"Metric eligibility: {identity}")
        raw.append(row)
    for row in classes:
        row["status"] = status[canonical(row)]["current_status"]
    require(len(classes) == len({key(r) + (r["class_name"],) for r in classes}), "Unique class rows")
    require(sum(r["mAP50"] != "" for r in raw) == 900, "900 actual eligible metrics")
    require(sum(r["native_mAP50"] != "" for r in raw) == 905, "905 native including five invalid")
    require(all(r["mAP50"] == "" for r in raw if r["eligible"] == "false"), "No invalid zero filling")
    return raw, classes


def aggregate(raw):
    groups, seed_groups, domain_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in raw:
        group = (row["dataset"], row["method"], row["role"])
        groups[group].append(row)
        seed_groups[group + (row["seed"],)].append(row)
        domain_groups[group + (row["domain"],)].append(row)
    per_seed = []
    for group, rows in sorted(seed_groups.items()):
        valid = [r for r in rows if r["mAP50"] != ""]
        complete = len(valid) == len(rows)
        per_seed.append({
            **dict(zip(("dataset", "method", "role", "seed"), group)),
            "budget_detail": budget(group[1]), "valid_domains": len(valid), "expected_domains": len(rows),
            "complete": complete, "missing_or_invalid_domains": ";".join(r["domain"] for r in rows if r["mAP50"] == ""),
            "mean_mAP50_percent": mean([float(r["mAP50_percent"]) for r in valid]) if complete else "",
            "mean_delta_vs_source_pp": mean([float(r["delta_vs_source_pp"]) for r in valid]) if complete else "",
        })
    coverage, summary = [], []
    for group, rows in sorted(groups.items()):
        valid = [r for r in rows if r["mAP50"] != ""]
        complete = [r for r in per_seed if (r["dataset"], r["method"], r["role"]) == group and r["complete"]]
        values = [r["mean_mAP50_percent"] for r in complete]
        shared = dict(zip(("dataset", "method", "role"), group))
        coverage.append({
            **shared, "expected_cells": len(rows), "eligible_cells": sum(r["eligible"] == "true" for r in rows),
            "metric_covered_cells": len(valid), "eligible_missing_metrics": sum(r["eligible"] == "true" and r["mAP50"] == "" for r in rows),
            "held_cells": sum(r["status"].startswith("held_") for r in rows),
            "invalid_native_cells": sum(r["status"] == "invalid_native_ema_nonfinite" for r in rows),
            "frozen_eligible_cells": sum(r["metric_origin"] == "pr21_frozen_accepted" and r["mAP50"] != "" for r in rows),
            "new_native_cells": sum(r["metric_origin"] == "post_PR21_native_metadata" for r in rows),
            "complete_seeds": ";".join(r["seed"] for r in complete),
            "expected_seeds": len({r["seed"] for r in rows}), "report_cutoff_utc": CUTOFF,
        })
        summary.append({
            **shared, "budget_detail": budget(group[1]), "n_complete_seeds": len(complete),
            "expected_seeds": len({r["seed"] for r in rows}), "seeds": ";".join(r["seed"] for r in complete),
            "mean_mAP50_percent": mean(values), "sample_std_percent": std(values),
            "mean_delta_vs_source_pp": mean([r["mean_delta_vs_source_pp"] for r in complete]),
            "valid_cells": len(valid), "expected_cells": len(rows),
            "domains_per_complete_seed": {"RSAR": 8, "DIOR": 4}[group[0]],
        })
    per_domain = []
    for group, rows in sorted(domain_groups.items()):
        valid = [r for r in rows if r["mAP50"] != ""]
        values = [float(r["mAP50_percent"]) for r in valid]
        per_domain.append({
            **dict(zip(("dataset", "method", "role", "domain"), group)),
            "budget_detail": budget(group[1]), "valid_seeds": len(valid), "expected_seeds": len(rows),
            "seeds": ";".join(r["seed"] for r in valid),
            "mean_mAP50_percent": mean(values), "sample_std_percent": std(values),
            "mean_delta_vs_source_pp": mean([float(r["delta_vs_source_pp"]) for r in valid]),
        })
    return per_seed, per_domain, summary, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare all committed outputs to recomputation")
    args = parser.parse_args()
    raw, classes = assemble()
    seeds, domains, summary, coverage = aggregate(raw)
    progress = json.loads((ROOT / "current_report_summary.json").read_text())
    require(progress["test_metrics"]["eligible_metrics_integrated"] ==
            sum(r["metric_covered_cells"] for r in coverage) == 900,
            "Current summary and coverage agree")
    roi, archive = progress["roi"], progress["archival"]
    require(roi["completed_unique_keys"] + roi["pending_unique_keys"] == roi["eligible_target"] == 720
            and sum(roi["pending_families"].values()) == roi["pending_unique_keys"] == 343
            and roi["eligible_target"] + roi["held_outside_eligible_target"] == 732,
            "ROI canonical denominators and pending families")
    require(archive["delivered_unique_roi_keys"] == 286
            and archive["delivered_unique_roi_keys"] + archive["completed_not_delivered"] ==
            archive["completed_roi_denominator"] == roi["completed_unique_keys"] == 377
            and archive["batch8_counted_as_delivered"] == 0
            and archive["verified_archives"] == 247 and archive["verified_bytes"] == 167044220165,
            "Accepted archival key/file/assigned separation")
    require(sum(progress["training"]["family_cells"].values()) ==
            progress["training"]["prescribed_budget_complete"] == 540
            and progress["training"]["models_repaired"] == 0,
            "Prescribed TRAIN budgets do not imply numerical repair")
    lineage = json.loads((ROOT / "provenance.json").read_text())["new_native_evidence"]
    require(sum(lineage[k]["new_keys"] for k in ("AASFOD_ema", "AASFOD_student_SFYOLO",
                                               "B_REG_F_variants", "Oracle")) == 229,
            "New native source lineage totals")
    tables = dict(raw_metrics=raw, per_class=classes, per_seed=seeds, per_domain=domains,
                  summary=summary, coverage=coverage)
    for name, rows in tables.items():
        path, content = ROOT / f"{name}.csv", csv_text(rows)
        if args.check:
            require(path.read_text() == content, f"Stale or modified generated output: {path.name}")
        else:
            path.write_text(content)
    print(json.dumps({
        "status": "PASS", "canonical_keys": len(raw), "eligible_metric_keys": 900,
        "eligible_missing_metrics": 0, "held_keys": 7, "invalid_native_ema_keys": 5,
        "frozen_native_preserved": 676, "new_native_bound": 229,
        "class_rows": len(classes), "summary_rows": len(summary),
        "cutoff_utc": CUTOFF, "mode": "check" if args.check else "build",
    }))


if __name__ == "__main__":
    main()
