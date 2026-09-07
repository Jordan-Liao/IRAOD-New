"""Collect full qualitative evidence independently of pending quantitative runs."""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from experiments.comparison import report_qualitative
from experiments.comparison.final_report import write_csv
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


ROOT = Path(__file__).resolve().parents[2]


def collect_completion(plan_path, embedding_root, out_dir, expected_detections=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    source_dir = out / "collector_sources"
    source_dir.mkdir()
    for source in (Path(__file__), Path(report_qualitative.__file__)):
        shutil.copyfile(source, source_dir / source.name)
    fragment = {
        "qualitative_plan": str(Path(plan_path).resolve()),
        "embeddings": [
            {"dataset": ds, "domain": domain, "comparison": role,
             "directory": str(Path(embedding_root) / ds / domain / role)}
            for ds, domains in DOMAINS.items() for domain in domains
            for role in ("ema", "student")],
    }
    write_json(out / "report_input_fragment.json", fragment)
    count = 0
    writer = None
    temporary = out / "roi_vis_coverage.csv.partial"
    with temporary.open("w", newline="") as stream:
        def sink(row):
            nonlocal count, writer
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            count += 1
            if count % 25000 == 0:
                print(f"validated_roi_image_roles={count}", flush=True)

        _, embeddings, coverage = report_qualitative.qualitative_evidence(fragment, sink)
    if expected_detections is not None and coverage["roi_detection_rows"] != expected_detections:
        raise ValueError(
            f"Actual detection rows {coverage['roi_detection_rows']} != expected {expected_detections}")
    temporary.replace(out / "roi_vis_coverage.csv")
    write_csv(out / "roi_group_summary.csv", coverage["roi_group_index"], ("run_id", "status"))
    write_csv(out / "embedding_index.csv", embeddings, ("dataset", "domain", "comparison", "status"))
    complete = (coverage["roi_complete_groups"] == 132
                and coverage["roi_complete"] == coverage["roi_expected_image_roles"]
                and coverage["vis_complete"] == 3520
                and coverage["embeddings_complete"] == 24)
    summary = {
        "schema": "iraod-qualitative-completion-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "seed42_full_test_roi_and_frozen_visualizations_and_joint_embeddings",
        "qualitative_status": "complete" if complete else "partial",
        "quantitative_status": "pending_separate_evidence",
        "all_eight_items_complete": False,
        "collector_base_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "collector_worktree_modified": bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
        "collector_source_snapshot": str(source_dir),
        "plan": fragment["qualitative_plan"],
        "evidence_root": str(out),
        "roi_image_manifest": str(out / "roi_vis_coverage.csv"),
        "roi_group_summary": str(out / "roi_group_summary.csv"),
        "embedding_index": str(out / "embedding_index.csv"),
        "report_input_fragment": str(out / "report_input_fragment.json"),
        **{k: v for k, v in coverage.items() if k != "roi_group_index"},
        "embedding_points": sum(e["n_points"] or 0 for e in embeddings),
        "note": "Detection count is summed from validated NPZ rows, not a manual estimate. "
                "Sampling and qualitative completion do not certify quantitative or all-item completion.",
    }
    write_json(out / "qualitative_summary.json", summary)
    print(f"{summary['qualitative_status']}: ROI={coverage['roi_complete']}, "
          f"detections={coverage['roi_detection_rows']}, vis={coverage['vis_complete']}, "
          f"embeddings={coverage['embeddings_complete']}; quantitative=pending", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--embedding-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-detections", type=int)
    args = parser.parse_args()
    collect_completion(args.plan, args.embedding_root, args.out_dir, args.expected_detections)


if __name__ == "__main__":
    main()
