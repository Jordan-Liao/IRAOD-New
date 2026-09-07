"""Build an evidence-backed, versioned CPU comparison report; never run models."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

from experiments.comparison.report_inputs import collect_quantitative, historical_paths, read_rows
from experiments.comparison.report_qualitative import qualitative_evidence
from experiments.comparison.report_statistics import summarize
from experiments.comparison.result_completion import read_json, write_json


SCHEMA = "iraod-comparison-report-v1"
ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows, empty_fields):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def figures(report, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved = []
    for ds in ("RSAR", "DIOR"):
        for role in report["roles"]:
            rows = [r for r in report["summary"] if r["dataset"] == ds and r["metric"] == "mPC"
                    and r["role"] in ("source", role) and r["mean"] is not None]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(5, 3))
            for row in rows:
                x = "ABCDEF".index(row["method"])
                ax.errorbar(x, row["mean"], yerr=row["sample_std"], fmt="o",
                            capsize=3, color="black" if row["method"] == "A" else "#0072B2")
            ax.set_xticks(range(6), list("ABCDEF"))
            ax.set_ylabel("mPC (AP50, 0-1)")
            ax.spines[["top", "right"]].set_visible(False)
            for suffix in ("png", "pdf"):
                name = f"{ds}_{role}_mpc.{suffix}"
                fig.savefig(out / name, dpi=300, bbox_inches="tight")
                saved.append(name)
            plt.close(fig)
    return saved


def render_report(report, out, docx_python):
    """Render already collected numbers locally without reopening remote model artifacts."""
    out = Path(out)
    report["figures"] = figures(report, out)
    report["rendering"] = "figures_and_docx"
    write_json(out / "report.json", report)
    subprocess.run([
        docx_python, "-m", "tools.build_multiseed_comparison_report",
        "--report", str((out / "report.json").resolve()),
        "--out", str((out / "comparison_report_cn.docx").resolve()),
    ], cwd=ROOT, check=True)
    write_json(out / "build_status.json", {
        "report_build": "complete", "result_status": report["status"],
        "full_test_roi": report["qualitative_coverage"]["full_test_roi"]})
    return report


def build_report(manifest_path, out_dir, docx_python=None, metadata_only=False):
    if not metadata_only and not docx_python:
        raise ValueError("A DOCX interpreter is required unless metadata_only is explicit")
    manifest = read_json(manifest_path)
    if manifest["schema"] != SCHEMA:
        raise ValueError("Expected iraod-comparison-report-v1 input manifest")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    prediction_fields = [
        "dataset", "domain", "method", "seed", "role", "prediction_image_index",
        "image_id", "n_post_nms_detections", "predictions", "prediction_image_ids",
        "scope", "cell_status",
    ]
    with (out / "prediction_image_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=prediction_fields)
        writer.writeheader()
        raw, per_class, _, roles = collect_quantitative(manifest, writer.writerow)
    stats = summarize(raw, roles)
    roi_fields = [
        "dataset", "domain", "method", "role", "seed", "checkpoint_domain", "checkpoint",
        "config", "image_id", "scope", "roi_status", "vis_status", "feature_file",
        "n_detections", "visualization_file", "evidence",
    ]
    if manifest.get("qualitative_evidence"):
        _, embeddings, coverage = qualitative_evidence(manifest)
        (out / "roi_vis_coverage.csv").symlink_to(coverage["roi_image_manifest"])
    else:
        with (out / "roi_vis_coverage.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=roi_fields)
            writer.writeheader()
            _, embeddings, coverage = qualitative_evidence(manifest, writer.writerow)
    complete_cells = sum(r["status"] == "complete" for r in raw)
    complete = (complete_cells == len(raw)
                and coverage["roi_complete"] == coverage["roi_expected_image_roles"]
                and coverage["vis_complete"] == 3520 and coverage["embeddings_complete"] == 24)
    report = {
        "schema": SCHEMA, "input_manifest": str(Path(manifest_path).resolve()),
        "artifact_directory": str(out.resolve()),
        "code_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "status": "declared_scopes_complete" if complete else "partial",
        "roles": roles, "quantitative_expected_cells": len(raw),
        "quantitative_complete_cells": complete_cells,
        "quantitative_scope": "full TEST predictions; all post-NMS rows; image-order evidence required",
        "per_class_precision": "printed AP table; never used to reconstruct full-precision metric.mAP",
        "raw_results": raw, "per_class": per_class, **stats,
        "producer_metadata_corrections": [
            r["producer_metadata_correction"] for r in raw if "producer_metadata_correction" in r],
        "qualitative_coverage": coverage, "embedding_index": embeddings,
        "collection": manifest.get("collection", {}),
        "source_ids": manifest["source_ids"],
        "source_provenance": manifest.get("source_provenance", {}),
        "checkpoints": read_rows(manifest["checkpoints"]),
    }
    for name, rows, fields in (
        ("raw_results", raw, ("dataset", "domain", "method", "seed", "role", "mAP50", "status")),
        ("per_class", per_class, ("dataset", "domain", "method", "seed", "role", "class_name", "AP50")),
        ("per_seed", stats["per_seed"], ("dataset", "method", "role", "seed", "mPC")),
        ("per_domain", stats["per_domain"], ("dataset", "domain", "method", "role", "mean")),
        ("summary", stats["summary"], ("dataset", "method", "role", "metric", "mean", "sample_std")),
        ("paired_statistics", stats["paired_statistics"], ("dataset", "method", "role", "metric", "n")),
        ("recovery", stats["recovery"], ("dataset", "domain", "method", "seed", "recovery_ratio")),
        ("embedding_index", embeddings, ("dataset", "domain", "comparison", "status")),
        ("roi_group_coverage", coverage["roi_group_index"], ("run_id", "status")),
        ("producer_metadata_audit", report["producer_metadata_corrections"],
         ("dataset", "domain", "method", "field", "recorded", "actual", "source_record")),
    ):
        write_csv(out / f"{name}.csv", rows, fields)
    history = historical_paths(manifest) + historical_paths(manifest, per_class=True)
    if history:
        (out / "historical").mkdir()
        names = set()
        for path in history:
            if Path(path).name in names:
                raise ValueError("Historical file basenames must be distinct")
            names.add(Path(path).name)
            shutil.copyfile(path, out / "historical" / Path(path).name)
    report["figures"] = []
    report["rendering"] = "deferred_metadata_only"
    write_json(out / "report.json", report)
    # Evidence collection may be partial; this marker means only report construction finished.
    write_json(out / "build_status.json", {
        "report_build": "metadata_complete",
        "result_status": report["status"],
        "full_test_roi": coverage["full_test_roi"]})
    if not metadata_only:
        render_report(report, out, docx_python)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--render-dir", help="Render an existing collected report without artifact reinspection")
    parser.add_argument("--out-dir")
    rendering = parser.add_mutually_exclusive_group(required=True)
    rendering.add_argument("--docx-python",
                           help="Existing python-docx interpreter; no installation or GPU launch")
    rendering.add_argument("--metadata-only", action="store_true",
                           help="Scoped evidence inspection without rendering figures or final DOCX")
    args = parser.parse_args()
    if args.render_dir:
        if not args.docx_python or args.out_dir:
            parser.error("--render-dir requires --docx-python and no --out-dir")
        result = render_report(read_json(Path(args.render_dir) / "report.json"),
                               args.render_dir, args.docx_python)
    else:
        if not args.out_dir:
            parser.error("--manifest requires --out-dir")
        result = build_report(args.manifest, args.out_dir, args.docx_python, args.metadata_only)
    print(f"{result['status']}: {result['quantitative_complete_cells']}/"
          f"{result['quantitative_expected_cells']} quantitative cells; "
          f"RoI {result['qualitative_coverage']['roi_complete']}/"
          f"{result['qualitative_coverage']['roi_expected_image_roles']} (full TEST)")


if __name__ == "__main__":
    main()
