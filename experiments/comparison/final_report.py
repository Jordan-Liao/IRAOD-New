"""Build an evidence-backed, versioned CPU comparison report; never run models."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

from experiments.comparison.report_inputs import collect_quantitative, historical_paths, read_rows
from experiments.comparison.report_qualitative import qualitative_evidence, validate_plan
from experiments.comparison.report_statistics import comparison_groups, declared_methods, summarize
from experiments.comparison.result_completion import read_json, write_json, plan_methods


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
    groups = (comparison_groups(report["methods"]) if "methods" in report
              else {"": list("BCDEF")})
    for group, members in groups.items():
        methods = ["A", *members]
        for ds in ("RSAR", "DIOR"):
            for role in report["roles"]:
                rows = [r for r in report["summary"] if r["dataset"] == ds and r["metric"] == "mPC"
                        and r["role"] in ("source", role) and r["mean"] is not None
                        and r["method"] in methods]
                if not rows:
                    continue
                fig, ax = plt.subplots(figsize=(max(5, len(methods)) if group else 5, 3))
                for row in rows:
                    x = methods.index(row["method"])
                    ax.errorbar(x, row["mean"], yerr=row["sample_std"], fmt="o",
                                capsize=3, color="black" if row["method"] == "A" else "#0072B2")
                ax.set_xticks(range(len(methods)), methods, rotation=25 if group else 0)
                ax.set_ylabel("mPC (AP50, 0-1)")
                if group:
                    ax.set_title(group + " (declared order; no cross-budget ranking)")
                ax.spines[["top", "right"]].set_visible(False)
                for suffix in ("png", "pdf"):
                    name = f"{ds}_{role}{'_' + group if group else ''}_mpc.{suffix}"
                    fig.savefig(out / name, dpi=300, bbox_inches="tight")
                    saved.append(name)
                plt.close(fig)
    return saved


def extension_tables(report, out):
    """Booktabs exports never mix checkpoint roles, budgets or supervision groups."""
    from experiments.comparison.publish_report import tex_escape

    values = {(r["dataset"], r["method"], r["role"], r["metric"]): r
              for r in report["summary"]}
    saved = []
    for group, members in comparison_groups(report["methods"]).items():
        methods = ["A", *members]
        for role in report["roles"]:
            lines = [
                "% Requires booktabs. Missing complete seed blocks remain --.",
                r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{1.15}",
                r"\caption{" + tex_escape(group + " / " + role) +
                r". Mean $\pm$ sample standard deviation over three adaptation seeds; "
                r"A is one fixed reference. Values in percentage points. "
                r"No cross-budget or cross-supervision ranking.}",
                r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
                r"Method & \multicolumn{3}{c}{RSAR} & \multicolumn{3}{c}{DIOR-R} \\",
                r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
                r" & Clean $\uparrow$ & mPC $\uparrow$ & $\Delta_A$ $\uparrow$"
                r" & Clean $\uparrow$ & mPC $\uparrow$ & $\Delta_A$ $\uparrow$ \\",
                r"\midrule",
            ]
            for method in methods:
                row = [tex_escape(method)]
                selected_role = "source" if method == "A" else role
                for dataset in ("RSAR", "DIOR"):
                    for metric in ("clean_mAP50", "mPC", "delta_A"):
                        entry = values[dataset, method, selected_role, metric]
                        if entry["mean"] is None:
                            row.append("--")
                            continue
                        text = f"{100 * entry['mean']:.2f}"
                        if entry["sample_std"] is not None:
                            text += r" \pm " + f"{100 * entry['sample_std']:.2f}"
                        row.append("$" + text + "$")
                lines.append(" & ".join(row) + r" \\")
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
            filename = f"{group}_{role}_table.tex"
            (Path(out) / filename).write_text("\n".join(lines))
            saved.append(filename)
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
    methods = declared_methods(manifest.get("methods"))
    quant_only = manifest.get("quantitative_only", False)
    if "methods" in manifest and not quant_only:
        if not manifest.get("qualitative_plan"):
            raise ValueError("Explicit-method reports need a declared qualitative plan or quantitative-only scope")
        qualitative_plan = read_json(manifest["qualitative_plan"])
        validate_plan(qualitative_plan)
        if set(plan_methods(qualitative_plan)) - set(methods) - set("ABCDEF"):
            raise ValueError("New qualitative methods must belong to the declared report method set")
    if "comparison_groups" in manifest and manifest["comparison_groups"] != comparison_groups(methods):
        raise ValueError("Comparison groups must preserve the declared budget/supervision boundaries")
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
    stats = summarize(raw, roles, manifest.get("methods"))
    roi_fields = [
        "dataset", "domain", "method", "role", "seed", "checkpoint_domain", "checkpoint",
        "config", "image_id", "scope", "roi_status", "vis_status", "feature_file",
        "n_detections", "visualization_file", "evidence",
    ]
    if quant_only:
        embeddings = []
        coverage = {
            "status": "pending_not_collected", "full_test_roi": False,
            "roi_complete": 0, "roi_expected_image_roles": None,
            "vis_complete": 0, "vis_expected_image_roles": None,
            "embeddings_complete": 0, "embeddings_expected": None,
            "roi_group_index": [],
            "scope": "Extension RoI/visualizations/t-SNE pending; core denominators do not apply",
        }
        write_csv(out / "roi_vis_coverage.csv", [], roi_fields)
    elif manifest.get("qualitative_evidence"):
        _, embeddings, coverage = qualitative_evidence(manifest)
        (out / "roi_vis_coverage.csv").symlink_to(coverage["roi_image_manifest"])
    else:
        with (out / "roi_vis_coverage.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=roi_fields)
            writer.writeheader()
            _, embeddings, coverage = qualitative_evidence(manifest, writer.writerow)
    complete_cells = sum(r["status"] == "complete" for r in raw)
    expected_vis = coverage["vis_expected_image_roles"] if "methods" in manifest else 3520
    expected_embeddings = coverage["embeddings_expected"] if "methods" in manifest else 24
    complete = (not quant_only and complete_cells == len(raw)
                and coverage["roi_complete"] == coverage["roi_expected_image_roles"]
                and coverage["vis_complete"] == expected_vis
                and coverage["embeddings_complete"] == expected_embeddings)
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
    if quant_only:
        report["quantitative_only"] = True
        report["quantitative_status"] = "complete" if complete_cells == len(raw) else "incomplete"
    if "methods" in manifest:
        report["methods"] = methods
        report["comparison_groups"] = comparison_groups(methods)
        report["latex_tables"] = extension_tables(report, out)
        if not quant_only:
            report["qualitative_adaptation_seed"] = qualitative_plan.get("adaptation_seed", 42)
            report["qualitative_reference_methods"] = [
                m for m in plan_methods(qualitative_plan) if m not in methods]
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
