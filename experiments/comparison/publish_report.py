"""Stage the exact final delivery from a complete CPU report and terminal evidence."""

import argparse
from pathlib import Path
import shutil
import subprocess

from experiments.comparison.final_report import ROOT, write_csv
from experiments.comparison.report_inputs import CLASSES, read_rows
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


EXPERIMENT_FILES = ("protocol.md", "method_matrix.csv", "run_manifest.yaml",
                    "leakage_audit.md", "failures.md", "reproduction_commands.sh")
REQUIRED_RESULTS = ("raw_results.csv", "per_corruption_summary.csv", "per_class_summary.csv",
                    "statistical_tests.csv", "main_table.tex", "ablation_table.tex",
                    "oracle_table.tex", "result_summary_cn.md", "visualization_manifest.csv",
                    "IRAOD_full_comparison_report_cn.docx")


def admit_final(report, terminal):
    expected = {(ds, d, m, s, "source" if m == "A" else "ema")
                for ds, domains in DOMAINS.items() for d in domains for m in "ABCDEF"
                for s in ((42,) if m == "A" else (42, 43, 44))}
    identity = lambda r: (r["dataset"], r["domain"], r["method"], r["seed"], r["role"])
    raw = report["raw_results"]
    if (report["status"] != "declared_scopes_complete" or report["roles"] != ["ema"]
            or report["quantitative_complete_cells"] != 192
            or report["quantitative_expected_cells"] != 192
            or len(raw) != 192 or {identity(r) for r in raw} != expected
            or any(r["status"] != "complete" for r in raw)):
        raise ValueError("Final publication requires all 192 complete native EMA/source cells")
    cells = terminal["cells"]
    trained = {(ds, d, m, s) for ds, domains in DOMAINS.items() for d in domains
               for m in "BCDEF" for s in (42, 43, 44) if s != 42 or d == "clean"}
    terminal_id = lambda r: tuple(r["cell"][k] for k in ("dataset", "domain", "method", "seed"))
    if (terminal["status"] != "complete" or terminal["active"] or len(cells) != 192
            or {terminal_id(r) for r in cells} != {r[:4] for r in expected}
            or {terminal_id(r) for r in cells if r["train_requested"]} != trained
            or any(r["eval"] != "complete" or r["train"] != "complete" for r in cells)):
        raise ValueError("Parent gate requires terminal 130/130 training and 192/192 evaluation evidence")
    expected_classes = {(*identity(r), c) for r in raw for c in CLASSES[r["dataset"]]}
    classes = report["per_class"]
    if (len(classes) != len(expected_classes)
            or {(*identity(r), r["class_name"]) for r in classes} != expected_classes
            or any(r["status"] != "complete" for r in classes)):
        raise ValueError("Final publication requires all 2048 bound per-class AP rows")
    cov = report["qualitative_coverage"]
    for key, count in (("roi_complete_groups", 132), ("roi_complete", 1267816),
                       ("roi_detection_rows", 18468211), ("vis_complete", 3520),
                       ("embeddings_complete", 24)):
        if cov[key] != count:
            raise ValueError(f"Final qualitative count mismatch: {key}")
    if len(cov.get("visualization_index", [])) != 3520:
        raise ValueError("Final publication requires the accepted frozen visualization index")
    checkpoints = {(r["dataset"], r["domain"], r["method"], r["role"]): r["checkpoint"]
                   for r in raw if r["seed"] == 42}
    for r in cov["visualization_index"]:
        if r["role"] != "student" and r["checkpoint"] != checkpoints[
                r["dataset"], r["domain"], r["method"], r["role"]]:
            raise ValueError("Qualitative and quantitative seed42 checkpoints differ")
    if any(r.get("effective_training_code_sha") is None or not r.get("evaluation_code_sha")
           or not r.get("source_id") or not r.get("prediction_image_ids") for r in raw):
        raise ValueError("Final source/checkpoint/ID/code provenance is incomplete")


def tex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&")


def result_table(report, methods, caption, label):
    # Adapted from the academic-latex-tables grouped-header booktabs template.
    metrics = ("clean_mAP50", "mPC", "rPC_percent", "delta_A")
    values = {(r["dataset"], r["method"], r["metric"]): r for r in report["summary"]}
    lines = [
        "% Requires booktabs, graphicx, and xcolor with the table option.",
        r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\caption{" + caption + " " +
        r"Source-Free, OBB, target labels for offline evaluation only. "
        r"Source-available and target-supervised groups are not run. "
        r"Mean $\pm$ sample std over three adaptation seeds; A is one fixed source (SD N/A). "
        r"AP and $\Delta_A$ in percentage points; historical rPC in percent. "
        r"Bold: best mean within the displayed Source-Free group, separately per dataset/metric.}",
        r"\label{tab:" + label + "}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}lllcccccccc@{}}", r"\toprule",
        r"Method & VLM & Target labels & \multicolumn{4}{c}{RSAR} & \multicolumn{4}{c}{DIOR-R} \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        r" & & & Clean $\uparrow$ & mPC $\uparrow$ & rPC $\uparrow$ & $\Delta_A$ $\uparrow$"
        r" & Clean $\uparrow$ & mPC $\uparrow$ & rPC $\uparrow$ & $\Delta_A$ $\uparrow$ \\",
        r"\midrule",
    ]
    method_info = {r["id"]: r for r in report["method_matrix"]}
    for method in methods:
        info = method_info[method]
        row = [tex_escape(method + " " + info["name"]), tex_escape(info["vlm"]), "eval only"]
        for dataset in DOMAINS:
            for metric in metrics:
                entry = values[dataset, method, metric]
                scale = 1 if metric == "rPC_percent" else 100
                best = max(values[dataset, m, metric]["mean"] for m in methods)
                mean = f"{scale * entry['mean']:.2f}"
                if entry["mean"] == best:
                    mean = r"\mathbf{" + mean + "}"
                std = entry["sample_std"]
                row.append("$" + mean + (r" \pm " + f"{scale * std:.2f}" if std is not None
                                        else r"\;(\mathrm{N/A})") + "$")
        if method == "F":
            lines.append(r"\rowcolor{blue!7}")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return "\n".join(lines)


def stage_publication(report_dir, terminal_path, out_dir, docx_python):
    source = Path(report_dir)
    report = read_json(source / "report.json")
    terminal = read_json(terminal_path)
    admit_final(report, terminal)
    if read_json(source / "build_status.json")["report_build"] != "complete":
        raise ValueError("Final publication requires rendered report, not metadata-only inspection")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    results = out / "results/paper_comparison"
    experiment = out / "experiments/comparison"
    results.mkdir(parents=True)
    experiment.mkdir(parents=True)
    original = ROOT / "results/paper_comparison"
    history = results / "historical/seed42"
    history.mkdir(parents=True)
    frozen = original / "historical/seed42"
    tracked = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "results/paper_comparison"], text=True).splitlines()
    if frozen.is_dir():
        shutil.copytree(frozen, history, dirs_exist_ok=True)
    else:
        for name in tracked:
            path = ROOT / name
            if path.parent == original:
                shutil.copyfile(path, history / path.name)
    for name in ("raw_results.csv", "dior_raw_results.csv", "per_class_summary.csv", "dior_per_class.csv"):
        if (source / "historical" / name).read_bytes() != (history / name).read_bytes():
            raise ValueError("Consumer and publication must use the same byte-preserved seed42 history")
    historical_docs = results / "historical/experiment_docs"
    historical_docs.mkdir()
    for name in EXPERIMENT_FILES:
        shutil.copyfile(ROOT / "experiments/comparison" / name, historical_docs / name)
    # Keep the accepted small evidence and the source manifest needed by the collector.
    for name in tracked:
        path = ROOT / name
        if path.parent == original and (path.name.startswith("qualitative")
                                       or path.name in ("source_producer_metadata_corrections.json",
                                                        "dior_checkpoint_manifest.json")):
            shutil.copyfile(path, results / path.name)
    aliases = {"per_corruption_summary.csv": "per_domain.csv",
               "per_class_summary.csv": "per_class.csv", "statistical_tests.csv": "paired_statistics.csv"}
    for path in source.iterdir():
        if path.suffix in (".csv", ".png", ".pdf") and path.name not in (
                "prediction_image_coverage.csv", "roi_vis_coverage.csv"):
            shutil.copyfile(path, results / path.name)
    for name, existing in aliases.items():
        shutil.copyfile(source / existing, results / name)
    methods = read_rows(ROOT / "experiments/comparison/method_matrix.csv")
    for row in methods:
        row["status"] = "complete" if row["id"] in tuple("ABCDEF") else "not_run"
    report["method_matrix"] = methods
    report["parent_terminal_evidence"] = str(Path(terminal_path).resolve())
    write_json(results / "terminal_state.json", terminal)
    write_json(results / "report.json", report)
    write_csv(results / "visualization_manifest.csv",
              report["qualitative_coverage"]["visualization_index"], ("dataset", "corruption", "image"))
    write_csv(experiment / "method_matrix.csv", methods, ("id", "fairness_group", "status"))
    (results / "main_table.tex").write_text(result_table(
        report, "ABCDEF", "Strict A--F comparison.", "main"), encoding="utf-8")
    (results / "ablation_table.tex").write_text(result_table(
        report, "BDEF", "Existing component combinations: B (neither), D (CGA), E (VLST), "
        "F (both). Reuses the main runs; no extra ablation or interaction significance claimed. "
        "Other planned ablations are not run.", "ablation"), encoding="utf-8")
    (results / "oracle_table.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\setlength{\\tabcolsep}{5pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\caption{Separate fairness groups; no oracle scores or extra runs are invented. "
        "Clean/mPC/rPC/delta and mean/std are N/A.}\\label{tab:oracle}\n"
        "\\begin{tabular}{@{}llll@{}}\\toprule\n"
        "Group / box & VLM & Target labels & Status \\\\\n\\midrule\n"
        "Source-available / OBB & N/A & N/A & Not run \\\\\n"
        "Target-supervised / OBB & SARCLIP-LoRA & training & Not run \\\\\n"
        "\\bottomrule\\end{tabular}\n\\end{table}\n", encoding="utf-8")
    summary = [
        "# IRAOD 最终三种子比较", "",
        "本交付对应 130/130 新训练和 192/192 native-ID TEST 评测终态；"
        "A 为每数据集固定 source，不复制为三种子。B–F 为独立 clean/腐蚀适配后的 final EMA。",
        "", "OBB / VOC AP50，Source-Free；目标标签仅用于离线评测。"
        "Source-available、target-supervised oracle 和额外消融未运行。"
        "B/D/E/F 表仅复用已完成组件组合，不是新增实验。",
        "", "## 完整原始数据", "",
        "raw_results.csv：192 单元；per_class_summary.csv：2048 类别 AP 行（保留打印精度）；"
        "per_corruption_summary.csv：含 clean 的逐域均值与样本标准差；"
        "statistical_tests.csv：配对种子检验；recovery.csv：逐种子逐腐蚀恢复比。",
        "", "## 结果", "", "| 数据集 | 方法 | 指标 | mean ± sample std |",
        "| --- | --- | --- | --- |",
    ]
    for r in report["summary"]:
        if r["metric"] in ("clean_mAP50", "mPC", "rPC_percent", "delta_A"):
            sd = "N/A（固定 source）" if r["sample_std"] is None else f"{r['sample_std']:.8g}"
            summary.append(f"| {r['dataset']} | {r['method']} | {r['metric']} | {r['mean']:.8g} ± {sd} |")
    summary.extend([
        "", "1. 先在种子内平均全部腐蚀，再跨种子统计；std 使用 ddof=1。"
        "n=3、低功效，置信区间仅反映固定 source 条件下的适配随机性。",
        "2. 历史 rPC=100×mPC/固定 A clean TEST；不是 source VAL，也不改成 method clean。"
        "method_clean_normalized_percent 另列。",
        "3. Recovery=(method−A_corruption)/(A_clean_TEST−A_corruption)。"
        "分母非正时 N/A，接近零时不稳定；保留负结果，不截断、不排名。",
        "", "## 证据与边界", "",
        "132 ROI groups / 1,267,816 image-role records / 18,468,211 detections；"
        "3,520 固定子集可视化，24 joint embeddings。定性覆盖仍为 seed42，"
        "不冒充三种子的 ROI 导出。复用已接受 streaming 审计，不重扫百万 NPZ。",
        "原始 seed42 文件逐字节保存在 historical/seed42；"
        "producer_metadata_audit.csv 保留 recorded/actual source producer 更正，原 sidecar 不变。",
        "完整逐图 prediction/ROI 清单与大文件位置见 artifact_manifest.json；"
        "源码、checkpoint、实际 ID sidecar 和 class AP 路径见 raw_results.csv、checkpoints.json。",
        "", "复现入口：experiments/comparison/reproduction_commands.sh。"
        "本交付不声称未运行的 baseline/oracle 已完成。", "",
    ])
    (results / "result_summary_cn.md").write_text("\n".join(summary), encoding="utf-8")
    artifacts = {"report_directory": report["artifact_directory"],
                 "prediction_image_coverage": report["artifact_directory"] + "/prediction_image_coverage.csv",
                 "roi_image_coverage": report["qualitative_coverage"]["roi_image_manifest"],
                 "qualitative_evidence": report["qualitative_coverage"]["evidence_reuse"]}
    write_json(results / "artifact_manifest.json", artifacts)
    write_json(results / "checkpoints.json", read_rows(report["checkpoints"]))
    write_json(results / "source_provenance.json", report["source_provenance"])
    write_json(experiment / "run_manifest.yaml", {
        "status": "complete_declared_A_F_matrix", "consumer_commit": report["code_commit"],
        "parent_terminal": "results/paper_comparison/terminal_state.json",
        "adaptation_seeds": [42, 43, 44], "fixed_source_seed": 42,
        "source_ids": report["source_ids"], "new_training_cells": 130, "native_eval_cells": 192,
        "checkpoints": "results/paper_comparison/checkpoints.json",
        "raw_results": "results/paper_comparison/raw_results.csv", "artifacts": artifacts,
        "unrun": [r["id"] for r in methods if r["status"] == "not_run"],
    })  # JSON is valid YAML; no additional parser dependency.
    (experiment / "protocol.md").write_text(
        "# Final strict A-F protocol\n\n"
        "RSAR: 8 domains, 8538 TEST images, 6 classes. DIOR-R: 4 domains, 11738 TEST images, "
        "20 classes. Domains are clean plus the frozen corruption matrix in run evidence.\n\n"
        "Shared fixed seed42 source per dataset; adaptation seeds 42/43/44. "
        "B-F adapt independently on image-only VAL for each clean/corrupted domain; no source forward, "
        "no target GT in adaptation or checkpoint selection. Global batch 32, LR .02, one epoch; "
        "B-D 1x32, E/F 2x16. Final EMA iterations: RSAR 266, DIOR 185. Final Student retained.\n\n"
        "OBB le90 VOC AP50 (not COCO AP50:95). Class AP retains printed precision; "
        "mAP uses exact metric.mAP. mPC averages all corruptions within seed; delta_A pairs "
        "the same domains with fixed A. Across-seed mean and sample std (ddof=1); A has no SD. "
        "Historical rPC=100*mPC/fixed A clean TEST. Method-clean normalization is separate. "
        "Recovery=(method-A_corruption)/(A_clean_TEST-A_corruption), undefined for nonpositive "
        "denominator, unstable near zero and never ranked. Paired tests/CI at n=3 are low-power "
        "and conditional on the fixed source; see report.json statistical_protocol.\n\n"
        "130 new training and 192 native-ID eval cells are complete per terminal_state.json "
        "and the complete artifact consumer. Qualitative seed42 coverage is independently "
        "accepted: 132 groups, 1267816 NPZs, 18468211 detections, 3520 visualizations, "
        "24 embeddings. Legacy versions and frozen seed42 reports are historical, not current evidence.\n\n"
        "Source-Free A-F only in main tables; source-available/oracle/unported baselines and extra "
        "ablations remain not run. No test-selected checkpoints or negative-result exclusions.\n",
        encoding="utf-8")
    (experiment / "leakage_audit.md").write_text(
        "# Final delivery leakage/provenance audit\n\n"
        "The frozen strict A-F method and training policy remain unchanged: source-free image-only "
        "VAL, no source forward or target-label prototypes, frozen base CLIP/SARCLIP, no LoRA, "
        "final EMA rather than TEST selection. TEST labels are used only in offline AP evaluation. "
        "The report consumer does not train or select models.\n\n"
        "Exact per-cell checkpoint, source ID, native inference ID sidecar, recorded/effective "
        "training SHA and evaluation SHA are in raw_results.csv. The RSAR source producer "
        "correction b474aaa versus recorded adaptation 0f98 is explicit in producer_metadata_audit.csv; "
        "raw sidecars are preserved. Source-available and target-supervised groups are not run.\n\n"
        "The earlier executed-config audit is preserved in "
        "results/paper_comparison/historical/experiment_docs/leakage_audit.md. "
        "This delivery adds provenance evidence, not a new training-code audit.\n", encoding="utf-8")
    (experiment / "failures.md").write_text(
        "# Final delivery status and historical failures\n\n"
        "The supplied finite terminal state has 130/130 requested training cells and "
        "192/192 native evaluations complete, with no active jobs. All 192 consumer cells "
        "and 2048 per-class rows are complete; no metric conflicts are suppressed. "
        "Terminal per-cell reasons are retained in terminal_state.json. Previous failed attempts "
        "are not erased or counted as successful runs.\n\n"
        "Legacy stalls/OOM/export limitations and their recorded resolutions remain in "
        "results/paper_comparison/historical/experiment_docs/failures.md. "
        "Unported baselines/oracle/extra ablations remain not run; see method_matrix.csv.\n",
        encoding="utf-8")
    shutil.copyfile(ROOT / "experiments/comparison/reproduction_commands.sh",
                    experiment / "reproduction_commands.sh")
    subprocess.run([docx_python, "-m", "tools.build_multiseed_comparison_report",
                    "--report", str((results / "report.json").resolve()),
                    "--out", str((results / "IRAOD_full_comparison_report_cn.docx").resolve())],
                   cwd=ROOT, check=True)
    for path in [*(results / n for n in REQUIRED_RESULTS),
                 *(experiment / n for n in EXPERIMENT_FILES)]:
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Required final delivery missing: {path}")
    write_json(out / "publication.json", {
        "status": "complete_staged_not_installed", "report_commit": report["code_commit"],
        "results": list(REQUIRED_RESULTS), "experiments": list(EXPERIMENT_FILES),
        "parent_terminal_evidence": str(Path(terminal_path).resolve()),
    })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--terminal-state", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--docx-python", required=True)
    args = parser.parse_args()
    print(stage_publication(args.report_dir, args.terminal_state, args.out_dir, args.docx_python))


if __name__ == "__main__":
    main()
