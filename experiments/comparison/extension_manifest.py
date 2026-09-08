"""Prepare finite nontraining extensions; GPU execution belongs to the compute owner."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

from experiments.comparison.collect_report_manifest import load_resolver
from experiments.comparison.report_inputs import class_table, EXPECTED_IMAGES
from experiments.comparison.report_qualitative import validate_plan
from experiments.comparison.result_completion import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]


def cell_key(dataset, domain, seed, method):
    return f"{dataset}/{domain}/{seed}/{method}"


def seeded_plan(base, paths, seed, output):
    if seed not in (43, 44) or base.get("adaptation_seed", 42) != 42:
        raise ValueError("Extensions reuse a seed42 base and adapt seeds43/44 only")
    validate_plan(base)
    runs = []
    for original in base["runs"]:
        if original["method"] == "A":
            runs.append(dict(original))
            continue
        resolve = paths.ema_path if original["role"] == "ema" else paths.student_path
        checkpoint = resolve(original["dataset"], original["domain"], str(seed), original["method"])
        runs.append({
            **original, "seed": seed, "checkpoint": checkpoint,
            "out_dir": str(Path(output) / f"seed_{seed}" / base["schema"] / original["run_id"]),
        })
    plan = {**base, "adaptation_seed": seed, "source_seed": 42, "runs": runs,
            "source_reuse": "A/source run objects and exports reused unchanged; no A re-extraction"}
    validate_plan(plan)
    return plan


def prepare(base_plan, core_report, core_paths, out_dir, eval_code, python, artifact_root=None):
    base = read_json(base_plan)
    report = read_json(core_report)
    validate_plan(base)
    if report["status"] != "declared_scopes_complete" or report["quantitative_complete_cells"] != 192:
        raise ValueError("Nontraining extensions must bind the completed core A-F delivery")
    paths = load_resolver(core_paths)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    artifacts = Path(artifact_root).resolve() if artifact_root else out
    queue = out / "student_queue"
    queue.mkdir()
    eval_code = Path(eval_code).resolve()
    evaluation_sha = subprocess.check_output(
        ["git", "-C", str(eval_code), "rev-parse", "HEAD"], text=True).strip()
    core = {cell_key(r["dataset"], r["domain"], r["seed"], r["method"]): r
            for r in report["raw_results"] if r["method"] != "A"}
    domains = {(r["dataset"], r["domain"]): r for r in base["runs"] if r["method"] == "A"}
    cells = {}
    for key, evidence in core.items():
        ds, domain, seed, method = (evidence[k] for k in ("dataset", "domain", "seed", "method"))
        if evidence["checkpoint"] != paths.ema_path(ds, domain, str(seed), method):
            raise ValueError(f"Resolver does not bind the completed EMA evidence: {key}")
        checkpoint = Path(paths.student_path(ds, domain, str(seed), method))
        if not checkpoint.is_file() or not checkpoint.stat().st_size:
            raise ValueError(f"Missing retained final Student checkpoint: {checkpoint}")
        reference = domains[ds, domain]
        cells[key] = {
            "dataset": ds, "domain": domain, "seed": seed, "method": method, "role": "student",
            "checkpoint": str(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "config": evidence["config"], "ann_file": reference["ann_file"],
            "img_prefix": reference["img_prefix"],
            "training_code_sha": evidence["effective_training_code_sha"],
            "source_id": evidence["source_id"],
            "eval_dir": str(Path(paths.eval_full_dir(ds, domain, str(seed), method)).with_name(
                f"eval_full_{domain}_student_ids_v1")),
            "status": "bound_not_inspected",
        }
    if len(cells) != 180:
        raise ValueError("Student extension must contain exactly 180 B-F cells")
    runtime = {
        "schema": "iraod-nontraining-extensions-v1", "python": str(Path(python).absolute()),
        "evaluation_code": str(eval_code), "evaluation_code_sha": evaluation_sha,
        "core_report": str(Path(core_report).resolve()), "core_paths": str(Path(core_paths).resolve()),
        "core_plan": str(Path(base_plan).resolve()), "student_cells": cells,
        "legacy_student_queue": str(artifacts),
    }
    write_json(queue / "runtime.json", runtime)
    shutil.copyfile(core_paths, queue / "core_paths_snapshot.py")
    (queue / "paths.py").write_text(
        "from pathlib import Path\n"
        "from experiments.comparison.collect_report_manifest import load_resolver\n"
        "from experiments.comparison.result_completion import read_json\n"
        "HERE = Path(__file__).resolve().parent\n"
        "CORE = load_resolver(HERE / 'core_paths_snapshot.py')\n"
        "DATA = read_json(HERE / 'runtime.json')\n"
        "EVALUATION_CODE_SHA = DATA['evaluation_code_sha']\n"
        "LEGACY_STUDENT_QUEUE = DATA['legacy_student_queue']\n"
        "EXPECT_PRED = CORE.EXPECT_PRED\n"
        "root, method_dir = CORE.root, CORE.method_dir\n"
        "ema_path, student_path = CORE.ema_path, CORE.student_path\n"
        "eval_full_dir = CORE.eval_full_dir\n"
        "def eval_student_dir(ds, domain, seed, method):\n"
        "    return DATA['student_cells'][f'{ds}/{domain}/{seed}/{method}']['eval_dir']\n")
    (queue / "run_eval_full.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "exec " + shlex.quote(str(python)) +
        " -m experiments.comparison.extension_manifest evaluate --queue " +
        shlex.quote(str(queue)) +
        ' --gpu "$1" --dataset "$2" --domain "$3" --seed "$4" --method "$5" --role "$6"\n')
    (queue / "with_gpu_lock.sh").symlink_to(Path(core_paths).resolve().parent / "with_gpu_lock.sh")
    (queue / "student.list").write_text("".join(
        f"{r['dataset']} {r['domain']} {r['seed']} {r['method']} student\n" for r in cells.values()))
    roi_jobs, embedding_jobs = [], []
    for seed in (43, 44):
        plan = seeded_plan(base, paths, seed, artifacts / "qualitative")
        filename = out / f"qualitative_seed{seed}.json"
        write_json(filename, plan)
        for run in plan["runs"]:
            if run["method"] == "A":
                continue
            roi_jobs.append({
                "seed": seed, "run_id": run["run_id"], "plan": str(filename),
                "out_dir": run["out_dir"], "checkpoint": run["checkpoint"],
                "export_code": str(eval_code), "export_code_sha": evaluation_sha,
                "export_argv": [str(python), "-m",
                                "experiments.comparison.dior_recovery.extract_roi_pre_fc_cls",
                                "--plan", str(filename), "--run-id", run["run_id"]],
                "visualize_argv": [str(python), "-m", "experiments.comparison.result_completion",
                                   "visualize", "--plan", str(filename), "--run-id", run["run_id"]],
            })
        for dataset, domain in domains:
            for role in ("ema", "student"):
                embedding_jobs.append({
                    "seed": seed, "dataset": dataset, "domain": domain, "comparison": role,
                    "argv": [str(python), "-m", "experiments.comparison.joint_tsne",
                             "--plan", str(filename), "--dataset", dataset, "--domain", domain,
                             "--comparison", role, "--out-dir",
                             str(artifacts / "qualitative" / f"seed_{seed}" / "tsne" / dataset / domain / role)],
                })
    write_json(out / "roi_jobs.json", roi_jobs)
    write_json(out / "embedding_jobs.json", embedding_jobs)
    write_json(out / "extension_scope.json", {
        "status": "prepared_not_execution_evidence", "student_evaluations": 180,
        "new_roi_groups": 240, "reused_source_groups": 12,
        "new_roi_image_roles": sum(len(r["image_ids"]) for r in base["runs"] if r["method"] != "A") * 2,
        "new_visualizations": 6400, "new_embeddings": 48,
        "qualitative_jobs": str(out / "roi_jobs.json"),
        "embedding_jobs": str(out / "embedding_jobs.json"),
        "note": "One plan per adaptation seed; A/source42 is reused, never counted as new inference.",
    })
    return out


def evaluate(queue, gpu, dataset, domain, seed, method, role):
    if role != "student":
        raise ValueError("This entry selects only Student bindings")
    runtime = read_json(Path(queue) / "runtime.json")
    cell = runtime["student_cells"][cell_key(dataset, domain, seed, method)]
    evaluate_binding(cell, runtime, gpu)


def evaluate_binding(cell, runtime, gpu):
    """Execute one explicit native evaluation binding under its owner's GPU lock."""
    if gpu not in (4, 5, 6, 7) or cell["role"] not in ("ema", "student"):
        raise ValueError("Only approved-GPU final EMA/Student bindings are supported")
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Invoke via the finite worker holding the shared GPU lock")
    dataset, domain, seed, method = (cell[k] for k in ("dataset", "domain", "seed", "method"))
    out = Path(cell["eval_dir"])
    out.mkdir(parents=True, exist_ok=False)
    python, code = runtime["python"], Path(runtime["evaluation_code"])
    prefix = str(Path(python).parent.parent)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "PYTHONPATH": str(code), "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
           "IRAOD_RUNTIME_READY": "1", "CONDA_PREFIX": prefix,
           "LD_LIBRARY_PATH": prefix + "/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
           "RSAR_ROOT": "/mnt/shared/zechuan/iraod_data/RSAR"}
    command = [
        python, str(code / "test.py"), cell["config"], cell["checkpoint"],
        "--eval", "mAP", "--out", str(out / "predictions.pkl"), "--work-dir", str(out),
        "--training-code-sha", cell["training_code_sha"], "--cfg-options",
        "data.test.ann_file=" + cell["ann_file"], "data.test.img_prefix=" + cell["img_prefix"],
    ]
    write_json(out / "execution.json", {**cell, "command": command,
                                      "evaluation_code_sha": runtime["evaluation_code_sha"]})
    with (out / "eval.log").open("w") as log:
        result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
    (out / "eval_status").write_text(
        f"eval_exit={result.returncode} {datetime.now(timezone.utc).isoformat()} "
        f"name={method} domain={domain} seed={seed} role={cell['role']} "
        f"checkpoint={cell['checkpoint']} sidecar=predictions.pkl.image_ids.json\n")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    text = (out / "eval.log").read_text()
    table = re.search(r"(\| class\s+\|[\s\S]+?\| mAP\s+\|[\s\S]+?\n)", text)
    if table is None:
        raise ValueError("Native evaluation did not emit a class AP table")
    (out / "class_ap.txt").write_text(table.group(1) + "\n")
    class_table(out / "class_ap.txt", dataset)
    order = read_json(out / "predictions.pkl.image_ids.json")
    if (order["checkpoint"] != cell["checkpoint"]
            or order["evaluation_code_sha"] != runtime["evaluation_code_sha"]
            or order["n_images"] != EXPECTED_IMAGES[dataset]):
        raise ValueError("Native checkpoint/code/full-TEST identity mismatch")
    (out / "pred_count.txt").write_text(f"{order['n_images']} expect={EXPECTED_IMAGES[dataset]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("base-plan", "core-report", "core-paths", "out-dir", "eval-code", "python"):
        prepare_parser.add_argument("--" + name, required=True)
    prepare_parser.add_argument("--artifact-root",
                                help="Existing compute-owned extension root; metadata uses a NEW out-dir")
    evaluate_parser = commands.add_parser("evaluate")
    for name in ("queue", "dataset", "domain", "method", "role"):
        evaluate_parser.add_argument("--" + name, required=True)
    evaluate_parser.add_argument("--gpu", required=True, type=int)
    evaluate_parser.add_argument("--seed", required=True, type=int)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = prepare(**args) if command == "prepare" else evaluate(**args)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
