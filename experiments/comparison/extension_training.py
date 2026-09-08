"""Finite official-core OBB preparation; execution is explicit and GPU-lock owned."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess

from experiments.comparison.collect_report_manifest import load_resolver
from experiments.comparison.extension_manifest import cell_key, evaluate_binding
from experiments.comparison.report_inputs import EXPECTED_IMAGES, FINAL_ITERATION
from experiments.comparison.report_qualitative import validate_plan
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
PORT_METHODS = ("IRG", "LPLD", "SFUT")
METHODS = (*PORT_METHODS, "B_REG")
SEEDS = (42, 43, 44)
TARGET_VAL_SIZE = {"RSAR": 8467, "DIOR": 5863}
EVALUATION_SHA = "331d2131b84651f0a2930a3d53faeefad8701531"


def code_sha(code):
    return subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()


def require_file(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing nonempty file: {path}")
    return path


def target_val(dataset, domain, test_prefix):
    """Only the accepted datasets' TEST-to-VAL image layout, never annotations."""
    test = Path(test_prefix)
    if dataset == "RSAR" and test.parts[-2:] == ("test", "images"):
        val = test.parent.parent / "val" / "images"
    elif dataset == "DIOR" and test.parts[-2:] == (domain, "test"):
        val = test.with_name("val")
    else:
        raise ValueError(f"Unsupported {dataset} TEST image layout: {test}")
    for directory in (test, val):
        if not directory.is_dir():
            raise ValueError(f"Missing image directory: {directory}")
    return str(val)


def prepare(base_plan, core_report, core_paths, out_dir, artifact_root,
            eval_code, python, methods):
    """Write only a NEW metadata queue; never create the formal output root."""
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError("Explicitly select IRG, LPLD, SFUT and/or B_REG")
    methods = tuple(dict.fromkeys(methods))
    base, report = read_json(base_plan), read_json(core_report)
    validate_plan(base)
    if base.get("adaptation_seed", 42) != 42:
        raise ValueError("The accepted base plan must use adaptation seed42")
    if (report["status"] != "declared_scopes_complete"
            or report["quantitative_complete_cells"] != 192):
        raise ValueError("Bind the completed 192-cell core report")
    out, artifacts = Path(out_dir).resolve(), Path(artifact_root).resolve()
    if out.exists() or artifacts.exists():
        raise ValueError("Metadata directory and formal artifact root must be NEW")
    if out == artifacts or out in artifacts.parents or artifacts in out.parents:
        raise ValueError("Metadata and formal artifacts must have separate roots")
    core_paths = Path(core_paths).resolve()
    paths = load_resolver(core_paths)
    lock = require_file(core_paths.parent / "with_gpu_lock.sh")
    # Inspect the shared owner's lock, not a private/generated substitute.
    if not any(line.startswith("LOCKDIR=") for line in lock.read_text().splitlines()):
        raise ValueError("Core queue wrapper must declare its shared LOCKDIR")
    eval_code = Path(eval_code).resolve()
    evaluation_sha = code_sha(eval_code)
    if evaluation_sha != EVALUATION_SHA:
        raise ValueError("Native inference must use the unchanged331d checkout")
    training_sha = code_sha(ROOT)
    python = str(Path(python).absolute())  # Keep a venv's interpreter prefix.
    require_file(python)
    require_file(ROOT / "train.py")
    require_file(eval_code / "test.py")
    sources = {(r["dataset"], r["domain"]): r for r in report["raw_results"]
               if r["method"] == "A" and int(r["seed"]) == 42}
    references = {(r["dataset"], r["domain"]): r for r in base["runs"]
                  if r["method"] == "A"}
    expected = {(ds, domain) for ds, domains in DOMAINS.items() for domain in domains}
    if set(sources) != expected:
        raise ValueError("Completed core A must bind exactly the existing12 domains")
    cells, overlays, regression_audits = {}, {}, {}
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            source, reference = sources[dataset, domain], references[dataset, domain]
            checkpoint = paths.ema_path(dataset, domain, "42", "A")
            if (source["status"] != "complete" or source["checkpoint"] != checkpoint
                    or reference["checkpoint"] != checkpoint or not source["source_id"]):
                raise ValueError(f"Completed source A identity mismatch: {dataset}/{domain}")
            require_file(checkpoint)
            eval_config = str(require_file(eval_code / source["config"]))
            val = target_val(dataset, domain, reference["img_prefix"])
            for seed in SEEDS:
                for method in methods:
                    root = artifacts / dataset.lower() / domain / f"seed_{seed}"
                    method_dir = root / "methods" / method
                    work = method_dir / "work"
                    iteration = FINAL_ITERATION[dataset]
                    key = cell_key(dataset, domain, seed, method)
                    cell_code, cell_sha = str(ROOT), training_sha
                    if method == "B_REG":
                        from experiments.comparison.b_regression import build_b_regression_spec

                        overrides = {
                            "data.samples_per_gpu": 32, "optimizer.lr": 0.02,
                            "model.cfg.strict_source_free": True, "model.cfg.weight_l": 0,
                            "model.cfg.weight_u": 1, "data.train.type": "StrictSourceFreeDOTADataset",
                            "data.train.img_prefix": val,
                            "data.train.unlabeled_epoch_size": TARGET_VAL_SIZE[dataset],
                            "corrupt": domain, "load_from": checkpoint, "model.ema_ckpt": checkpoint,
                            "seed": seed, "work_dir": str(work),
                            "checkpoint_config.max_keep_ckpts": 2, "checkpoint_config.save_last": True,
                        }
                        spec = build_b_regression_spec(paths, dataset, overrides)
                        name = f"b_reg_{dataset.lower()}.py"
                        overlays[name] = spec["overlay_text"]
                        config = str(out / name)
                        cell_code, cell_sha = spec["training_code"], spec["training_code_sha"]
                        regression_audits[key] = {
                            **{k: v for k, v in spec.items() if k != "overlay_text"},
                            "common_launch_overrides": overrides,
                        }
                    else:
                        config = str(require_file(
                            ROOT / "configs/unbiased_teacher/sfod/extensions"
                            / f"{method.lower()}_{dataset.lower()}.py"))
                    cells[key] = {
                        "dataset": dataset, "domain": domain, "seed": seed, "method": method,
                        "role": "ema", "source_checkpoint": checkpoint,
                        "source_id": source["source_id"], "source_seed": 42,
                        "config": config, "eval_config": eval_config,
                        "ann_file": reference["ann_file"], "img_prefix": reference["img_prefix"],
                        "target_val": val, "unlabeled_epoch_size": TARGET_VAL_SIZE[dataset],
                        "training_code": cell_code, "training_code_sha": cell_sha,
                        "root": str(root), "method_dir": str(method_dir), "work_dir": str(work),
                        "student_checkpoint": str(work / f"iter_{iteration}.pth"),
                        "checkpoint": str(work / f"iter_{iteration}_ema.pth"),
                        "eval_dir": str(method_dir / f"eval_full_{domain}_ids_v1"),
                        "terminal_status": str(method_dir / "terminal_status"),
                        "status": "prepared_not_execution_evidence",
                    }
    runtime = {
        "schema": "iraod-extension-training-v1", "status": "prepared_not_execution_evidence",
        "python": python, "training_code": str(ROOT), "training_code_sha": training_sha,
        "evaluation_code": str(eval_code), "evaluation_code_sha": evaluation_sha,
        "base_plan": str(Path(base_plan).resolve()), "core_report": str(Path(core_report).resolve()),
        "core_paths": str(core_paths), "artifact_root": str(artifacts),
        "methods": methods, "seeds": SEEDS, "domains": DOMAINS,
        "train_cells": len(cells), "eval_cells": len(cells),
        "source_training_cells": 0, "cells": cells,
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, text in overlays.items():
        (out / name).write_text(text)
    if regression_audits:
        write_json(out / "b_regression_config_diff.json", regression_audits)
    write_json(out / "runtime.json", runtime)
    write_json(out / "cells.json", cells)
    rows = "".join(f"{c['dataset']} {c['domain']} {c['seed']} {c['method']}\n"
                   for c in cells.values())
    for phase in ("train", "eval"):
        (out / f"{phase}.list").write_text(rows)
    (out / "paths.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "DATA = json.loads((Path(__file__).resolve().parent / 'runtime.json').read_text())\n"
        "EVALUATION_CODE_SHA = DATA['evaluation_code_sha']\n"
        f"EXPECT_PRED = {EXPECTED_IMAGES!r}\n"
        "def root(ds, domain, seed):\n"
        "    return str(Path(DATA['artifact_root']) / ds.lower() / domain / f'seed_{seed}')\n"
        "def binding(ds, domain, seed, method):\n"
        "    return DATA['cells'][f'{ds}/{domain}/{seed}/{method}']\n"
        "def method_dir(ds, domain, seed, method):\n"
        "    return binding(ds, domain, seed, method)['method_dir']\n"
        "def ema_path(ds, domain, seed, method):\n"
        "    return binding(ds, domain, seed, method)['checkpoint']\n"
        "def student_path(ds, domain, seed, method):\n"
        "    return binding(ds, domain, seed, method)['student_checkpoint']\n"
        "def eval_full_dir(ds, domain, seed, method):\n"
        "    return binding(ds, domain, seed, method)['eval_dir']\n")
    for action, filename in (("train", "run_train_1gpu.sh"), ("evaluate", "run_eval_full.sh")):
        script = out / filename
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "export PYTHONNOUSERSITE=1\n"
            f"export PYTHONPATH={shlex.quote(str(ROOT))}\n"
            f"exec {shlex.quote(python)} -m experiments.comparison.extension_training {action}"
            f" --queue {shlex.quote(str(out))}"
            ' --gpu "$1" --dataset "$2" --domain "$3" --seed "$4" --method "$5"\n')
        script.chmod(0o755)
    (out / "with_gpu_lock.sh").symlink_to(lock)
    return out


def load_cell(queue, dataset, domain, seed, method):
    runtime = read_json(Path(queue) / "runtime.json")
    return runtime, runtime["cells"][cell_key(dataset, domain, seed, method)]


def train(queue, gpu, dataset, domain, seed, method):
    if gpu not in (4, 5, 6, 7):
        raise ValueError("Training requires an approved GPU4-7")
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Invoke via the finite worker holding the shared GPU lock")
    runtime, cell = load_cell(queue, dataset, domain, seed, method)
    work = Path(cell["work_dir"])
    if work.exists():
        raise FileExistsError(f"Refusing existing work directory: {work}")
    source = str(require_file(cell["source_checkpoint"]))
    python = runtime["python"]
    code = Path(cell.get("training_code", runtime["training_code"]))
    producer_sha = code_sha(code)
    if method == "B_REG" and producer_sha != cell["training_code_sha"]:
        raise ValueError("B_REG frozen B training code changed after preparation")
    command = [
        python, str(code / "train.py"), cell["config"], "--work-dir", str(work),
        "--gpus", "1", "--seed", str(seed), "--deterministic", "--no-validate",
        "--cfg-options", "data.samples_per_gpu=32", "optimizer.lr=0.02",
        "model.cfg.strict_source_free=True", "model.cfg.weight_l=0", "model.cfg.weight_u=1",
        "model.cfg.use_bbox_reg=True", "data.train.type=StrictSourceFreeDOTADataset",
        "data.train.img_prefix=" + cell["target_val"],
        "data.train.unlabeled_epoch_size=" + str(cell["unlabeled_epoch_size"]),
        "corrupt=" + domain, "load_from=" + source, "model.ema_ckpt=" + source,
        "checkpoint_config.max_keep_ckpts=2", "checkpoint_config.save_last=True",
    ]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CGA_", "SARCLIP_", "VLST_"))}
    prefix = str(Path(python).parent.parent)
    env.update({
        "PYTHONPATH": str(code), "CUDA_VISIBLE_DEVICES": str(gpu),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1", "IRAOD_RUNTIME_READY": "1", "CONDA_PREFIX": prefix,
        "LD_LIBRARY_PATH": prefix + "/lib:" + env.get("LD_LIBRARY_PATH", ""),
    })
    if method == "B_REG":
        env.update(CGA_SCORER="none", CGA_BACKEND="none", CGA_FILTER_MODE="none",
                   PYTHONDONTWRITEBYTECODE="1")
    work.mkdir(parents=True, exist_ok=False)
    write_json(Path(cell["method_dir"]) / "execution.json", {
        **cell, "training_code": str(code), "training_code_sha": producer_sha,
        "orchestration_code": str(ROOT), "orchestration_code_sha": code_sha(ROOT),
        "command": command, "gpu": gpu, "status": "invoked_not_completion_evidence",
    })
    terminal = Path(cell["terminal_status"])
    rc = 1
    try:
        with (Path(cell["method_dir"]) / "train.log").open("w") as log:
            result = subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            rc = result.returncode
            raise subprocess.CalledProcessError(result.returncode, command)
        require_file(cell["student_checkpoint"])
        require_file(cell["checkpoint"])
        rc = 0
    finally:
        with terminal.open("a") as stream:
            stream.write(f"tmux_wrap_exit={rc}\n")


def evaluate(queue, gpu, dataset, domain, seed, method):
    runtime, cell = load_cell(queue, dataset, domain, seed, method)
    execution = read_json(Path(cell["method_dir"]) / "execution.json")
    binding = {
        **cell, "role": "ema", "checkpoint": cell["checkpoint"],
        "config": cell["eval_config"], "training_code_sha": execution["training_code_sha"],
    }
    return evaluate_binding(binding, runtime, gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("base-plan", "core-report", "core-paths", "out-dir", "artifact-root",
                 "eval-code", "python"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--method", dest="methods", action="append", choices=METHODS, required=True)
    for action in ("train", "evaluate"):
        entry = commands.add_parser(action)
        for name in ("queue", "dataset", "domain"):
            entry.add_argument("--" + name, required=True)
        entry.add_argument("--method", choices=METHODS, required=True)
        entry.add_argument("--gpu", type=int, required=True)
        entry.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"prepare": prepare, "train": train, "evaluate": evaluate}[command](**args)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
