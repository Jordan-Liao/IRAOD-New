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
STUDENT_METHODS = (*PORT_METHODS, "AASFOD", "SFYOLO")
F_DELETIONS = ("F_text_only", "F_veto_only")
METHODS = (*PORT_METHODS, "AASFOD", "SFYOLO", "B_REG", *F_DELETIONS)
ALLOWED_GPUS = (4, 5, 6, 7)
PAIR_PORTS = {(4, 5): 29804, (6, 7): 29806}
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
            eval_code, python, methods, sarclip_base=None, tam_plan=None):
    """Write only a NEW metadata queue; never create the formal output root."""
    if not methods or any(method not in METHODS for method in methods):
        raise ValueError("Explicitly select a supported port or approved ablation")
    if any(method in F_DELETIONS for method in methods) and not sarclip_base:
        raise ValueError("F deletions require the explicit frozen SARCLIP base")
    tam_fits = {}
    if "SFYOLO" in methods:
        if not tam_plan:
            raise ValueError("SFYOLO requires --tam-plan from prepare_tam")
        tam_fits = read_json(tam_plan)["fits"]
        expected_fits = {f"{ds}/{domain}" for ds, domains in DOMAINS.items() for domain in domains}
        if set(tam_fits) != expected_fits:
            raise ValueError("SFYOLO requires exactly12 domain TAM fits")
        for key, fit in tam_fits.items():
            ds, domain = key.split("/")
            if (fit["identity"] != {"dataset": ds, "domain": domain, "seed": 42}
                    or fit["outer_iterations"] != 160000):
                raise ValueError("TAM requires a160k outer-iteration domain fit at seed42")
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
    cells, overlays, regression_audits, f_audits = {}, {}, {}, {}
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
                    world_size = 2 if method in F_DELETIONS else 1
                    work = method_dir / "ddp2/work" if world_size == 2 else method_dir / "work"
                    epochs = 2 if method == "SFYOLO" else 1
                    # Single-group sampler pads to32. Epoch-end save is runner.iter+1,
                    # after all updates, not double the old one-epoch filename label.
                    iterations_per_epoch = (TARGET_VAL_SIZE[dataset] + 31) // 32
                    iteration = (epochs * iterations_per_epoch + 1 if method == "SFYOLO"
                                 else FINAL_ITERATION[dataset])
                    key = cell_key(dataset, domain, seed, method)
                    cell_code, cell_sha = str(ROOT), training_sha
                    model_environment, wrapper_environment = {}, {}
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
                    elif method in F_DELETIONS:
                        from experiments.comparison.f_deletion import build_f_deletion_spec

                        overrides = {
                            "data.samples_per_gpu": 16, "optimizer.lr": 0.02,
                            "find_unused_parameters": True,
                            "model.cfg.strict_source_free": True, "model.cfg.weight_l": 0,
                            "model.cfg.weight_u": 1, "data.train.type": "StrictSourceFreeDOTADataset",
                            "data.train.img_prefix": val,
                            "data.train.unlabeled_epoch_size": TARGET_VAL_SIZE[dataset],
                            "corrupt": domain, "load_from": checkpoint, "model.ema_ckpt": checkpoint,
                            "seed": seed, "work_dir": str(work),
                            "checkpoint_config.max_keep_ckpts": 2, "checkpoint_config.save_last": True,
                        }
                        spec = build_f_deletion_spec(paths, dataset, method, sarclip_base, overrides)
                        name = f"{method}_{dataset}_{domain}_{seed}.py"
                        overlays[name] = spec["config_text"]
                        config = str(out / name)
                        cell_code, cell_sha = spec["training_code"], spec["training_code_sha"]
                        model_environment = spec["effective_model_environment"]
                        wrapper_environment = spec["wrapper_environment"]
                        f_audits[key] = {k: v for k, v in spec.items() if k != "config_text"}
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
                        "world_size": world_size, "samples_per_gpu": 32 // world_size,
                        "use_bbox_reg": method not in F_DELETIONS,
                        "model_environment": model_environment,
                        "wrapper_environment": wrapper_environment,
                        "root": str(root), "method_dir": str(method_dir), "work_dir": str(work),
                        "student_checkpoint": str(work / f"iter_{iteration}.pth"),
                        "checkpoint": str(work / f"iter_{iteration}_ema.pth"),
                        "eval_dir": str(work.parent / f"eval_full_{domain}_ids_v1"),
                        "terminal_status": str(work.parent / "terminal_status"),
                        "status": "prepared_not_execution_evidence",
                    }
                    if method == "SFYOLO":
                        fit = tam_fits[f"{dataset}/{domain}"]
                        if fit["target_val"] != val:
                            raise ValueError("TAM and detector must use the same target VAL domain")
                        cells[key].update(
                            tam_checkpoint=fit["checkpoint"], tam_identity=fit["identity"],
                            detector_epochs=epochs, detector_optimizer_updates=epochs * iterations_per_epoch,
                            final_checkpoint_iteration=iteration,
                            budget_group="extended_two_epoch_TAM",
                            rank_with_common_one_epoch=False)
                    if method == "AASFOD":
                        from experiments.comparison.aasfod_protocol import budget, TSD_CHOICE

                        cells[key].update(
                            aasfod_budget=budget(TARGET_VAL_SIZE[dataset]),
                            tsd_choice=TSD_CHOICE, tsd_split=str(method_dir / "tsd.json"),
                            tsd_command=[
                                python, "-m", "experiments.comparison.aasfod_tsd",
                                "--queue", str(out), "--dataset", dataset,
                                "--domain", domain, "--seed", str(seed)],
                            detector_optimizer_updates=iterations_per_epoch,
                            budget_group="common_one_epoch_disclosed_AASFOD",
                            samples_per_gpu=16,
                            global_original_images_per_update=32)
    runtime = {
        "schema": "iraod-extension-training-v1", "status": "prepared_not_execution_evidence",
        "python": python, "training_code": str(ROOT), "training_code_sha": training_sha,
        "evaluation_code": str(eval_code), "evaluation_code_sha": evaluation_sha,
        "base_plan": str(Path(base_plan).resolve()), "core_report": str(Path(core_report).resolve()),
        "core_paths": str(core_paths), "artifact_root": str(artifacts),
        "methods": methods, "seeds": SEEDS, "domains": DOMAINS,
        "train_cells": len(cells), "eval_cells": len(cells),
        "source_training_cells": 0, "cells": cells,
        "allowed_gpus": list(ALLOWED_GPUS),
        "pair_ports": {",".join(map(str, pair)): port for pair, port in PAIR_PORTS.items()},
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, text in overlays.items():
        (out / name).write_text(text)
    if regression_audits:
        write_json(out / "b_regression_config_diff.json", regression_audits)
    if f_audits:
        write_json(out / "f_deletion_config_diff.json", f_audits)
    write_json(out / "runtime.json", runtime)
    write_json(out / "cells.json", cells)
    if "AASFOD" in methods:
        (out / "aasfod_tsd_commands.txt").write_text(
            "\n".join(shlex.join(c["tsd_command"]) for c in cells.values()
                      if c["method"] == "AASFOD") + "\n")
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
    write_runners(out, python, lock)
    return out


def write_runners(out, python, lock):
    """Use the current isolated executor, not wrappers in a live source queue."""
    for action, filename in (("train", "run_train_1gpu.sh"), ("evaluate", "run_eval_full.sh")):
        script = out / filename
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "export PYTHONNOUSERSITE=1\n"
            f"export PYTHONPATH={shlex.quote(str(ROOT))}\n"
            f"exec {shlex.quote(python)} -m experiments.comparison.extension_training {action}"
            f" --queue {shlex.quote(str(out))}"
            ' --gpu "$1" --dataset "$2" --domain "$3" --seed "$4" --method "$5"'
            + (' --role "${6:-ema}"' if action == "evaluate" else '') + '\n')
        script.chmod(0o755)
    ddp = out / "run_train_2gpu.sh"
    ddp.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(ROOT))}\n"
        f"exec {shlex.quote(python)} -m experiments.comparison.extension_training train-ddp"
        f" --queue {shlex.quote(str(out))}"
        ' --gpu-pair "$1" --port "$2" --dataset "$3" --domain "$4" --seed "$5" --method "$6"\n')
    ddp.chmod(0o755)
    (out / "with_gpu_lock.sh").symlink_to(lock)


def load_cell(queue, dataset, domain, seed, method, role="ema"):
    runtime = read_json(Path(queue) / "runtime.json")
    key = cell_key(dataset, domain, seed, method)
    cell = runtime["student_cells" if role == "student" else "cells"][key]
    route = key + ("/student" if role == "student" else "")
    while "source_queues" in runtime:
        origin = runtime["source_queues"].get(route, runtime["source_queues"][key])
        runtime = read_json(Path(origin) / "runtime.json")
    return runtime, cell


def require_prerequisites(cell):
    """The same scientific admission boundary for producer and native worker."""
    if cell["method"] == "AASFOD":
        from experiments.comparison.aasfod_protocol import validate_split

        validate_split(read_json(require_file(cell["tsd_split"])), cell)
    elif cell["method"] == "SFYOLO":
        from experiments.comparison.tam_artifacts import load_completed_tam

        expected = {"dataset": cell["dataset"], "domain": cell["domain"], "seed": 42}
        if cell["tam_identity"] != expected:
            raise ValueError("SFYOLO requires its matching domain TAM fit at seed42")
        # Validates actual160k completion, preprocessing, learned components and
        # strict external Oxford encoder tensors. Never runs a detector forward.
        load_completed_tam(require_file(cell["tam_checkpoint"]), expected, "cpu")


def train(queue, gpu, dataset, domain, seed, method):
    if gpu not in ALLOWED_GPUS:
        raise ValueError("Training requires an approved GPU4,5,6,7")
    return _train(queue, (gpu,), None, dataset, domain, seed, method)


def train_ddp(queue, gpu_pair, port, dataset, domain, seed, method):
    gpus = tuple(int(gpu) for gpu in gpu_pair.split(","))
    if PAIR_PORTS.get(gpus) != port:
        raise ValueError("DDP requires approved pair4,5/29804 or6,7/29806")
    return _train(queue, gpus, port, dataset, domain, seed, method)


def _train(queue, gpus, port, dataset, domain, seed, method):
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Invoke via the finite worker holding the shared GPU lock")
    runtime, cell = load_cell(queue, dataset, domain, seed, method)
    if len(gpus) != cell.get("world_size", 1):
        raise ValueError("GPU topology differs from the frozen cell binding")
    require_prerequisites(cell)
    work = Path(cell["work_dir"])
    if work.exists():
        raise FileExistsError(f"Refusing existing work directory: {work}")
    code, producer_sha, command, env = training_invocation(
        queue, runtime, cell, gpus, port, work)
    work.mkdir(parents=True, exist_ok=False)
    write_json(Path(cell["method_dir"]) / "execution.json", {
        **cell, "training_code": str(code), "training_code_sha": producer_sha,
        "orchestration_code": str(ROOT), "orchestration_code_sha": code_sha(ROOT),
        "command": command, "gpu": gpus[0] if len(gpus) == 1 else None,
        "gpus": list(gpus), "status": "invoked_not_completion_evidence",
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


def training_invocation(queue, runtime, cell, gpus, port, work):
    """One frozen native command/environment shared by formal and bounded smoke."""
    dataset, domain, seed, method = (cell[k] for k in ("dataset", "domain", "seed", "method"))
    source = str(require_file(cell["source_checkpoint"]))
    python = runtime["python"]
    code = Path(cell.get("training_code", runtime["training_code"]))
    producer_sha = code_sha(code)
    if method in ("B_REG", *F_DELETIONS) and producer_sha != cell["training_code_sha"]:
        raise ValueError("Frozen baseline training code changed after preparation")
    command = [python]
    if len(gpus) == 2:
        command += ["-m", "torch.distributed.launch", "--nproc_per_node=2", f"--master_port={port}"]
    command += [
        str(code / "train.py"), cell["config"], "--work-dir", str(work),
        *(["--gpus", "1"] if len(gpus) == 1 else ["--launcher", "pytorch"]),
        "--seed", str(seed), "--deterministic", "--no-validate",
        "--cfg-options", f"data.samples_per_gpu={32 // len(gpus)}", "optimizer.lr=0.02",
        "model.cfg.strict_source_free=True", "model.cfg.weight_l=0", "model.cfg.weight_u=1",
        f"model.cfg.use_bbox_reg={cell.get('use_bbox_reg', True)}",
        "data.train.type=StrictSourceFreeDOTADataset",
        "data.train.img_prefix=" + cell["target_val"],
        "data.train.unlabeled_epoch_size=" + str(cell["unlabeled_epoch_size"]),
        "corrupt=" + domain, "load_from=" + source, "model.ema_ckpt=" + source,
        "checkpoint_config.max_keep_ckpts=2", "checkpoint_config.save_last=True",
    ]
    if len(gpus) == 2:
        command.append("find_unused_parameters=True")
    if method == "SFYOLO":
        command += [
            "runner.max_epochs=2", "model.cfg.tam_checkpoint=" + cell["tam_checkpoint"],
            "model.cfg.tam_dataset=" + dataset, "model.cfg.tam_domain=" + domain,
            "model.cfg.tam_seed=42",
        ]
    if method == "AASFOD":
        command = [
            python, "-m", "experiments.comparison.train_aasfod",
            "--queue", str(Path(queue).resolve()), "--dataset", dataset,
            "--domain", domain, "--seed", str(seed)]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CGA_", "SARCLIP_", "VLST_"))}
    prefix = str(Path(python).parent.parent)
    env.update({
        "PYTHONPATH": str(code), "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1", "IRAOD_RUNTIME_READY": "1", "CONDA_PREFIX": prefix,
        "LD_LIBRARY_PATH": prefix + "/lib:" + env.get("LD_LIBRARY_PATH", ""),
    })
    if method == "B_REG":
        env.update(CGA_SCORER="none", CGA_BACKEND="none", CGA_FILTER_MODE="none",
                   PYTHONDONTWRITEBYTECODE="1")
    if method in F_DELETIONS:
        env.update(cell["model_environment"])
        env.update(cell["wrapper_environment"])
        env.update(MASTER_PORT=str(port), PYTHONDONTWRITEBYTECODE="1")
    return code, producer_sha, command, env


def evaluate(queue, gpu, dataset, domain, seed, method, role="ema"):
    if gpu not in ALLOWED_GPUS:
        raise ValueError("Evaluation requires an approved GPU4,5,6,7")
    if role not in ("ema", "student") or role == "student" and method not in STUDENT_METHODS:
        raise ValueError("Student native evaluations are limited to the five approved ports")
    runtime, cell = load_cell(queue, dataset, domain, seed, method, role)
    execution = read_json(Path(cell["method_dir"]) / "execution.json")
    binding = {
        **cell, "role": role, "checkpoint": cell["checkpoint"],
        "config": cell["eval_config"], "training_code_sha": execution["training_code_sha"],
        "training_code": execution["training_code"],
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
    prep.add_argument("--sarclip-base", help="Required frozen base SARCLIP for F deletions")
    prep.add_argument("--tam-plan", help="SFYOLO: prepare_tam runtime.json with12 seed42 fits")
    for action in ("train", "train-ddp", "evaluate"):
        entry = commands.add_parser(action)
        for name in ("queue", "dataset", "domain"):
            entry.add_argument("--" + name, required=True)
        entry.add_argument("--method", choices=METHODS, required=True)
        if action == "evaluate":
            entry.add_argument("--role", choices=("ema", "student"), default="ema")
        if action == "train-ddp":
            entry.add_argument("--gpu-pair", required=True)
            entry.add_argument("--port", required=True, type=int)
        else:
            entry.add_argument("--gpu", type=int, required=True)
        entry.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = {"prepare": prepare, "train": train, "train-ddp": train_ddp,
              "evaluate": evaluate}[command](**args)
    if result is not None:
        print(result)


if __name__ == "__main__":
    main()
