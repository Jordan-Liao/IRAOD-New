"""Run two fixed-budget AASFOD stages using fresh native runners/optimizers."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

# This helper is invoked by absolute path with PYTHONPATH pointing at bound model
# code. Keep orchestration (including the native shim) current; run() restores the
# bound model path before configs are loaded and native subprocesses are launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.comparison.aasfod_protocol import validate_split
from experiments.comparison.extension_training import load_cell, require_file
from experiments.comparison.result_completion import write_json
from experiments.comparison import host_binding as host


def stage_specs(cell, work, smoke_steps=None):
    """Pure CPU interface shared by execution and tests; no hidden extra epoch."""
    plan = cell["aasfod_budget"]
    if smoke_steps is not None and not 1 <= smoke_steps <= 4:
        raise ValueError("AASFOD smoke allows one to four optimizer updates per stage")
    previous = cell["source_checkpoint"]
    specs = []
    for stage in ("alignment", "fns"):
        steps = plan[stage + "_updates"] if smoke_steps is None else smoke_steps
        directory = Path(work) / stage
        specs.append(dict(
            stage=stage, updates=steps, work_dir=str(directory),
            load_from=previous, teacher_initialization=previous,
            samples_per_gpu=16 if stage == "alignment" else 32,
            optimizer=dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001),
            # Preserve the common100-update linear warmup once, not once per stage.
            warmup_iters=100 if stage == "alignment" else 0,
            ema_interval=plan["ema_interval"],
            student_checkpoint=str(directory / f"iter_{steps}.pth"),
            teacher_checkpoint=str(directory / f"iter_{steps}_ema.pth")))
        previous = specs[-1]["student_checkpoint"]
    return specs


def run(queue, dataset, domain, seed, smoke_steps=None):
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Use the existing owner's shared GPU lock")
    runtime, cell = load_cell(queue, dataset, domain, seed, "AASFOD")
    validate_split(host.read_json(require_file(cell["tsd_split"])), cell)
    code = Path(cell.get("training_code", runtime["training_code"]))
    if host.is_target_host():
        sys.path.insert(0, str(code))
        os.chdir(code)
    work = Path(cell["work_dir"])
    if smoke_steps is not None:
        work = work.with_name("smoke_work")
        if work.exists():
            raise FileExistsError(work)
    from mmcv import Config

    specs = stage_specs(cell, work, smoke_steps)
    work.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        directory = Path(spec["work_dir"])
        if directory.exists():
            raise FileExistsError(directory)
        with host.native_config_paths():
            cfg = Config.fromfile(cell["config"])
        cfg.model.cfg.update(aasfod_stage=spec["stage"],
                             aasfod_ema_interval=spec["ema_interval"])
        # Runner loads the student; hook copies its entire detector into teacher.
        cfg.model.ema_ckpt = spec["teacher_initialization"]
        cfg.load_from = spec["load_from"]
        cfg.resume_from = None
        cfg.corrupt = domain
        cfg.data.samples_per_gpu = spec["samples_per_gpu"]
        cfg.data.train.update(stage=spec["stage"], tsd_split=cell["tsd_split"],
                              img_prefix=cell["target_val"],
                              unlabeled_epoch_size=cell["unlabeled_epoch_size"])
        cfg.runner = dict(type="SemiIterBasedRunner", max_iters=spec["updates"])
        cfg.optimizer = spec["optimizer"]
        cfg.lr_config = dict(policy="Fixed", by_epoch=False)
        if spec["warmup_iters"]:
            cfg.lr_config.update(warmup="linear", warmup_iters=spec["warmup_iters"],
                                 warmup_ratio=0.001)
        cfg.checkpoint_config = dict(by_epoch=False, interval=spec["updates"],
                                     save_last=True, max_keep_ckpts=2)
        cfg.work_dir = str(directory)
        config_path = work / (spec["stage"] + ".py")
        # MMCV renders transform instances as bare constructors, without imports
        # (same resolved-config boundary as f_deletion._resolved_text).
        config_path.write_text(
            "from torchvision.transforms import ColorJitter\n"
            + cfg.pretty_text
            + "\ndel ColorJitter\n")
        command = [runtime["python"], str(code / "train.py"), str(config_path),
                   "--work-dir", str(directory), "--gpus", "1", "--seed", str(seed),
                   "--deterministic", "--no-validate"]
        command = host.native_command(command, code / "train.py")
        spec["command"] = command
        write_json(work / "stages.json", {
            "status": "invoked_not_completion_evidence", "budget": cell["aasfod_budget"],
            "smoke_steps": smoke_steps, "stages": specs})
        subprocess.run(command, cwd=code, env={**os.environ, "PYTHONPATH": str(code)}, check=True)
        require_file(spec["student_checkpoint"])
        require_file(spec["teacher_checkpoint"])
    if smoke_steps is None:
        # Preserve the common report's epoch-end filename convention. Stage-local
        # native checkpoints record their own correct iter counts; stages.json
        # records total updates rather than pretending FNS itself took a full epoch.
        shutil.copyfile(specs[-1]["student_checkpoint"], cell["student_checkpoint"])
        shutil.copyfile(specs[-1]["teacher_checkpoint"], cell["checkpoint"])
    write_json(work / "stages.json", {
        "status": "complete" if smoke_steps is None else "smoke_not_formal",
        "budget": cell["aasfod_budget"], "smoke_steps": smoke_steps, "stages": specs})
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "dataset", "domain"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--smoke-steps", type=int)
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
