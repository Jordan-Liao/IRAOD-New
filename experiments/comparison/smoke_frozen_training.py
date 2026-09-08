"""Bounded NON_RESULT B_REG/F execution using the prepared frozen train.py."""

import argparse
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys


class SmokeFinished(Exception):
    """Stop after a real optimizer update, without epoch/final checkpoint hooks."""


class SmokeEvidence:
    """Mixin kept independent of native imports for CPU optimizer-hook tests."""

    def __init__(self, updates, output):
        self.updates, self.output = updates, Path(output)
        self.steps = 0
        self.losses = []

    def before_run(self, runner):
        self.model = getattr(runner.model, "module", runner.model)
        self.teacher = getattr(self.model.ema_model, "module", self.model.ema_model)
        self.parameters = [p for group in runner.optimizer.param_groups for p in group["params"]
                           if p.requires_grad]
        self.initial = [p.detach().cpu().clone() for p in self.parameters]
        self.ema_initial = [p.detach().cpu().clone() for p in self.teacher.parameters()]
        original_step = runner.optimizer.step

        def step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.steps += 1
            return result

        runner.optimizer.step = step

    def after_train_iter(self, runner):
        loss = float(runner.outputs["loss"].detach().cpu())
        if not math.isfinite(loss):
            raise FloatingPointError("NON_RESULT smoke observed a nonfinite native loss")
        self.losses.append({
            "iteration": runner.iter, "optimizer_updates": self.steps, "loss": loss,
            "log_vars": {key: float(value) for key, value in runner.outputs["log_vars"].items()},
        })
        if self.steps < self.updates:
            return
        delta = sum(float((p.detach().cpu() - before).abs().sum())
                    for p, before in zip(self.parameters, self.initial))
        ema_delta = sum(float((p.detach().cpu() - before).abs().sum())
                        for p, before in zip(self.teacher.parameters(), self.ema_initial))
        evidence = {
            "status": "NON_RESULT", "optimizer_updates": self.steps,
            "losses": self.losses, "student_parameter_l1_delta": delta,
            "ema_parameter_l1_delta": ema_delta,
            "ema_timing": "unchanged frozen iteration-start .998; one update can have zero EMA delta",
            "bounded_pass": self.steps == self.updates and math.isfinite(delta) and delta > 0
                            and math.isfinite(ema_delta),
        }
        self.output.write_text(json.dumps(evidence, indent=2) + "\n")
        if not evidence["bounded_pass"]:
            raise RuntimeError("NON_RESULT smoke did not prove the bounded parameter update")
        raise SmokeFinished()


def native(argv):
    """Inject only observation/termination; all detector imports resolve in0f98."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--frozen-train", required=True)
    parser.add_argument("--updates", required=True, type=int)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args, training_args = parser.parse_known_args(argv)
    if os.environ.get("IRAOD_GPU_LOCKED") != "1" or not 1 <= args.updates <= 4:
        raise ValueError("Native NON_RESULT entry requires held shared locks and1-4 updates")
    sys.path.insert(0, str(Path(args.frozen_train).parent))
    from mmcv.runner import HOOKS, Hook
    import mmdet_extension.apis

    @HOOKS.register_module()
    class FrozenSmokeHook(SmokeEvidence, Hook):
        pass

    original = mmdet_extension.apis.train_detector

    def bounded_train(model, datasets, cfg, **kwargs):
        # Do not shorten the epoch or LR schedule, change the sampler, or flush EMA.
        cfg.checkpoint_config = None
        cfg.custom_hooks = [*cfg.get("custom_hooks", []), dict(
            type="FrozenSmokeHook", priority="LOWEST", updates=args.updates,
            output=str(Path(args.evidence_dir) / f"rank_{os.environ.get('RANK', '0')}.json"))]
        return original(model, datasets, cfg, **kwargs)

    mmdet_extension.apis.train_detector = bounded_train
    sys.argv = [args.frozen_train, *training_args, f"--local-rank={args.local_rank}"]
    try:
        runpy.run_path(args.frozen_train, run_name="__main__")
    except SmokeFinished:
        pass
    else:
        raise RuntimeError("Native training returned before bounded smoke termination")


def run(queue, dataset, domain, seed, method, gpus, out_dir, updates=2):
    from experiments.comparison.extension_training import (
        F_DELETIONS, load_cell, training_invocation)
    from experiments.comparison.b_regression import TRAINING_CODE_SHA
    from experiments.comparison.finite_resumer import Cell, idle_devices, lock_root, take_lock
    from experiments.comparison.result_completion import write_json

    if method not in ("B_REG", *F_DELETIONS) or not 1 <= updates <= 4:
        raise ValueError("Select B_REG or one F deletion and1-4 optimizer updates")
    devices = tuple(map(int, gpus.split(",")))
    if ((method == "B_REG" and devices not in ((4,), (5,), (6,)))
            or (method in F_DELETIONS and devices != (4, 5))):
        raise ValueError("B_REG requires one GPU4/5/6; F requires pair4,5/29804")
    runtime, cell = load_cell(queue, dataset, domain, seed, method)
    if cell["training_code_sha"] != TRAINING_CODE_SHA or cell["world_size"] != len(devices):
        raise ValueError("Smoke requires the original0f98 binding and unchanged topology")
    out = Path(out_dir).resolve()
    formal = Path(cell["method_dir"]).resolve()
    if out == formal or formal in out.parents or out in formal.parents:
        raise ValueError("NON_RESULT outputs must be separate from the formal cell")
    if out.exists():
        raise FileExistsError(f"NON_RESULT output must be NEW: {out}")
    code, sha, command, env = training_invocation(
        queue, runtime, cell, devices, 29804 if len(devices) == 2 else None, out / "work")
    train_index = command.index(str(code / "train.py"))
    command[train_index:train_index + 1] = [
        str(Path(__file__).resolve()), "--native", "--frozen-train", str(code / "train.py"),
        "--updates", str(updates), "--evidence-dir", str(out)]
    # The only config deviation is disabling checkpoint output, also enforced at
    # the actual native train_detector seam. No terminal_status is ever written.
    start = command.index("--cfg-options")
    command = [part for i, part in enumerate(command)
               if i <= start or not part.startswith("checkpoint_config.")]
    command.append("checkpoint_config=None")
    root = lock_root(queue)
    locks = []
    try:
        cell_lock = root.parent / "cell_locks" / (Cell(dataset, domain, seed, method).session("train") + ".lock")
        for path in (cell_lock, *(root / f"gpu{g}.lock" for g in devices)):
            lock = take_lock(path)
            if lock is None:
                raise RuntimeError(f"Actual shared lock busy: {path}")
            locks.append(lock)
        if not set(devices).issubset(idle_devices(devices)):
            raise RuntimeError("Assigned GPU occupied; no smoke runner invoked")
        out.mkdir(parents=True, exist_ok=False)
        env["IRAOD_GPU_LOCKED"] = "1"
        write_json(out / "invocation.json", {
            "status": "NON_RESULT", "cell": cell, "training_code_sha": sha,
            "gpus": devices, "port": 29804 if len(devices) == 2 else None,
            "effective_global_batch": 32, "optimizer_updates": updates,
            "command": command, "formal_outputs_written": False,
        })
        with (out / "train.log").open("x") as log:
            subprocess.run(command, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        for rank in range(len(devices)):
            evidence = json.loads((out / f"rank_{rank}.json").read_text())
            if not evidence["bounded_pass"] or evidence["optimizer_updates"] != updates:
                raise RuntimeError(f"Rank{rank} did not prove bounded NON_RESULT execution")
        return out
    finally:
        for lock in reversed(locks):
            lock.close()


def main():
    if "--native" in sys.argv:
        native(sys.argv[1:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "dataset", "domain", "method", "gpus", "out-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--updates", type=int, default=2)
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
