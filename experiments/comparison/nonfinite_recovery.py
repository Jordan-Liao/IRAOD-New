"""One explicitly bound recovery run, with native finite gates and final-pair validation."""

from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time
from unittest.mock import patch


def load_checks(path):
    spec = importlib.util.spec_from_file_location("_recovery_finite_checks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def observe_budget(runner_class, expected_epochs, state):
    original = runner_class.call_hook

    def call_hook(runner, name):
        if name == "before_train_epoch":
            if runner.max_epochs != expected_epochs or len(runner.data_loader) != 265:
                raise ValueError("Recovery must preserve the prescribed epochs and265 updates/epoch")
        result = original(runner, name)
        if name == "after_train_iter":
            state["completed_updates"] = runner.iter + 1
        elif name == "after_train_epoch":
            state["completed_epochs"] += 1
        elif name == "after_run":
            state["after_run"] = True
        return result

    with patch.object(runner_class, "call_hook", call_hook):
        yield


def validate_final_pair(case, checks):
    import torch

    evidence = {}
    for role, filename in case["final_pair"].items():
        path = Path(case["work_dir"]) / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not payload.get("state_dict") or not payload.get("optimizer", {}).get("state"):
            raise ValueError(f"Missing model/optimizer state in final {role} checkpoint")
        meta = payload.get("meta", {})
        if meta.get("iter") != case["terminal_iteration"] or meta.get("epoch") != case["epochs"]:
            raise ValueError(f"Final {role} checkpoint does not have the prescribed budget")
        checks.require_finite(payload, f"written final {role} checkpoint")
        groups = payload["optimizer"].get("param_groups")
        if not groups:
            raise ValueError(f"Missing optimizer parameter groups in {role}")
        for group in groups:
            if any(isinstance(value, float) and not math.isfinite(value) for value in group.values()):
                raise FloatingPointError(f"Nonfinite optimizer scalar in {role}")
        evidence[role] = {
            "path": str(path), "bytes": path.stat().st_size,
            "epoch": meta["epoch"], "iteration": meta["iter"],
            "model_tensor_count": sum(1 for _ in checks._tensors(payload["state_dict"], "state_dict")),
            "optimizer_tensor_count": sum(1 for _ in checks._tensors(payload["optimizer"], "optimizer")),
            "all_checkpoint_tensors_finite": True, "optimizer_scalars_finite": True,
        }
        del payload
    if set(evidence) != {"student", "ema"}:
        raise ValueError("Recovery requires exactly the prescribed student and EMA final pair")
    return evidence


def main():
    case_path = Path(sys.argv[1]).resolve()
    gpu = int(sys.argv[2])
    case = json.loads(case_path.read_text())
    code = Path(case["code_root"]).resolve()
    sys.argv = [str(Path(__file__).resolve()), str(case_path), str(gpu)]
    os.chdir(code)
    sys.path.insert(0, str(code))
    os.environ["PYTHONPATH"] = str(code)
    os.environ["IRAOD_CONDA_PREFIX"] = str(Path(case["python"]).parent.parent)
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()
    if (socket.gethostname() != case["hostname"] or gpu not in case["allowed_gpus"]
            or os.environ.get("IRAOD_GPU_LOCKED") != "1"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu)):
        raise ValueError("Use the operator's exclusively admitted approved host/GPU")
    actual_sha = subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if actual_sha != case["patched_source_sha"]:
        raise ValueError("Recovery scientific checkout differs from the accepted causal patch")
    root = Path(case["case_root"])
    root.mkdir(parents=True, exist_ok=False)
    if Path(case["work_dir"]).exists():
        raise FileExistsError(case["work_dir"])
    (root / "execution.json").write_text(json.dumps({
        **case, "actual_code_sha": actual_sha, "physical_gpu": gpu,
        "entry": str(Path(__file__).resolve()),
    }, indent=2) + "\n")
    checks = load_checks(case["finite_helper"])
    from mmcv.runner import BaseRunner
    state = {"completed_updates": 0, "completed_epochs": 0, "after_run": False}
    started = time.monotonic()
    sys.argv = case["train_argv"]
    with checks.native_boundaries(evaluate=False), observe_budget(BaseRunner, case["epochs"], state):
        runpy.run_path(sys.argv[0], run_name="__main__")
    expected = {"completed_updates": case["updates"], "completed_epochs": case["epochs"], "after_run": True}
    if state != expected:
        raise ValueError(f"Native recovery budget/hook completion differs: {state}")
    pair = validate_final_pair(case, checks)
    result = {
        "status": "FINITE_FINAL_PAIR_VERIFIED", "cell": case["cell"],
        "base_source_sha": case["base_source_sha"], "patched_source_sha": actual_sha,
        **state, "final_pair": pair,
        "training_and_validation_seconds": time.monotonic() - started,
        "originals_preserved": True, "automatic_retry": False,
    }
    with (root / "recovery_result.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
