"""Observe the original a39 LPLD epoch with separately supplied finite checks."""

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from unittest.mock import patch


SOURCE_SHA = "a39c83221c73e71d273b96ef95c65f6d8d1e84f9"
UPDATES = 265


def bootstrap(code, helper, arguments):
    code, helper = Path(code).resolve(), Path(helper).resolve()
    sys.argv = [str(Path(__file__).resolve()), "--code-root", str(code),
                "--finite-helper", str(helper), "--", *arguments]
    os.chdir(code)
    sys.path.insert(0, str(code))
    os.environ["PYTHONPATH"] = str(code)
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()


def load_finite_helper(path):
    spec = importlib.util.spec_from_file_location("_attempt6_finite_checks", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def observe_natural_epoch(runner_class, emit):
    original = runner_class.call_hook
    state = {"completed_iterations": 0, "completed_epoch_hooks": 0, "after_run": False}

    def call_hook(runner, name):
        if name == "before_train_epoch":
            if runner.max_epochs != 1 or len(runner.data_loader) != UPDATES:
                raise ValueError("Attempt6 requires the original one-epoch265-update budget")
        result = original(runner, name)
        if name == "after_train_iter":
            state["completed_iterations"] = runner.iter + 1
            emit({"event": "iteration_hooks_complete",
                  "completed_iterations": state["completed_iterations"]})
        elif name == "after_train_epoch":
            state["completed_epoch_hooks"] += 1
            emit({"event": "epoch_hooks_complete", **state})
        elif name == "after_run":
            state["after_run"] = True
        return result

    with patch.object(runner_class, "call_hook", call_hook):
        yield state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, help="Original frozen a39 checkout")
    parser.add_argument("--finite-helper", required=True, help="Separate unchanged finite-check file")
    parser.add_argument("train_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    code, helper_path = Path(args.code_root).resolve(), Path(args.finite_helper).resolve()
    arguments = args.train_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    bootstrap(code, helper_path, arguments)
    actual_sha = subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if actual_sha != SOURCE_SHA:
        raise ValueError("Use original a39 science, not the adapter/helper checkout")
    work = Path(arguments[arguments.index("--work-dir") + 1])
    if work.exists():
        raise FileExistsError(f"Use a fresh diagnostic work directory: {work}")
    checks = load_finite_helper(helper_path)
    from mmcv.runner import BaseRunner
    started = time.monotonic()

    def emit(record):
        print("NONFINITE_EPOCH " + json.dumps({
            **record, "elapsed_seconds": time.monotonic() - started, "diagnostic_only": True,
        }), flush=True)

    emit({"event": "start", "source_sha": actual_sha, "finite_helper": str(helper_path),
          "save_on_cpu_enabled": False, "anomaly_activation_by_adapter": False})
    sys.argv = [str(code / "train.py"), *arguments]
    with checks.native_boundaries(evaluate=False), observe_natural_epoch(BaseRunner, emit) as state:
        runpy.run_path(sys.argv[0], run_name="__main__")
    if state != {"completed_iterations": UPDATES, "completed_epoch_hooks": 1, "after_run": True}:
        raise ValueError(f"Native training did not complete the prescribed epoch and hooks: {state}")
    emit({"event": "natural_epoch_complete_no_nonfinite", **state,
          "expected_terminal_checkpoint_iteration": 266,
          "causal_package_qualified": False,
          "causal_status": "INSUFFICIENT_EVIDENCE_NO_FAILURE_REPRODUCED"})


if __name__ == "__main__":
    main()
