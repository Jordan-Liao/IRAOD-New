"""Diagnostic-only native training window; no tensor repair or CPU offload."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from unittest.mock import patch


SOURCE_SHA = "d668a7719833ea59c6320380b6bc4f80a59b5399"
STOP_AFTER = 130


class WindowComplete(RuntimeError):
    pass


@contextmanager
def stop_after_iteration(runner_class, emit):
    original = runner_class.call_hook

    def call_hook(runner, name):
        result = original(runner, name)
        if name == "after_train_iter":
            completed = runner.iter + 1
            emit({"event": "iteration_hooks_complete", "completed_iterations": completed})
            if completed >= STOP_AFTER:
                raise WindowComplete()
        return result

    with patch.object(runner_class, "call_hook", call_hook):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", required=True, help="Original frozen d668 checkout")
    parser.add_argument("train_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    code = Path(args.code_root).resolve()
    arguments = args.train_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    sys.argv = [str(Path(__file__).resolve()), "--code-root", str(code), "--", *arguments]
    os.chdir(code)
    sys.path.insert(0, str(code))
    os.environ["PYTHONPATH"] = str(code)
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()

    actual_sha = subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if actual_sha != SOURCE_SHA:
        raise ValueError("Use the original d668 scientific checkout, not the adapter checkout")
    work = Path(arguments[arguments.index("--work-dir") + 1])
    if work.exists():
        raise FileExistsError(f"Use a fresh diagnostic work directory: {work}")
    from experiments.comparison.numerical_integrity import native_boundaries
    from mmcv.runner import BaseRunner
    started = time.monotonic()

    def emit(record):
        print("NONFINITE_WINDOW " + json.dumps({
            **record, "elapsed_seconds": time.monotonic() - started,
            "diagnostic_only": True,
        }), flush=True)

    emit({"event": "start", "source_sha": actual_sha, "stop_after": STOP_AFTER,
          "save_on_cpu_enabled": False, "anomaly_activation_by_adapter": False})
    sys.argv = [str(code / "train.py"), *arguments]
    try:
        with native_boundaries(evaluate=False), stop_after_iteration(BaseRunner, emit):
            runpy.run_path(sys.argv[0], run_name="__main__")
    except WindowComplete:
        emit({"event": "window_complete_no_nonfinite", "completed_iterations": STOP_AFTER,
              "status": "INSUFFICIENT_EVIDENCE_NO_FAILURE_IN_WINDOW"})
        raise SystemExit(3)
    emit({"event": "native_returned", "status": "INSUFFICIENT_EVIDENCE_NO_FAILURE_RECORDED"})
    raise SystemExit(3)


if __name__ == "__main__":
    main()
