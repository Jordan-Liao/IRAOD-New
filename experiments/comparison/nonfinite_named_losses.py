"""Failure-only named-loss observation; original a39 forward/parser remain intact."""

import argparse
from contextlib import contextmanager, ExitStack
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
from unittest.mock import patch


def bootstrap(code, helper, epoch_adapter, failure_dir, arguments):
    code, helper = Path(code).resolve(), Path(helper).resolve()
    epoch_adapter, failure_dir = Path(epoch_adapter).resolve(), Path(failure_dir).resolve()
    sys.argv = [
        str(Path(__file__).resolve()), "--code-root", str(code),
        "--finite-helper", str(helper), "--epoch-adapter", str(epoch_adapter),
        "--failure-dir", str(failure_dir), "--", *arguments,
    ]
    os.chdir(code)
    sys.path.insert(0, str(code))
    os.environ["PYTHONPATH"] = str(code)
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()


def rng_snapshot():
    import numpy as np
    import torch

    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def tensor_metadata(name, value):
    if isinstance(value, list):
        return [item for index, tensor in enumerate(value)
                for item in tensor_metadata(f"{name}[{index}]", tensor)]
    import torch

    tensor = value.detach()
    return [{
        "name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "scalar_repr": repr(tensor.item()) if tensor.numel() == 1 else None,
        "is_nan": bool(torch.isnan(tensor).any().item()),
        "is_inf": bool(torch.isinf(tensor).any().item()),
    }]


class NamedLossProbe:
    def __init__(self, failure_dir, require_finite, save_failure_batch):
        self.failure_dir = Path(failure_dir)
        self.require_finite = require_finite
        self.save_failure_batch = save_failure_batch
        self.context = None
        self.original_parser = None
        self.failed = False

    def persist_failure(self, stage, losses, trigger, parsed=None):
        if self.failed:
            raise RuntimeError("The first loss failure was already captured; no continuation is allowed")
        self.failed = True
        if self.context is None:
            raise RuntimeError("Failure has no current training batch/pre-forward RNG context")
        named = [(key, value) for key, value in losses.items() if "loss" in key]
        raw = [item for name, value in named for item in tensor_metadata(name, value)]
        log_vars = [] if parsed is None else [
            {"name": name, "dtype": type(value).__name__, "shape": [], "numel": 1,
             "scalar_repr": repr(value), "is_nan": math.isnan(float(value)),
             "is_inf": math.isinf(float(value))}
            for name, value in parsed[1].items() if "loss" in name
        ]
        aggregate = [] if parsed is None else tensor_metadata("aggregate", parsed[0])
        record = {
            "schema": "iraod-named-loss-failure-v1", "diagnostic_only": True,
            "stage": stage, "runner_iter_zero_based": self.context["iteration"],
            "update_one_based": self.context["iteration"] + 1,
            "ordered_loss_names": [name for name, _ in named],
            "raw_named_values": raw, "native_loss_log_vars": log_vars,
            "aggregate": aggregate, "trigger": trigger,
            "first_observed_nonfinite_name": next(
                item["name"] for item in raw + log_vars + aggregate
                if item["is_nan"] or item["is_inf"]),
            "parser_source": self.original_parser.__func__.__code__.co_filename,
            "batch_keys": list(self.context["batch"]),
            "batch_rng_file": "failure_batch_rng.pt",
            "capture_status": "pending",
            "not_a_model_or_resume_checkpoint": True,
        }
        self.failure_dir.mkdir(parents=True, exist_ok=False)
        metadata = self.failure_dir / "failure.json"
        metadata.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary = self.failure_dir / "failure_batch_rng.pt.partial"
        # This serializer is private to diagnostic inputs. Native checkpoint
        # writes still use the installed finite-checked torch.save.
        self.save_failure_batch({
            "schema": "iraod-failure-batch-rng-v1", "diagnostic_only": True,
            "not_a_model_or_resume_checkpoint": True,
            "batch": self.context["batch"], "pre_forward_rng": self.context["rng"],
        }, temporary)
        temporary.replace(self.failure_dir / "failure_batch_rng.pt")
        record["capture_status"] = "complete"
        metadata.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        print("NONFINITE_LOSS_FAILURE " + json.dumps(record, allow_nan=False), flush=True)

    def checked_parse(self, losses):
        named = {key: value for key, value in losses.items() if "loss" in key}
        try:
            self.require_finite(named, "named losses after forward")
        except FloatingPointError as error:
            self.persist_failure("named_loss_before_aggregate", losses, str(error))
            raise
        result = self.original_parser(losses)
        try:
            self.require_finite(result[0], "aggregate after native parser")
        except FloatingPointError as error:
            self.persist_failure("aggregate_only", losses, str(error), result)
            raise
        bad_logs = [name for name, value in result[1].items()
                    if "loss" in name and not math.isfinite(float(value))]
        if bad_logs:
            error = FloatingPointError(f"Nonfinite native loss log_vars: {bad_logs}")
            self.persist_failure("aggregate_only", losses, str(error), result)
            raise error
        return result

    @contextmanager
    def install(self, runner_class):
        from mmcv.parallel import is_module_wrapper

        original_hook = runner_class.call_hook
        with ExitStack() as stack:
            def call_hook(runner, name):
                result = original_hook(runner, name)
                if name == "before_run":
                    model = runner.model.module if is_module_wrapper(runner.model) else runner.model
                    self.original_parser = model._parse_losses
                    stack.enter_context(patch.object(model, "_parse_losses", self.checked_parse))
                elif name == "before_train_iter":
                    self.context = {
                        "iteration": runner.iter, "batch": runner.data_batch,
                        "rng": rng_snapshot(),
                    }
                elif name == "after_train_iter":
                    self.context = None
                return result

            stack.enter_context(patch.object(runner_class, "call_hook", call_hook))
            try:
                yield self
            finally:
                self.context = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("code-root", "finite-helper", "epoch-adapter", "failure-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("train_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    code, helper = Path(args.code_root).resolve(), Path(args.finite_helper).resolve()
    epoch_path, failure_dir = Path(args.epoch_adapter).resolve(), Path(args.failure_dir).resolve()
    arguments = args.train_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    bootstrap(code, helper, epoch_path, failure_dir, arguments)
    if failure_dir.exists():
        raise FileExistsError(f"Use a fresh failure-only directory: {failure_dir}")
    spec = importlib.util.spec_from_file_location("_attempt7_epoch_adapter", epoch_path)
    epoch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(epoch)
    checks = epoch.load_finite_helper(helper)
    import torch

    probe = NamedLossProbe(failure_dir, checks.require_finite, torch.save)
    original_observer = epoch.observe_natural_epoch

    @contextmanager
    def observe_with_losses(runner_class, emit):
        with original_observer(runner_class, emit) as state, probe.install(runner_class):
            yield state

    with patch.object(epoch, "load_finite_helper", return_value=checks), \
            patch.object(epoch, "observe_natural_epoch", observe_with_losses), \
            patch.object(sys, "argv", [
                str(epoch_path), "--code-root", str(code), "--finite-helper", str(helper),
                "--", *arguments]):
        epoch.main()


if __name__ == "__main__":
    main()
