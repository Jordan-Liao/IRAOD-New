"""One NEW authorized first-image capture around an immutable ROI consumer.

Invoke this file by absolute path, not ``-m``: the scientific package must load
from spec.consumer_checkout, not this instrumentation checkout. The operator
owns the single launch slot, GPU lock and <=300-second cap including teardown.
No capture outcome is a completed ROI export. Match exits3; mismatch retains
the original exception and exits1. There is no retry or second-image path.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from unittest.mock import patch


class FirstImageCaptured(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_head(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def select_run(spec, completion):
    run = completion.load_run(spec["plan"], spec["run_id"])
    expected = spec["identity"]
    if any(run[key] != value for key, value in expected.items()):
        raise ValueError("Capture tuple differs from the authorized first-image case")
    if run["export_code_sha"] != spec["consumer_sha"]:
        raise ValueError("Capture plan does not bind the frozen consumer revision")
    if run["image_ids"][0] != spec["image_id"]:
        raise ValueError("Authorized image is not the unchanged consumer's first image")
    root = Path(run["native_prediction"]["eval_dir"])
    sidecar = completion.read_json(root / "predictions.pkl.image_ids.json")
    index = spec["native_prediction_index"]
    if sidecar["image_ids"].index(spec["image_id"]) != index:
        raise ValueError("Native first-image prediction index differs from authorization")
    image = Path(spec["image_path"])
    if image.stem != spec["image_id"] or not image.parent.samefile(run["img_prefix"]):
        raise ValueError("Capture image path is outside the selected run's image root")
    if (image.stat().st_size != spec["image_bytes"]
            or sha256(spec["image_path"]) != spec["image_sha256"]):
        raise ValueError("The authorized first-image file identity changed")
    return run


class OperandCapture:
    def __init__(self, spec, run, directory):
        self.spec, self.run = spec, run
        self.directory = Path(directory)
        self.calls = 0
        self.comparison = "NOT_REACHED"
        self.saved = False

    def save_then_compare(self, arrays, per_class, compare):
        import numpy as np

        self.calls += 1
        if self.calls != 1:
            raise RuntimeError("A second comparison is outside the one-image capture scope")
        image_ids = arrays["image_ids"]
        if len(image_ids) and not np.all(image_ids == self.spec["image_id"]):
            raise ValueError("Materialized arrays belong to another image")
        actual = {key: np.asarray(arrays[key]) for key in ("boxes", "scores", "labels")}
        native = {f"class_{label}": np.asarray(rows)
                  for label, rows in enumerate(per_class)}
        for name, values in (("actual_operands.npz", actual), ("native_operands.npz", native)):
            with (self.directory / name).open("xb") as stream:
                np.savez_compressed(stream, **values)
                stream.flush()
                os.fsync(stream.fileno())
        self.saved = True
        counts = [len(rows) for rows in per_class]
        write_json(self.directory / "capture.json", {
            "kind": "new_authorized_diagnostic_operands_not_roi_export",
            "scope_id": self.spec["scope_id"],
            "image_id": self.spec["image_id"],
            "native_prediction_index": self.spec["native_prediction_index"],
            "identity": self.spec["identity"],
            "checkpoint": self.run["checkpoint"], "config": self.run["config"],
            "native_prediction": self.run["native_prediction"],
            "original_roi_out_dir": self.run["out_dir"],
            "image_path": self.spec["image_path"],
            "image_sha256": self.spec["image_sha256"],
            "native_class_row_counts": counts,
            "actual_shapes": {key: list(value.shape) for key, value in actual.items()},
            "actual_dtypes": {key: str(value.dtype) for key, value in actual.items()},
            "operand_files": {
                name: sha256(self.directory / name)
                for name in ("actual_operands.npz", "native_operands.npz")
            },
            "full_features_saved": False, "model_state_saved": False,
            "roi_completion": False,
        })
        if (counts != self.spec["native_class_row_counts"]
                or not all(np.isfinite(rows).all() for rows in per_class)):
            raise ValueError("Passed native operands differ from the authorized reference facts")
        try:
            compare(arrays, per_class)
        except ValueError:
            self.comparison = "MISMATCH"
            raise
        self.comparison = "MATCH_DIAGNOSTIC_ONLY"
        raise FirstImageCaptured("First image matched; stop without ROI NPZ/index or second image")


@contextmanager
def observe_consumer(spec, run, capture, completion, alignment):
    original_load = completion.load_run
    original_compare = alignment.validate_native_predictions

    def load_run(plan, run_id):
        if not Path(plan).samefile(spec["plan"]) or run_id != spec["run_id"]:
            raise ValueError("Consumer requested a different plan/run")
        selected = original_load(plan, run_id)
        if selected != run:
            raise ValueError("The prepared run changed before the one-image consumer")
        return {**selected, "out_dir": str(capture.directory / "unused_roi_output")}

    def compare(arrays, per_class):
        return capture.save_then_compare(arrays, per_class, original_compare)

    with patch.object(completion, "load_run", load_run), \
            patch.object(alignment, "validate_native_predictions", compare):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--cpu-entry-check", action="store_true")
    args = parser.parse_args()
    entry = Path(__file__).resolve()
    spec_file = Path(args.spec).resolve()
    spec = json.loads(spec_file.read_text())
    consumer = Path(spec["consumer_checkout"]).resolve()
    os.chdir(consumer)
    sys.path.insert(0, str(consumer))
    sys.argv = [str(entry), "--spec", str(spec_file),
                "--physical-gpu", str(args.physical_gpu)]
    if args.cpu_entry_check:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.argv.append("--cpu-entry-check")
    os.environ["IRAOD_CONDA_PREFIX"] = str(Path(spec["python"]).parent.parent)
    runtime = load_module("iraod_runtime", consumer / "iraod_runtime.py")
    runtime.ensure_iraod_runtime()

    if git_head(consumer) != spec["consumer_sha"]:
        raise ValueError("Scientific ROI consumer checkout is not the frozen revision")
    if git_head(entry.parents[2]) != spec["capture_commit"]:
        raise ValueError("Instrumentation checkout differs from the reviewed capture commit")
    if sha256(spec["resolver"]["path"]) != spec["resolver"]["sha256"]:
        raise ValueError("Audited native resolver identity changed")
    if not Path(sys.prefix).samefile(Path(spec["python"]).parent.parent):
        raise ValueError("Capture interpreter differs from the bound native environment")
    if args.physical_gpu not in spec["allowed_gpus"]:
        raise ValueError("Physical GPU is outside the one-image authorization")

    from experiments.comparison import aligned_roi, result_completion
    run = select_run(spec, result_completion)
    info = {
        "scope_id": spec["scope_id"], "new_authorized_execution": True,
        "consumer_sha": spec["consumer_sha"], "capture_commit": spec["capture_commit"],
        "capture_file_sha256": sha256(entry),
        "resolver": spec["resolver"], "cwd": str(Path.cwd()),
        "python": sys.executable, "hostname": socket.gethostname(),
        "physical_gpu": args.physical_gpu, "runtime_ready": os.environ.get("IRAOD_RUNTIME_READY"),
        "image_id": spec["image_id"], "native_prediction_index": spec["native_prediction_index"],
        "identity": spec["identity"], "checkpoint": run["checkpoint"], "config": run["config"],
        "image_path": spec["image_path"], "image_sha256": spec["image_sha256"],
        "native_prediction": run["native_prediction"],
    }
    if args.cpu_entry_check:
        print(json.dumps({**info, "cpu_entry_check": True, "model_forward": False}))
        return 0

    directory = Path(spec["capture_dir"])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "entry.json", info)
    capture = OperandCapture(spec, run, directory)
    started = time.monotonic()
    outcome, error, code = "PRE_CAPTURE_FAILURE", None, 1
    try:
        host = load_module("one_image_native_host", spec["resolver"]["path"])
        native_entry = consumer / "experiments/comparison/dior_recovery/extract_roi_pre_fc_cls.py"
        sys.argv = [spec["resolver"]["path"], "native", str(native_entry),
                    "--plan", spec["plan"], "--run-id", spec["run_id"],
                    "--physical-gpu", str(args.physical_gpu)]
        with observe_consumer(spec, run, capture, result_completion, aligned_roi):
            host.main()
        raise RuntimeError("Frozen consumer returned without the mandatory first-image stop")
    except FirstImageCaptured:
        outcome, code = "CAPTURED_MATCH_DIAGNOSTIC_ONLY", 3
    except BaseException as failure:
        outcome = "CAPTURED_MISMATCH" if capture.comparison == "MISMATCH" else "CAPTURE_FAILED"
        error = {"type": type(failure).__name__, "message": str(failure),
                 "traceback": traceback.format_exc()}
        raise
    finally:
        versions = {
            name: getattr(sys.modules.get(name), "__version__", None)
            for name in ("torch", "mmcv", "mmdet", "numpy")
        }
        write_json(directory / "terminal.json", {
            **info, "outcome": outcome, "comparison": capture.comparison,
            "operands_saved": capture.saved, "comparison_calls": capture.calls,
            "runtime_versions": versions,
            "elapsed_host_seconds": time.monotonic() - started,
            "error": error, "roi_completion": False,
            "operator_deadline": "External <=300s cap includes all startup/capture/teardown",
        })
    return code


if __name__ == "__main__":
    raise SystemExit(main())
