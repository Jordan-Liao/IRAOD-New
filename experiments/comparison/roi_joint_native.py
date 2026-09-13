"""Opt-in native TEST + streamed ROI; never invoked by the production queue."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ENTRY = Path(__file__).with_name("joint_native") / "test.py"


def capture_command(command, case_root, dataset, gpu):
    # The basename preserves the existing host wrapper's evaluate=True guard.
    return [
        command[0], str(CAPTURE_ENTRY), "--case-root", str(case_root),
        "--dataset", dataset, "--physical-gpu", str(gpu), "--", *command[1:],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True, help="Fresh candidate case root")
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--baseline-root", help="Completed ordinary native case for exact validation")
    args = parser.parse_args()
    source_spec = Path(args.spec).resolve()
    root = Path(args.out).resolve()
    baseline = Path(args.baseline_root).resolve() if args.baseline_root else None
    spec = json.loads(source_spec.read_text())
    profile = spec["profile"]
    writer = profile["writer_code"]
    sys.argv = [
        str(Path(__file__).resolve()), "--spec", str(source_spec), "--out", str(root),
        "--physical-gpu", str(args.physical_gpu),
    ]
    if baseline is not None:
        sys.argv += ["--baseline-root", str(baseline)]
    os.chdir(writer)
    os.environ["PYTHONPATH"] = writer
    os.environ["IRAOD_CONDA_PREFIX"] = str(Path(profile["python"]).parent.parent)
    sys.path.insert(0, writer)
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()

    root.mkdir(parents=True, exist_ok=False)
    spec["case_root"] = str(root)
    candidate = root / "case_spec.json"
    candidate.write_text(json.dumps(spec, indent=2) + "\n")
    code_sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    module_spec = importlib.util.spec_from_file_location(
        "joint_native_producer", Path(__file__).with_name("roi_rollout_native.py"))
    native = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(native)
    original_loader = native.load_profile_producer

    def load_producer(selected):
        host, evaluate = original_loader(selected)

        def joint_evaluate(cell, runtime, gpu):
            if baseline is not None:
                previous = json.loads((baseline / "native_pass.json").read_text())
                fields = ("dataset", "domain", "seed", "method", "role", "checkpoint",
                          "config", "ann_file", "img_prefix", "training_code_sha", "source_id")
                if (previous["status"] != "complete"
                        or any(previous["cell"][key] != cell[key] for key in fields)
                        or previous["profile"] != selected
                        or previous["physical_gpu"] != gpu
                        or previous["evaluation_code_sha"] != runtime["evaluation_code_sha"]):
                    raise ValueError("Baseline must be the same case, host, GPU and evaluation code")
            cell["joint_exporter"] = {
                "code_commit": code_sha, "entry": str(CAPTURE_ENTRY),
                "mode": "single-forward-native-roi",
            }
            original_command = host.native_command

            def joint_command(command, entry):
                if Path(entry) != Path(runtime["evaluation_code"]) / "test.py":
                    raise ValueError("Joint capture only wraps the selected native TEST entry")
                return original_command(
                    capture_command(command, root, cell["dataset"], gpu), CAPTURE_ENTRY)

            with patch.object(host, "native_command", joint_command):
                return evaluate(cell, runtime, gpu)

        return host, joint_evaluate

    with patch.object(native, "load_profile_producer", load_producer), \
            patch.object(sys, "argv", [native.__file__, str(candidate), str(args.physical_gpu)]):
        native.main()
    command = [
        profile["python"], "-m", "experiments.comparison.joint_roi",
        "--case-root", str(root),
    ]
    if baseline is not None:
        command += ["--baseline-root", str(baseline)]
    subprocess.run(command, cwd=ROOT, check=True, env={
        **os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
    })


if __name__ == "__main__":
    main()
