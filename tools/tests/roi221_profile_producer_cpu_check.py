"""One actual221 CPU import/producer dispatch check; child launch is blocked."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch


class ChildLaunchBlocked(RuntimeError):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--old-profile", required=True)
    parser.add_argument("--new-profile", required=True)
    parser.add_argument("--phase", choices=("red", "green"))
    args = parser.parse_args()
    if args.phase is None:
        results = {}
        for phase, profile_path in (("red", args.old_profile), ("green", args.new_profile)):
            profile = json.loads(Path(profile_path).read_text())
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": profile["writer_code"],
                   "PYTHONDONTWRITEBYTECODE": "1"}
            command = [profile["python"], str(Path(__file__).resolve()),
                       "--entry", args.entry, "--old-profile", args.old_profile,
                       "--new-profile", args.new_profile, "--phase", phase]
            result = subprocess.run(command, cwd=profile["writer_code"], env=env,
                                    capture_output=True, text=True, timeout=60)
            results[phase] = {"rc": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        print(json.dumps(results, indent=2))
        if (results["red"]["rc"] != 1
                or "ImportError: cannot import name 'host_binding'" not in results["red"]["stderr"]
                or results["green"]["rc"] != 0):
            raise SystemExit(1)
        return

    profile_path = args.old_profile if args.phase == "red" else args.new_profile
    profile = json.loads(Path(profile_path).read_text())
    os.chdir(profile["writer_code"])
    sys.path.insert(0, profile["writer_code"])
    if args.phase == "red":
        from experiments.comparison import host_binding
        raise RuntimeError(f"Old split profile unexpectedly resolved {host_binding.__file__}")

    spec = importlib.util.spec_from_file_location("fixed_native_entry", args.entry)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    host, evaluate = entry.load_profile_producer(profile)
    writer = sys.modules[evaluate.__module__]
    assert Path(host.__file__).samefile(profile["resolver"])
    assert Path(writer.__file__).samefile(
        Path(profile["writer_code"]) / "experiments/comparison/extension_manifest.py")
    assert 1 in host.approved_gpus()
    observed = []

    def block(command, **kwargs):
        observed.append({"command": command, "cwd": str(kwargs["cwd"]),
                         "child_cuda_visible_devices": kwargs["env"]["CUDA_VISIBLE_DEVICES"],
                         "child_pythonpath": kwargs["env"]["PYTHONPATH"]})
        assert command[:4] == [
            profile["python"], profile["resolver"], "native",
            str(Path(profile["evaluation_code"]) / "test.py")]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == profile["evaluation_code"]
        raise ChildLaunchBlocked("Actual child-launch seam blocked; no evaluation")

    with tempfile.TemporaryDirectory(prefix="roi221-dispatch-cpu-") as temporary:
        output = Path(temporary) / "fixture-only-not-an-evaluation"
        cell = {"dataset": "DIOR", "domain": "cpu-fixture-only", "seed": 42,
                "method": "IRG", "role": "ema", "eval_dir": str(output),
                "config": "/CPU_FIXTURE_NOT_OPENED/config.py",
                "checkpoint": "/CPU_FIXTURE_NOT_OPENED/iter_185_ema.pth",
                "training_code_sha": "CPU_FIXTURE", "ann_file": "/CPU_FIXTURE/split",
                "img_prefix": "/CPU_FIXTURE/images"}
        runtime = {"python": profile["python"], "evaluation_code": profile["evaluation_code"],
                   "evaluation_code_sha": "CPU_FIXTURE_NOT_EXECUTED"}
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(writer.subprocess, "run", block):
            try:
                evaluate(cell, runtime, 1)
            except ChildLaunchBlocked:
                pass
            else:
                raise AssertionError("Guarded child-launch boundary was not reached")
        assert len(observed) == 1
        assert not (output / "eval_status").exists()
        assert not (output / "predictions.pkl").exists()
    import torch
    assert not torch.cuda.is_initialized()
    print(json.dumps({
        "status": "ACTUAL_PROFILE_IMPORT_AND_HOST_AWARE_DISPATCH_PASS",
        "writer_origin": writer.__file__, "resolver_origin": host.__file__,
        "approved_gpus": list(host.approved_gpus()), "observed": observed,
        "child_launches": 0, "model_data_checkpoint_work": False,
        "cuda_initialized": False, "scratch_fixture_cleaned": True,
    }, indent=2))


if __name__ == "__main__":
    main()
