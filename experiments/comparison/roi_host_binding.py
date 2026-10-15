"""Run one preserved frozen ROI job through the selected host's IO boundary."""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from experiments.comparison import host_binding as host
from experiments.comparison import result_completion as completion


ROOT = Path(__file__).resolve().parents[2]
EXPORT_MODULE = "experiments.comparison.dior_recovery.extract_roi_pre_fc_cls"


def require_owned_gpu(run):
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("ROI extraction requires the existing owner's actual GPU lock")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in tuple(map(str, host.approved_gpus())):
        raise ValueError("ROI extraction requires one bound physical target GPU0-4")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export(jobs_path, seed, run_id):
    if not host.is_target_host():
        raise ValueError("ROI overlay is only for the selected target host; use original jobs elsewhere")
    jobs = completion.read_json(jobs_path)
    selected = [job for job in jobs if job["seed"] == seed and job["run_id"] == run_id]
    if len(selected) != 1:
        raise ValueError("Select exactly one preserved ROI job by seed and run ID")
    job = selected[0]
    argv = job["export_argv"]
    if argv[1:] != ["-m", EXPORT_MODULE, "--plan", job["plan"], "--run-id", run_id]:
        raise ValueError("Frozen ROI argument vector differs from its job binding")
    run = completion.load_run(job["plan"], run_id)
    if (run["method"] == "A" or run["seed"] != seed
            or not host.same_data(run["checkpoint"], job["checkpoint"])
            or not host.same_data(run["out_dir"], job["out_dir"])):
        raise ValueError("ROI job differs from its preserved non-source plan binding")
    require_owned_gpu(run)
    code = host.read_path(job["cwd"] if "native_prediction" in run else job["export_code"])
    export_sha = subprocess.check_output(
        ["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if export_sha != job["export_code_sha"]:
        raise ValueError("Frozen ROI checkout differs from the recorded export code SHA")
    wrapper_sha = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    wrapper_modified = bool(subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
    entry = code / "experiments/comparison/dior_recovery/extract_roi_pre_fc_cls.py"
    physical_run = host.map_data(run)
    for key in ("checkpoint", "config", "ann_file", "img_prefix", "out_dir"):
        physical_run[key] = str(host.read_path(run[key]))

    # Only operational readers are current; the exporter, ROI arithmetic and
    # detector imports continue to come from the recorded frozen checkout.
    os.environ.update(host.native_environment())
    os.chdir(code)
    sys.path.insert(0, str(code))
    sys.modules["experiments.comparison.result_completion"] = completion
    aligned_name = "experiments.comparison.aligned_roi"
    sys.modules[aligned_name] = load_module(
        aligned_name, code / "experiments/comparison/aligned_roi.py")
    frozen = load_module("iraod_frozen_roi_export", entry)
    frozen.Path = host.read_path
    frozen.load_run = lambda plan, identity: physical_run
    frozen.require_owned_gpu = require_owned_gpu
    sys.argv = [str(entry), *argv[3:]]
    with host.native_config_paths():
        frozen.main()

    index_path = host.read_path(run["out_dir"]) / "index.json"
    index = completion.read_json(index_path)
    if (index["status"] != "complete" or index["code_commit"] != export_sha
            or not host.same_data(index["run"], run)):
        raise ValueError("Frozen export output differs from its preserved job identity")
    index["run"] = run
    index["host_binding"] = {
        "host": host.TARGET_HOST, "wrapper_code_sha": wrapper_sha,
        "wrapper_worktree_modified": wrapper_modified,
        "export_code_sha": export_sha, "export_code": str(code),
        "jobs": str(jobs_path), "export_argv": argv,
        "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]),
    }
    temporary = index_path.with_suffix(".json.partial")
    completion.write_json(temporary, index)
    temporary.replace(index_path)
    return index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    export(args.jobs, args.seed, args.run_id)


if __name__ == "__main__":
    main()
