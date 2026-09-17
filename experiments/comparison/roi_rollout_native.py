"""One assigned rollout case through the existing authentic native producer."""

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def load_profile_producer(profile):
    """Keep the selected writer package, but bind its explicit external resolver."""
    path = Path(profile["resolver"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != profile["resolver_sha256"]:
        raise ValueError("Selected profile resolver identity differs")
    name = "experiments.comparison.host_binding"
    spec = importlib.util.spec_from_file_location(name, path)
    host = importlib.util.module_from_spec(spec)
    sys.modules[name] = host
    spec.loader.exec_module(host)
    package = importlib.import_module("experiments.comparison")
    package.host_binding = host
    writer = importlib.import_module("experiments.comparison.extension_manifest")
    expected = Path(profile["writer_code"]) / "experiments/comparison/extension_manifest.py"
    if not Path(writer.__file__).samefile(expected):
        raise ValueError("Native producer did not resolve from the selected writer checkout")
    return host, writer.evaluate_binding


def main():
    spec_path = Path(sys.argv[1]).resolve()
    gpu = int(sys.argv[2])
    spec = json.loads(spec_path.read_text())
    profile = spec["profile"]
    writer = Path(profile["writer_code"])
    os.chdir(writer)
    os.environ["PYTHONPATH"] = str(writer)
    os.environ["IRAOD_CONDA_PREFIX"] = str(Path(profile["python"]).parent.parent)
    sys.path.insert(0, str(writer))
    sys.argv = [str(Path(__file__).resolve()), str(spec_path), str(gpu)]
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()
    host, evaluate_binding = load_profile_producer(profile)

    if gpu not in profile["allowed_gpus"] or gpu not in host.approved_gpus():
        raise ValueError("Use the operator-assigned approved GPU for this case/host")
    if spec["kind"] == "port":
        plan = json.loads(Path(host.map_path(spec["source_plan"])).read_text())
        original_run = next(r for r in plan["runs"] if r["run_id"] == spec["run_id"])
        run = host.map_data(original_run)
        execution = json.loads((Path(run["native_prediction"]["eval_dir"]) / "execution.json").read_text())
        cell = {k: run[k] for k in ("dataset", "domain", "seed", "method", "role",
                                    "checkpoint", "config", "ann_file", "img_prefix")}
        cell.update(training_code_sha=execution["training_code_sha"], source_id=execution["source_id"])
        original_input = {"run": original_run, "execution": execution,
                          "source_plan": spec["source_plan"]}
    elif spec["kind"] == "bf":
        inputs = json.loads(Path(spec["input_specs"]).read_text())["records"]
        record = next(r for r in inputs if r["canonical_key"] == spec["canonical_key"])
        mapped = host.map_data(record)
        cell = {k: mapped[k] for k in ("dataset", "domain", "seed", "method", "role",
                                      "checkpoint", "config", "ann_file", "img_prefix",
                                      "training_code_sha", "source_id")}
        original_input = record
    elif spec["kind"] == "mdp":
        module_spec = importlib.util.spec_from_file_location(
            "declared_mdp_inputs", Path(__file__).with_name("mdp_joint_inputs.py"))
        declarations = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(declarations)
        plan = json.loads(Path(spec["source_plan"]).read_text())
        original_run, declaration = declarations.declared_run(plan, spec["run_id"])
        run = host.map_data(original_run)
        if gpu not in run["allowed_gpus"]:
            raise ValueError("MDP GPU assignment differs from the declared role")
        cell = {k: run[k] for k in ("dataset", "domain", "seed", "method", "role",
                                    "checkpoint", "config", "ann_file", "img_prefix",
                                    "training_code_sha", "source_id")}
        original_input = {"run": original_run, "source_plan": spec["source_plan"],
                          "declaration": declaration}
    else:
        raise ValueError("Only authorized PORT/B-F/declared-MDP rollout records are supported")
    key = "/".join(str(cell[k]) for k in ("dataset", "domain", "seed", "method", "role"))
    if key != spec["canonical_key"]:
        raise ValueError("Input identity differs from the operator-assigned key")
    root = Path(spec["case_root"])
    cell["eval_dir"] = str(root / "native")
    code = profile["evaluation_code"]
    evaluation_sha = subprocess.check_output(["git", "-C", code, "rev-parse", "HEAD"], text=True).strip()
    if spec["kind"] == "mdp" and evaluation_sha != original_run["evaluation_code_sha"]:
        raise ValueError("MDP evaluator revision differs from its declaration")
    runtime = {"python": profile["python"], "evaluation_code": code,
               "evaluation_code_sha": evaluation_sha}
    evaluate_binding(cell, runtime, gpu)
    with (root / "native_pass.json").open("x") as stream:
        json.dump({"status": "complete", "scope_id": "iraod-roi-local-native-rollout-342-20260913",
                   "kind": spec["kind"], "canonical_key": key, "physical_gpu": gpu,
                   "cell": cell, "evaluation_code_sha": evaluation_sha,
                   "original_input": original_input, "profile": profile,
                   "reference_policy": "predeclared local native ROI reference; original metrics unchanged"},
                  stream, indent=2)


if __name__ == "__main__":
    main()
