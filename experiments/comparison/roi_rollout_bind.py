"""One CPU continuation from an actual rollout native PASS; no GPU launch."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import sys


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def load_host(profile):
    spec = importlib.util.spec_from_file_location("rollout_paths", profile["resolver"])
    host = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = host
    spec.loader.exec_module(host)
    return host


def bind_plan(root, export_code_sha="860d4a0afb32a8e75e612aec65398a73e4747877"):
    root = Path(root).resolve()
    passed = json.loads((root / "native_pass.json").read_text())
    if passed["status"] != "complete":
        raise ValueError("Existing native producer has not completed")
    profile, cell = passed["profile"], passed["cell"]
    host = load_host(profile)
    old = passed["original_input"]
    if passed["kind"] == "port":
        original = old["run"]
        plan_path = host.map_path(old["source_plan"])
    elif passed["kind"] == "mdp":
        from experiments.comparison.mdp_joint_inputs import declared_run

        plan_path = old["source_plan"]
        original, declaration = declared_run(
            json.loads(Path(plan_path).read_text()), old["run"]["run_id"])
        if original != old["run"] or declaration != old["declaration"]:
            raise ValueError("New MDP declaration differs from its executed input")
        if any(original[key] != cell[key] for key in
               ("dataset", "domain", "seed", "method", "role", "checkpoint", "config")):
            raise ValueError("New MDP native execution differs from the selected role")
        admission = json.loads((root / "mdp_checkpoint_load.json").read_text())
        if (admission["method"] != "MDP" or admission["role"] != cell["role"]
                or admission["checkpoint"] != cell["checkpoint"]
                or admission["strict_detector_load"] is not True
                or admission["detector_keys"] != 396
                or len(admission["auxiliary_keys"]) != (11 if cell["role"] == "student" else 0)):
            raise ValueError("MDP native checkpoint admission proof differs from its role")
    elif passed["kind"] == "bf":
        plan_path = host.map_path(old["original_plan"]["path"])
        source_plan = json.loads(Path(plan_path).read_text())
        original = source_plan["runs"][old["original_plan"]["run_index"]]
        if any(original[key] != cell[key] for key in ("dataset", "domain", "seed", "method", "role")):
            raise ValueError("Authentic BF plan row differs from the assigned case")
    else:
        raise ValueError("Unsupported rollout provenance kind")
    run = host.map_data(original)
    plan = json.loads(Path(plan_path).read_text())
    reference = {
        "eval_dir": cell["eval_dir"], "prepared_training_code_sha": cell["training_code_sha"],
        "evaluation_code_sha": passed["evaluation_code_sha"], "source_id": cell["source_id"],
        "format": None,
        "record_kind": "fresh-successful-destination-native-execution",
    }
    new_artifacts = {
        "execution_json": identity(Path(cell["eval_dir"]) / "execution.json"),
        "prediction_sidecar": identity(Path(cell["eval_dir"]) / "predictions.pkl.image_ids.json"),
    }
    if passed["kind"] == "bf":
        reference["historical_input_provenance"] = {
            "format": "comparison-report-v1",
            "original_report": old["original_report"],
            "recorded_checkpoint_proof": old["recorded_checkpoint_proof"],
            "recorded_source_proof": old["recorded_source_proof"],
        }
    elif passed["kind"] == "mdp":
        reference["declared_training_provenance"] = old["declaration"]
        reference["checkpoint_admission"] = admission
    proof_file = root / "input_proofs.json"
    previous = original.get("native_prediction", {}).get("path_binding")
    if proof_file.is_file() or previous is not None:
        inputs = (json.loads(proof_file.read_text()) if proof_file.is_file()
                  else previous["artifacts"])
        reference["path_binding"] = {
            "resolver": {"path": profile["resolver"], "sha256": profile["resolver_sha256"]},
            "artifacts": {key: inputs[key] for key in ("checkpoint", "config")},
        }
        reference["path_binding"]["artifacts"].update(new_artifacts)
    else:
        # Preserve the existing native record/report checks when no historical
        # transfer digest exists; never manufacture one or add a weight rescan.
        reference["reference_artifact_identities"] = new_artifacts
    run.update(
        **{key: cell[key] for key in ("checkpoint", "config", "ann_file", "img_prefix")},
        out_dir=str(root / "roi"), native_prediction=reference,
        export_code_sha=export_code_sha,
        allowed_gpus=[passed["physical_gpu"]] if passed["kind"] == "bf" else run["allowed_gpus"],
    )
    if Path(run["out_dir"]).exists():
        raise FileExistsError(run["out_dir"])
    plan["runs"] = [run if row["run_id"] == original["run_id"] else row for row in plan["runs"]]
    plan["local_native_rollout"] = {
        "scope_id": "iraod-roi-local-native-rollout-342-20260913",
        "canonical_key": passed["canonical_key"], "host": profile["host"],
        "original_native_preserved": (old["original_report"] if passed["kind"] == "bf"
                                      else None if passed["kind"] == "mdp"
                                      else original["native_prediction"]),
        "canonical_metrics_and_denominator_unchanged": True,
    }
    filename = root / "roi_plan.json"
    with filename.open("x") as stream:
        json.dump(plan, stream, indent=2)
    return passed, run, filename


def main():
    root = Path(sys.argv[1]).resolve()
    passed, run, filename = bind_plan(root)
    profile = passed["profile"]
    gpu = passed["physical_gpu"]
    code = profile["roi_code"]
    command = [
        "env", "IRAOD_GPU_LOCKED=1", f"CUDA_VISIBLE_DEVICES={gpu}", f"PYTHONPATH={code}",
        "PYTHONDONTWRITEBYTECODE=1", profile["python"], profile["resolver"], "native",
        str(Path(code) / "experiments/comparison/dior_recovery/extract_roi_pre_fc_cls.py"),
        "--plan", str(filename), "--run-id", run["run_id"],
        "--physical-gpu", str(gpu),
    ]
    with (root / "roi_entry.sh").open("x") as stream:
        stream.write("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(code) +
                     "\nexec " + shlex.join(command) + "\n")
    print(json.dumps({"status": "ROI_ENTRY_READY", "plan": str(filename),
                      "entry": str(root / "roi_entry.sh"), "roi_complete": False}))


if __name__ == "__main__":
    main()
