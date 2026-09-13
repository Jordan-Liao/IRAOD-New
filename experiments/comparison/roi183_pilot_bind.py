"""CPU continuation after the real native producer succeeds; never launches ROI."""

import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys


ROOT = Path("/home/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908")
PILOT = ROOT / "roi183_native_reference_pilot_20260913"
OLD_PLAN = ROOT / "qualitative_manifests_roi183_860d4a0/qualitative_seed42.irg-brightness-ema.json"
RUN_ID = "DIOR/brightness/seed_42/IRG/ema"
ENTRY = "/home/zechuan/iraod_scratch/roi183_entry_repair_20260913/host183_roi_entry.sh"


def identity(path):
    payload = Path(path).read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    gpu = int(sys.argv[1])
    if gpu not in (8, 9):
        raise ValueError("Use the same operator-selected183 GPU8/9 as the native stage")
    passed = json.loads((PILOT / "native_pass.json").read_text())
    if passed["status"] != "complete":
        raise ValueError("Native producer did not complete its existing checks")
    plan = json.loads(OLD_PLAN.read_text())
    index = next(i for i, row in enumerate(plan["runs"]) if row["run_id"] == RUN_ID)
    original = plan["runs"][index]
    cell = passed["cell"]
    keys = ("dataset", "domain", "seed", "method", "role", "checkpoint", "config",
            "ann_file", "img_prefix")
    if any(cell[key] != original[key] for key in keys):
        raise ValueError("Actual native producer case differs from the frozen ROI case")
    if cell["source_id"] != original["native_prediction"]["source_id"]:
        raise ValueError("Native reference changed the source identity")
    native = Path(cell["eval_dir"])
    proofs = copy.deepcopy(original["native_prediction"]["path_binding"])
    proofs["artifacts"]["execution_json"] = identity(native / "execution.json")
    proofs["artifacts"]["prediction_sidecar"] = identity(native / "predictions.pkl.image_ids.json")
    proofs["transfer_provenance"] = {
        "scope_id": "iraod-roi183-native-reference-pilot-20260913",
        "reference_host": 183, "native_producer_result": str(PILOT / "native_pass.json"),
        "checkpoint_config_proofs": "Reused existing verified immutable inputs; not rescanned",
        "execution_sidecar_proofs": "New genuine destination-native outputs",
    }
    run = {
        **original,
        "out_dir": str(PILOT / "roi" / RUN_ID),
        "native_prediction": {
            "eval_dir": str(native),
            "prepared_training_code_sha": cell["training_code_sha"],
            "evaluation_code_sha": passed["evaluation_code_sha"],
            "source_id": cell["source_id"], "path_binding": proofs,
        },
    }
    if Path(run["out_dir"]).exists():
        raise FileExistsError(run["out_dir"])
    plan["runs"][index] = run
    plan["roi_reference_pilot"] = {
        "scope_id": "iraod-roi183-native-reference-pilot-20260913",
        "authorized_at_utc": "2026-09-13T18:33:05.691Z",
        "run_id": RUN_ID, "predeclared_host": 183,
        "old_native_reference_preserved": original["native_prediction"],
        "policy": "Fresh native reference solely for this ROI; no TEST-score selection",
        "original67_PR25_metrics_unchanged": True,
        "canonical_denominator_changed": False,
    }
    filename = PILOT / "qualitative_seed42.local_native_ref.json"
    with filename.open("x") as stream:
        json.dump(plan, stream, indent=2)
    command = ["env", "IRAOD_GPU_LOCKED=1", f"CUDA_VISIBLE_DEVICES={gpu}",
               "bash", ENTRY, "--plan", str(filename), "--run-id", RUN_ID,
               "--physical-gpu", str(gpu)]
    with (PILOT / "roi_entry.sh").open("x") as stream:
        stream.write("#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
    print(json.dumps({"status": "ROI_PLAN_PREPARED_FROM_REAL_NATIVE_PASS",
                      "plan": str(filename), "roi_entry": str(PILOT / "roi_entry.sh"),
                      "physical_gpu": gpu, "roi_completed": False}))


if __name__ == "__main__":
    main()
