"""Authorized fixed-case native reference. pT owns the GPU/budget; no retries."""

import json
import os
from pathlib import Path
import subprocess
import sys


WRITER = Path("/mnt/SSD1_8TB/zechuan/IRAOD-New-host183-7ae4ac3")
EVALUATOR = Path("/home/zechuan/IRAOD-New-rc331d213")
PYTHON = "/home/zechuan/anaconda3/envs/iraod/bin/python"
ROOT = Path("/home/zechuan/iraod_artifacts/comparison/xaf_student_quant_20260908")
PILOT = ROOT / "roi183_native_reference_pilot_20260913"
PLAN = ROOT / "qualitative_manifests_roi183_860d4a0/qualitative_seed42.irg-brightness-ema.json"
RUN_ID = "DIOR/brightness/seed_42/IRG/ema"


def main():
    gpu = int(sys.argv[1])
    os.chdir(WRITER)
    os.environ["PYTHONPATH"] = str(WRITER)
    os.environ["IRAOD_CONDA_PREFIX"] = str(Path(PYTHON).parent.parent)
    sys.path.insert(0, str(WRITER))
    from iraod_runtime import ensure_iraod_runtime
    ensure_iraod_runtime()
    from experiments.comparison.extension_manifest import evaluate_binding
    from experiments.comparison import host_binding

    if gpu not in (8, 9) or gpu not in host_binding.approved_gpus():
        raise ValueError("Pilot requires the one operator-assigned approved183 GPU8/9")
    plan = json.loads(PLAN.read_text())
    run = next(row for row in plan["runs"] if row["run_id"] == RUN_ID)
    original = json.loads((Path(run["native_prediction"]["eval_dir"]) / "execution.json").read_text())
    evaluation_sha = subprocess.check_output(
        ["git", "-C", str(EVALUATOR), "rev-parse", "HEAD"], text=True).strip()
    if evaluation_sha != "331d2131b84651f0a2930a3d53faeefad8701531":
        raise ValueError(f"Selected native evaluator changed: {evaluation_sha}")
    cell = {
        **{key: run[key] for key in ("dataset", "domain", "seed", "method", "role",
                                    "checkpoint", "config", "ann_file", "img_prefix")},
        "eval_dir": str(PILOT / "native"),
        "training_code_sha": original["training_code_sha"],
        "source_id": original["source_id"],
    }
    runtime = {"python": PYTHON, "evaluation_code": str(EVALUATOR),
               "evaluation_code_sha": evaluation_sha}
    evaluate_binding(cell, runtime, gpu)
    # This marker exists only after the actual producer's AP/status/sidecar checks return.
    with (PILOT / "native_pass.json").open("x") as stream:
        json.dump({"status": "complete", "scope_id": "iraod-roi183-native-reference-pilot-20260913",
                   "cell": cell, "physical_gpu": gpu, "evaluation_code_sha": evaluation_sha,
                   "producer_file": str(Path(evaluate_binding.__code__.co_filename).resolve()),
                   "reference_policy": "destination-native reference for ROI only; old67/PR25 unchanged"},
                  stream, indent=2)


if __name__ == "__main__":
    main()
