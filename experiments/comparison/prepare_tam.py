"""CPU-only preparation of the approved12 domain TAM fits; never launches jobs."""

import argparse
from pathlib import Path
import shlex

from experiments.comparison.extension_training import ROOT, require_file, target_val
from experiments.comparison.report_qualitative import validate_plan
from experiments.comparison.result_completion import DOMAINS, read_json, write_json
from experiments.comparison.tam_artifacts import FIT_SEED, FORMAL_STEPS, NORMALIZATION


def prepare(base_plan, vgg_weights, out_dir, artifact_root, python, workers=16):
    plan = read_json(base_plan)
    validate_plan(plan)
    encoder = str(require_file(vgg_weights).resolve())
    python = str(require_file(python).absolute())
    out, artifacts = Path(out_dir).resolve(), Path(artifact_root).resolve()
    if out.exists() or artifacts.exists():
        raise ValueError("Metadata directory and formal TAM root must be NEW")
    if out == artifacts or out in artifacts.parents or artifacts in out.parents:
        raise ValueError("Metadata and formal TAM artifacts must have separate roots")
    if workers < 0:
        raise ValueError("Workers must be nonnegative")
    sources = {(r["dataset"], r["domain"]): r for r in plan["runs"]
               if r["method"] == "A" and r["role"] == "source"}
    fits = {}
    for dataset, domains in DOMAINS.items():
        for domain in domains:
            key = f"{dataset}/{domain}"
            val = target_val(dataset, domain, sources[dataset, domain]["img_prefix"])
            work = artifacts / dataset.lower() / domain / "seed_42"
            command = [
                python, "-m", "experiments.comparison.train_tam",
                "--base-plan", str(Path(base_plan).resolve()),
                "--dataset", dataset, "--domain", domain, "--seed", str(FIT_SEED),
                "--vgg-weights", encoder, "--out-dir", str(work),
                "--workers", str(workers),
            ]
            fits[key] = {
                "identity": {"dataset": dataset, "domain": domain, "seed": FIT_SEED},
                "target_val": val, "checkpoint": str(work / "tam.pth"),
                "outer_iterations": FORMAL_STEPS, "optimizer_updates": 2 * FORMAL_STEPS,
                "content_slots": 8, "style_slots": 8, "detector_seeds": [42, 43, 44],
                "command": command, "cwd": str(ROOT),
            }
    runtime = {
        "status": "prepared_not_execution_evidence",
        "encoder_weights": encoder, "normalization": NORMALIZATION,
        "fit_count": len(fits), "outer_iterations_total": len(fits) * FORMAL_STEPS,
        "optimizer_updates_total": len(fits) * FORMAL_STEPS * 2,
        "fits": fits,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "runtime.json", runtime)
    # Commands deliberately omit --gpu: the existing GPU owner chooses/locks it.
    (out / "fit_commands.txt").write_text(
        "\n".join(shlex.join(fit["command"]) for fit in fits.values()) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-plan", "vgg-weights", "out-dir", "artifact-root", "python"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=16)
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
