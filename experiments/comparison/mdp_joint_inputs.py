"""Declare new MDP provenance and a task-owned Host134 resolver."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from experiments.comparison.result_completion import validate_run


SCHEMA = "iraod-mdp-declared-roi-plan-v1"


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def declared_run(plan, run_id):
    from experiments.comparison.report_inputs import FINAL_ITERATION

    if plan["schema"] != SCHEMA:
        raise ValueError("MDP requires a genuine new-method declaration")
    rows = [row for row in plan["runs"] if row["run_id"] == run_id]
    if len(rows) != 1:
        raise ValueError("MDP declaration must select one unique run")
    run = rows[0]
    if (run["method"] != "MDP" or run["role"] not in ("student", "ema")
            or run["declaration_kind"] != "NEW_MDP_RUNTIME_ROW_NOT_HISTORICAL_REUSE"
            or "native_prediction" in run):
        raise ValueError("MDP input must be a new declared role, not historical native reuse")
    validate_run(run)
    proof = read_json(run["checkpoint_proof"])
    role = run["role"]
    key = "/".join(str(run[name]) for name in ("dataset", "domain", "seed", "method"))
    if run["canonical_key"] != key + "/" + role:
        raise ValueError("MDP declaration canonical key differs from its row")
    selected = proof[role]
    receipts = plan["input_receipts"]
    proof_identity = receipts["config_identities"]["producer_proof"]
    if identity(run["checkpoint_proof"]) != {
            name: proof_identity[name] for name in ("bytes", "sha256")}:
        raise ValueError("MDP producer evidence identity changed")
    if (proof["status"] != "FINITE_NATIVE_FINAL_PAIR_VALIDATED"
            or proof["cell"] != key or proof["algorithm_code_sha"] != run["training_code_sha"]
            or proof["valid_finite_diagnostic_pair"] is not True or selected["finite"] is not True
            or selected["epoch"] != 1
            or selected["iteration"] != FINAL_ITERATION[run["dataset"]]
            or proof["completed_updates"] != selected["iteration"] - 1
            or proof["lineage"]["seed"] != run["seed"]):
        raise ValueError("MDP producer proof differs from the selected final-budget role")
    transfers = [row for row in receipts["checkpoint_identities"] if row["role"] == role]
    if len(transfers) != 1:
        raise ValueError("MDP needs one authentic checkpoint transfer receipt per role")
    transfer = transfers[0]
    if (transfer["match"] is not True or transfer["source"] != transfer["destination"]
            or transfer["source_path"] != selected["path"]
            or Path(transfer["destination_path"]) != Path(run["checkpoint"])
            or transfer["source"]["bytes"] != selected["bytes"]
            or Path(run["checkpoint"]).stat().st_size != selected["bytes"]):
        raise ValueError("MDP staged checkpoint differs from its producer/transfer evidence")
    config_identity = receipts["config_identities"]["evaluator_config"]
    if identity(run["config"]) != {name: config_identity[name] for name in ("bytes", "sha256")}:
        raise ValueError("MDP evaluator configuration identity changed")
    selection_identity = receipts["config_identities"]["frozen32"]
    if identity(run["visualization_selection_evidence"]) != {
            name: selection_identity[name] for name in ("bytes", "sha256")}:
        raise ValueError("MDP frozen visualization selection identity changed")
    selection = read_json(run["visualization_selection_evidence"])
    if selection["image_ids"] != run["visualization_image_ids"]:
        raise ValueError("MDP visualization IDs differ from their frozen selection")
    return copy.deepcopy(run), {
        "kind": "new-trained-method-declaration",
        "producer_proof": {"path": run["checkpoint_proof"], **identity(run["checkpoint_proof"])},
        "checkpoint_transfer": transfer,
    }


def write_resolver(path, base_path):
    base_path = Path(base_path).resolve()
    digest = identity(base_path)["sha256"]
    text = f'''"""Task-owned Host134 GPU1/2 binding; the d668 resolver stays unchanged."""
import hashlib
import importlib.util
from pathlib import Path
import socket
import sys

BASE = Path({str(base_path)!r})
if hashlib.sha256(BASE.read_bytes()).hexdigest() != {digest!r}:
    raise ValueError("MDP base resolver identity changed")
spec = importlib.util.spec_from_file_location("mdp134_base_binding", BASE)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

def approved_gpus():
    if socket.gethostname() != base.TARGET_HOST_134:
        raise ValueError("This MDP resolver is restricted to Host134")
    return (1, 2)

def native_command(command, entry):
    command = list(command)
    index = command.index(str(entry))
    command[index:index] = [str(Path(__file__).resolve()), "native"]
    return command

base.approved_gpus = approved_gpus
base.native_command = native_command

def __getattr__(name):
    return getattr(base, name)

if __name__ == "__main__":
    base.main()
'''
    with Path(path).open("x") as stream:
        stream.write(text)


def prepare(inputs_file, evidence_file, roi_code, output, case_root):
    inputs = read_json(inputs_file)
    evidence = read_json(evidence_file)
    profile = copy.deepcopy(inputs["profile_input"])
    if profile["host"] != 134 or profile["allowed_gpus"] != [1, 2]:
        raise ValueError("This declaration is scoped to the approved Host134 GPU1/2 task")
    rows = copy.deepcopy(inputs["rows"])
    if ({row["canonical_key"] for row in rows} !=
            {"RSAR/chaff/42/MDP/student", "RSAR/chaff/42/MDP/ema"} or len(rows) != 2):
        raise ValueError("This task declares only the two approved RSAR/chaff/42 MDP roles")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    resolver = output / "mdp_host134_binding.py"
    write_resolver(resolver, Path(profile["writer_code"]) / "experiments/comparison/host_binding.py")
    profile.update(
        roi_code=str(Path(roi_code).resolve()), resolver=str(resolver),
        resolver_sha256=identity(resolver)["sha256"])
    profile.pop("resolver_requirement", None)
    for run in rows:
        run["allowed_gpus"] = [1 if run["role"] == "student" else 2]
        run["out_dir"] = str(Path(case_root).resolve() / run["role"] / "roi")
    plan = {
        "schema": SCHEMA, "scope_id": "iraod-mdp-host134-test-roi-20260916",
        "runs": rows, "input_receipts": evidence,
    }
    for run in rows:
        declared_run(plan, run["run_id"])
    plan_path = output / "mdp_declared_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    specs = {}
    for run in rows:
        filename = output / f"mdp_chaff42_{run['role']}.json"
        filename.write_text(json.dumps({
            "kind": "mdp", "canonical_key": run["canonical_key"],
            "run_id": run["run_id"], "source_plan": str(plan_path),
            "profile": profile,
        }, indent=2) + "\n")
        specs[run["role"]] = str(filename)
    return {"status": "DECLARED_NOT_EXECUTED", "plan": str(plan_path), "specs": specs,
            "profile": profile, "forward_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "evidence", "roi-code", "output", "case-root"):
        parser.add_argument("--" + name, required=True)
    options = parser.parse_args()
    print(json.dumps(prepare(
        options.inputs, options.evidence, options.roi_code, options.output, options.case_root)))
