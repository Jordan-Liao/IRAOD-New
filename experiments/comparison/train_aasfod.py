"""Run two fixed-budget AASFOD stages using fresh native runners/optimizers."""

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

# This helper is invoked by absolute path with PYTHONPATH pointing at bound model
# code. Keep orchestration (including the native shim) current; run() restores the
# bound model path before configs are loaded and native subprocesses are launched.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.comparison.aasfod_protocol import budget, validate_split
from experiments.comparison.extension_training import code_sha, load_cell, require_file
from experiments.comparison.result_completion import write_json
from experiments.comparison import host_binding as host


FNS_CELL = "RSAR/clean/42/AASFOD"
FNS_MODEL_SHA = "cbd0f75ab147fea6728f61d6fd696325cc195296"


def validate_fns_code(code, cell):
    code = host.read_path(code)
    if (host.same_path(str(code), cell["training_code"])
            or code_sha(code) != FNS_MODEL_SHA
            or subprocess.check_output(
                ["git", "-C", str(code), "status", "--porcelain", "--untracked-files=normal"],
                text=True).strip()):
        raise ValueError("FNS requires a separate clean accepted cbd0f75 model-code checkout")
    require_file(code / "train.py")
    return code


def validate_fns_evidence(cell, evidence_dir, moves=()):
    """Read original captures and retained files, never re-read large tensors.

    FNS command emission followed check=True alignment and both checkpoint
    checks in the original helper. The owner's CPU metadata and native logs
    corroborate that control-flow proof; root receipts may be an older attempt.
    """
    from mmcv import Config

    evidence = host.read_path(evidence_dir)
    method, work = Path(cell["method_dir"]), Path(cell["work_dir"])
    if ("/".join(str(cell[k]) for k in ("dataset", "domain", "seed", "method")) != FNS_CELL
            or cell["aasfod_budget"] != budget(8467) or cell["unlabeled_epoch_size"] != 8467
            or cell["training_code_sha"] != "36c20531574a3d85d0dc6f76f18d0c52cfd011c3"
            or work != method / "work" or work.is_symlink()
            or Path(cell["tsd_split"]) != method / "tsd.json"
            or Path(cell["terminal_status"]) != method / "terminal_status"
            or Path(cell["student_checkpoint"]) != work / "iter_266.pth"
            or Path(cell["checkpoint"]) != work / "iter_266_ema.pth"):
        raise ValueError("FNS continuation requires the unchanged RSAR/clean/42 native binding/budget")
    for path in (cell["student_checkpoint"], cell["checkpoint"]):
        if Path(path).exists():
            raise ValueError("FNS continuation cannot replace retained final checkpoints")

    def retained(path):
        path = Path(path)
        for move in moves:
            source = Path(move["source"])
            if path == source or source in path.parents:
                return Path(move["archive"]) / path.relative_to(source)
        return path

    summary = host.read_json(evidence / "EVIDENCE.json")
    old = summary["json"]["execution.json"]
    fields = ("dataset", "domain", "seed", "method", "training_code", "training_code_sha",
              "source_checkpoint", "config", "work_dir", "tsd_split", "aasfod_budget")
    if (not host.same_path(summary["method_dir"], str(method))
            or not host.same_data({k: old[k] for k in fields}, {k: cell[k] for k in fields})):
        raise ValueError("Original FNS evidence differs from the frozen cell binding")
    split_path = require_file(cell["tsd_split"])
    split = host.read_json(split_path)
    validate_split(split, cell)
    tsd = summary["json"]["tsd.json"]
    if (not host.same_data(tsd["identity"], split["identity"])
            or tsd["images"] != 8467 or tsd["stochastic_roi_passes"] != 169340
            or tsd["bytes"] != split_path.stat().st_size
            or tsd["sha256"] != hashlib.sha256(split_path.read_bytes()).hexdigest()):
        raise ValueError("Retained TSD differs from the completed original evidence")

    plan = host.read_json(evidence / "work/stages.json")
    specs = deepcopy(plan["stages"])
    if (plan["status"] != "invoked_not_completion_evidence" or plan["smoke_steps"] is not None
            or plan["budget"] != cell["aasfod_budget"] or len(specs) != 2):
        raise ValueError("Missing original alignment-to-FNS invocation proof")
    for spec in specs:
        command = spec.pop("command", [])
        entry = str(Path(cell["training_code"]) / "train.py")
        tail = [entry, str(work / (spec["stage"] + ".py")), "--work-dir", spec["work_dir"],
                "--gpus", "1", "--seed", "42", "--deterministic", "--no-validate"]
        if not command or entry not in command or not host.same_data(command[command.index(entry):], tail):
            raise ValueError("Missing original alignment-to-FNS invocation proof")
    if not host.same_data(specs, stage_specs(cell, work)):
        raise ValueError("Original stage binding/budget differs from stage_specs")
    # Captures are read-only copies of the producer's original artifacts.
    for relative in ("work/stages.json", "work/alignment.py", "work/fns.py"):
        if require_file(retained(method / relative)).read_bytes() != require_file(evidence / relative).read_bytes():
            raise ValueError(f"Original stage evidence changed: {relative}")
    for spec in specs:
        with host.native_config_paths():
            cfg = Config.fromfile(str(evidence / "work" / (spec["stage"] + ".py")),
                                  import_custom_modules=False)
        if (not host.same_path(cfg.load_from, spec["load_from"])
                or not host.same_path(cfg.model.ema_ckpt, spec["teacher_initialization"])
                or cfg.resume_from is not None or cfg.runner.max_iters != spec["updates"]
                or cfg.runner.type != "SemiIterBasedRunner" or cfg.optimizer != spec["optimizer"]
                or cfg.data.samples_per_gpu != spec["samples_per_gpu"]
                or cfg.data.train.stage != spec["stage"] or cfg.model.cfg.aasfod_stage != spec["stage"]
                or cfg.model.cfg.aasfod_ema_interval != 1
                or cfg.lr_config.get("warmup_iters", 0) != spec["warmup_iters"]
                or cfg.lr_config.policy != "Fixed"
                or spec["stage"] == "fns" and cfg.lr_config.get("warmup") is not None
                or not host.same_path(cfg.data.train.tsd_split, cell["tsd_split"])
                or not host.same_path(cfg.data.train.img_prefix, cell["target_val"])
                or cfg.data.train.unlabeled_epoch_size != 8467):
            raise ValueError("Original resolved stage config differs from its frozen specification")
    checkpoints = host.read_json(evidence / "checkpoint_evidence.json")["checkpoints"]
    for role, count in (("student", 419), ("teacher", 396)):
        path = specs[0][role + "_checkpoint"]
        records = [c for c in checkpoints if host.same_path(c["path"], path)]
        if (len(records) != 1 or records[0]["bytes"] != require_file(path).stat().st_size
                or records[0]["meta"]["iter"] != 159 or records[0]["meta"]["epoch"] != 1
                or records[0]["meta"]["seed"] != 42 or records[0]["state_tensors"] != count
                or records[0]["nonfinite_state_tensors"] != []):
            raise ValueError("Missing finite completed alignment checkpoint metadata")
    fns = retained(work / "fns")
    if not fns.is_dir() or any(fns.rglob("*.pth")):
        raise ValueError("Retained FNS checkpoints or missing failed stage require owner recovery")
    for stage, filename, marker in (
            ("alignment", "20260910_061657.log", "Saving checkpoint at 159 iterations"),
            ("fns", "20260910_062147.log", "max: 106 iters")):
        captured = require_file(evidence / "work" / stage / filename).read_bytes()
        if captured != require_file(retained(work / stage / filename)).read_bytes():
            raise ValueError("Original native stage log changed")
        text = captured.decode()
        if marker not in text or stage == "fns" and re.search(r"Iter \[\d+/|Saving checkpoint", text):
            raise ValueError("Native logs do not prove the supported pre-update FNS failure")
    owners = host.read_json(evidence / "owner_console_evidence.json")
    tails = [r["tail"] for r in owners
             if Path(r["path"]).name == "aasfod-RSAR-clean-42.recover-5427fdd.log"]
    if (len(tails) != 1 or not all(s in tails[0] for s in (
            "self.extract_feat(torch.stack(images))",
            "RuntimeError: stack expects each tensor to be equal size",
            "fns.py", "returned non-zero exit status 1"))):
        raise ValueError("Missing genuine owner mixed-size FNS failure evidence")
    return {**plan, "evidence_dir": str(evidence),
            "alignment_training_code": cell["training_code"],
            "alignment_training_code_sha": cell["training_code_sha"]}


def load_fns_continuation(record_path, cell):
    record = host.read_json(record_path)
    continuation = record["fns_continuation"]
    if (record["status"] != "archived" or record["phase"] != "train"
            or not host.same_data(continuation["binding"], cell)):
        raise ValueError("FNS continuation requires its selected archived recovery record")
    for move in record["moves"]:
        if not Path(move["archive"]).exists():
            raise ValueError("FNS continuation history archive is missing")
    proof = validate_fns_evidence(cell, continuation["evidence_dir"], record["moves"])
    code = validate_fns_code(continuation["model_code"], cell)
    if (Path(cell["work_dir"]) / "fns").exists():
        raise ValueError("FNS continuation already entered; preserve the new attempt")
    return code, proof


def stage_specs(cell, work, smoke_steps=None):
    """Pure CPU interface shared by execution and tests; no hidden extra epoch."""
    plan = cell["aasfod_budget"]
    if smoke_steps is not None and not 1 <= smoke_steps <= 4:
        raise ValueError("AASFOD smoke allows one to four optimizer updates per stage")
    previous = cell["source_checkpoint"]
    specs = []
    for stage in ("alignment", "fns"):
        steps = plan[stage + "_updates"] if smoke_steps is None else smoke_steps
        directory = Path(work) / stage
        specs.append(dict(
            stage=stage, updates=steps, work_dir=str(directory),
            load_from=previous, teacher_initialization=previous,
            samples_per_gpu=16 if stage == "alignment" else 32,
            optimizer=dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001),
            # Preserve the common100-update linear warmup once, not once per stage.
            warmup_iters=100 if stage == "alignment" else 0,
            ema_interval=plan["ema_interval"],
            student_checkpoint=str(directory / f"iter_{steps}.pth"),
            teacher_checkpoint=str(directory / f"iter_{steps}_ema.pth")))
        previous = specs[-1]["student_checkpoint"]
    return specs


def run(queue, dataset, domain, seed, smoke_steps=None, fns_continuation=None):
    if os.environ.get("IRAOD_GPU_LOCKED") != "1":
        raise RuntimeError("Use the existing owner's shared GPU lock")
    runtime, cell = load_cell(queue, dataset, domain, seed, "AASFOD")
    validate_split(host.read_json(require_file(cell["tsd_split"])), cell)
    code = Path(cell.get("training_code", runtime["training_code"]))
    proof = None
    if fns_continuation:
        if smoke_steps is not None:
            raise ValueError("FNS continuation cannot add smoke updates")
        code, proof = load_fns_continuation(fns_continuation, cell)
    if host.is_target_host():
        sys.path.insert(0, str(code))
        os.chdir(code)
    work = Path(cell["work_dir"])
    if smoke_steps is not None:
        work = work.with_name("smoke_work")
        if work.exists():
            raise FileExistsError(work)
    from mmcv import Config

    specs = stage_specs(cell, work, smoke_steps)
    if proof:
        specs[0] = {**proof["stages"][0],
                    "training_code": proof["alignment_training_code"],
                    "training_code_sha": proof["alignment_training_code_sha"],
                    "completion_evidence": proof["evidence_dir"], "preserved": True}
    work.mkdir(parents=True, exist_ok=True)
    for spec in specs[1:] if proof else specs:
        directory = Path(spec["work_dir"])
        if directory.exists():
            raise FileExistsError(directory)
        with host.native_config_paths():
            cfg = Config.fromfile(
                str(Path(proof["evidence_dir"]) / "work/fns.py") if proof else cell["config"])
        cfg.model.cfg.update(aasfod_stage=spec["stage"],
                             aasfod_ema_interval=spec["ema_interval"])
        # Runner loads the student; hook copies its entire detector into teacher.
        cfg.model.ema_ckpt = spec["teacher_initialization"]
        cfg.load_from = spec["load_from"]
        cfg.resume_from = None
        cfg.corrupt = domain
        cfg.data.samples_per_gpu = spec["samples_per_gpu"]
        cfg.data.train.update(stage=spec["stage"], tsd_split=cell["tsd_split"],
                              img_prefix=cell["target_val"],
                              unlabeled_epoch_size=cell["unlabeled_epoch_size"])
        cfg.runner = dict(type="SemiIterBasedRunner", max_iters=spec["updates"])
        cfg.optimizer = spec["optimizer"]
        cfg.lr_config = dict(policy="Fixed", by_epoch=False)
        if spec["warmup_iters"]:
            cfg.lr_config.update(warmup="linear", warmup_iters=spec["warmup_iters"],
                                 warmup_ratio=0.001)
        cfg.checkpoint_config = dict(by_epoch=False, interval=spec["updates"],
                                     save_last=True, max_keep_ckpts=2)
        cfg.work_dir = str(directory)
        config_path = work / (spec["stage"] + ".py")
        # MMCV renders transform instances as bare constructors, without imports
        # (same resolved-config boundary as f_deletion._resolved_text).
        config_path.write_text(
            "from torchvision.transforms import ColorJitter\n"
            + cfg.pretty_text
            + "\ndel ColorJitter\n")
        command = [runtime["python"], str(code / "train.py"), str(config_path),
                   "--work-dir", str(directory), "--gpus", "1", "--seed", str(seed),
                   "--deterministic", "--no-validate"]
        command = host.native_command(command, code / "train.py")
        spec["command"] = command
        if proof:
            spec.update(training_code=str(code), training_code_sha=FNS_MODEL_SHA,
                        config_binding=cell["config"], continuation_record=str(fns_continuation))
        write_json(work / "stages.json", {
            "status": "invoked_not_completion_evidence", "budget": cell["aasfod_budget"],
            "smoke_steps": smoke_steps, "stages": specs})
        subprocess.run(command, cwd=code, env={**os.environ, "PYTHONPATH": str(code)}, check=True)
        require_file(spec["student_checkpoint"])
        require_file(spec["teacher_checkpoint"])
    if smoke_steps is None:
        # Preserve the common report's epoch-end filename convention. Stage-local
        # native checkpoints record their own correct iter counts; stages.json
        # records total updates rather than pretending FNS itself took a full epoch.
        shutil.copyfile(specs[-1]["student_checkpoint"], cell["student_checkpoint"])
        shutil.copyfile(specs[-1]["teacher_checkpoint"], cell["checkpoint"])
    write_json(work / "stages.json", {
        "status": "complete" if smoke_steps is None else "smoke_not_formal",
        "budget": cell["aasfod_budget"], "smoke_steps": smoke_steps, "stages": specs})
    return work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "dataset", "domain"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--fns-continuation", help="Selected finite recovery record; never a full replay")
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
