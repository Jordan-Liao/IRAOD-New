"""CPU-only union of prepared detector queues; no rebinding or prerequisite jobs."""

import argparse
from pathlib import Path

from experiments.comparison.extension_training import ALLOWED_GPUS, PAIR_PORTS, STUDENT_METHODS, write_runners
from experiments.comparison.finite_resumer import Cell, load_cells, lock_root
from experiments.comparison.result_completion import read_json, write_json


def prepare(queues, out_dir, methods=None, role=None):
    """Optionally select only the prepared five-port Students, never their training."""
    if role not in (None, "student") or methods and role != "student":
        raise ValueError("Method selection requires explicit --role student")
    if methods and not set(methods).issubset(STUDENT_METHODS):
        raise ValueError("Student selection is limited to the five approved ports")
    selected = set(methods or STUDENT_METHODS)
    out = Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Mixed metadata directory must be NEW: {out}")
    cells, students, origins, runtimes, scopes = {}, {}, {}, [], {}
    for queue in queues:
        queue = Path(queue).resolve()
        runtime = read_json(queue / "runtime.json")
        runtimes.append((queue, runtime))
        if role == "student":
            scope = {Cell(*(binding[k] for k in ("dataset", "domain", "seed", "method")),
                          role="student"): False
                     for binding in runtime["cells"].values() if binding["method"] in selected}
        else:
            scope = load_cells([queue / "train.list"], [queue / "eval.list"])
        for cell, train in scope.items():
            key = cell.model.key
            binding = runtime["cells"][key]
            origin, source = queue, runtime
            while "source_queues" in source:
                origin = Path(source["source_queues"][key])
                source = read_json(origin / "runtime.json")
            if key in cells and (cells[key] != binding or origins[key] != str(origin)):
                raise ValueError(f"Conflicting original model binding: {key}")
            if cell in scopes:
                raise ValueError(f"Duplicate detector cell across prepared queues: {cell.key}")
            cells[key] = binding
            origins[key] = str(origin)
            if cell.role == "student":
                students[key] = (runtime["student_cells"][key] if role is None else {
                    **binding, "role": "student", "checkpoint": binding["student_checkpoint"],
                    "config": binding["eval_config"],
                    "eval_dir": str(Path(binding["eval_dir"]).with_name(
                        f"eval_full_{cell.domain}_student_ids_v1")),
                })
                student_origin, source = queue, runtime
                while "source_queues" in source:
                    student_origin = Path(source["source_queues"].get(cell.key, source["source_queues"][key]))
                    source = read_json(student_origin / "runtime.json")
                origins[cell.key] = str(student_origin)
            scopes[cell] = train
    if not scopes:
        raise ValueError("No selected prepared cells")
    first_queue, first = runtimes[0]
    shared_lock = lock_root(first_queue)
    for queue, runtime in runtimes:
        if (lock_root(queue) != shared_lock
                or runtime["python"] != first["python"]
                or runtime["evaluation_code_sha"] != first["evaluation_code_sha"]):
            raise ValueError("Union requires one shared lock, interpreter and native evaluation revision")
    # All scientific/path fields remain byte-for-byte JSON-equivalent bindings.
    # Per-cell runtime selection retains each original training/evaluation checkout.
    runtime = {
        "schema": "iraod-mixed-detector-queue-v1",
        "status": "prepared_not_execution_evidence",
        "python": first["python"],
        "evaluation_code_sha": first["evaluation_code_sha"],
        "allowed_gpus": list(ALLOWED_GPUS),
        "pair_ports": {",".join(map(str, pair)): port for pair, port in PAIR_PORTS.items()},
        "cells": cells, "student_cells": students, "source_queues": origins,
        "train_cells": sum(scopes.values()), "eval_cells": len(scopes),
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "runtime.json", runtime)
    write_json(out / "cells.json", cells)
    for phase in ("train", "eval"):
        (out / f"{phase}.list").write_text("".join(
            f"{c.dataset} {c.domain} {c.seed} {c.method}"
            + (" student" if c.role == "student" else "") + "\n"
            for c, train in scopes.items() if phase == "eval" or train))
    (out / "paths.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "from experiments.comparison.report_inputs import EXPECTED_IMAGES\n"
        "DATA = json.loads((Path(__file__).resolve().parent / 'runtime.json').read_text())\n"
        "EVALUATION_CODE_SHA = DATA['evaluation_code_sha']\n"
        "EXPECT_PRED = EXPECTED_IMAGES\n"
        "def binding(ds, domain, seed, method):\n"
        "    return DATA['cells'][f'{ds}/{domain}/{seed}/{method}']\n"
        "def method_dir(*args): return binding(*args)['method_dir']\n"
        "def ema_path(*args): return binding(*args)['checkpoint']\n"
        "def student_path(*args): return binding(*args)['student_checkpoint']\n"
        "def eval_full_dir(*args): return binding(*args)['eval_dir']\n"
        "def eval_student_dir(ds, domain, seed, method):\n"
        "    return DATA['student_cells'][f'{ds}/{domain}/{seed}/{method}']['eval_dir']\n")
    write_runners(out, first["python"], first_queue / "with_gpu_lock.sh")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", action="append", required=True, dest="queues")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--role", choices=("student",),
                        help="Emit only final Student evaluations from prepared EMA bindings")
    parser.add_argument("--method", action="append", dest="methods", choices=STUDENT_METHODS,
                        help="Student selection; defaults to all five ports present in the inputs")
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
