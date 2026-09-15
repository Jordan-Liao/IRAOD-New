"""Prepared queue identity, real prerequisite admission, and finite ready-work routing."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import shlex
import subprocess

import torch

from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite
from experiments.comparison import mixed_queue
from experiments.comparison.aasfod_protocol import TSD_CHOICE
from experiments.comparison.result_completion import DOMAINS, write_json


class MixedQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queues = []
        self.cells = {}
        for index, method in enumerate(("B_REG", "F_text_only", "AASFOD", "SFYOLO")):
            queue = self.root / method
            queue.mkdir()
            cell = finite.Cell("DIOR", "clean", 42, method)
            work = self.root / "outputs" / method / ("ddp2/work" if cell.width == 2 else "work")
            binding = {
                "dataset": cell.dataset, "domain": cell.domain, "seed": cell.seed, "method": method,
                "world_size": cell.width, "samples_per_gpu": 32 // cell.width,
                "method_dir": str(work.parent), "work_dir": str(work),
                "checkpoint": str(work / "final_ema.pth"), "student_checkpoint": str(work / "final.pth"),
                "eval_dir": str(work.parent / "eval"), "terminal_status": str(work.parent / "terminal_status"),
                "config": f"/preserved/config{index}.py", "training_code": f"/preserved/code{index}",
                "training_code_sha": f"frozen{index}", "source_checkpoint": "/preserved/source0f98.pth",
                "target_val": "/preserved/DIOR/clean/val", "unlabeled_epoch_size": 5,
            }
            if method == "AASFOD":
                binding["tsd_split"] = str(work.parent / "tsd.json")
            if method == "SFYOLO":
                binding.update(tam_checkpoint=str(self.root / "TAM.pt"),
                               tam_identity=dict(dataset="DIOR", domain="clean", seed=42))
            self.cells[cell.key] = binding
            write_json(queue / "runtime.json", {
                "cells": {cell.key: binding}, "python": "/python",
                "training_code": f"/original/code{index}", "evaluation_code": f"/original/eval{index}",
                "evaluation_code_sha": finite.EVAL_SHA,
            })
            (queue / "with_gpu_lock.sh").write_text(f"LOCKDIR={self.root / 'locks/gpu'}\n")
            for phase in ("train", "eval"):
                (queue / f"{phase}.list").write_text(f"DIOR clean 42 {method}\n")
            self.queues.append(queue)
        self.out = mixed_queue.prepare(self.queues, self.root / "union")
        self.paths = finite.load_paths(self.out)

    def tsd(self):
        cell = self.cells["DIOR/clean/42/AASFOD"]
        split = {
            "identity": {k: cell[k] for k in
                         ("dataset", "domain", "seed", "source_checkpoint", "target_val")},
            "choice": TSD_CHOICE, "status": "complete",
            "scores": {name: i for i, name in enumerate("abcde")},
            "similar": ["e"], "dissimilar": list("abcd"),
        }
        Path(cell["tsd_split"]).parent.mkdir(parents=True, exist_ok=True)
        write_json(cell["tsd_split"], split)
        return cell, split

    def test_union_preserves_every_binding_and_origin_runtime(self):
        self.assertEqual(self.paths.DATA["cells"], self.cells)
        self.assertEqual(finite.allowed_gpus(self.paths), (4, 5, 6, 7))
        self.assertEqual(self.paths.DATA["pair_ports"], {"4,5": 29804, "6,7": 29806})
        for index, cell in enumerate(finite.load_cells([self.out / "train.list"], [])):
            runtime, binding = training.load_cell(self.out, cell.dataset, cell.domain, cell.seed, cell.method)
            self.assertEqual(binding, self.cells[cell.key])
            self.assertEqual(runtime["training_code"], f"/original/code{index}")
            self.assertEqual(self.paths.ema_path(cell.dataset, cell.domain, cell.seed, cell.method),
                             binding["checkpoint"])
            command = finite.runner_command(self.out, cell, "train", (4, 5) if cell.width == 2 else (6,))
            self.assertEqual(command[1], str(self.out / f"run_train_{cell.width}gpu.sh"))
            if cell.width == 2:
                self.assertEqual(command[2:4], ["4,5", "29804"])
                self.assertEqual(
                    finite.runner_command(self.out, cell, "train", (6, 7))[2:4],
                    ["6,7", "29806"])
        self.assertIn(str(training.ROOT), (self.out / "run_train_2gpu.sh").read_text())
        self.assertEqual(finite.lock_root(self.out), self.root / "locks/gpu")
        self.assertFalse((self.root / "outputs").exists())
        with self.assertRaises(finite.Blocked):
            finite.TmuxBackend(self.out, self.root / "run", (3, 4))
        backend = finite.TmuxBackend(self.out, self.root / "four_gpu_run", (4, 5, 6, 7))
        backend.selector.close()

    def test_tsd_missing_partial_mismatched_and_completed(self):
        cell = finite.Cell("DIOR", "clean", 42, "AASFOD")
        self.assertEqual(finite.prerequisite_state(self.paths, cell)[0], "waiting")
        binding, split = self.tsd()
        self.assertEqual(finite.prerequisite_state(self.paths, cell), ("ready", ""))
        for field, value in (("status", "smoke_not_formal"), ("identity", {})):
            write_json(binding["tsd_split"], {**split, field: value})
            self.assertEqual(finite.prerequisite_state(self.paths, cell)[0], "waiting")

    def test_tam_requires_completed_matching_fit_and_encoder_loader(self):
        cell = finite.Cell("DIOR", "clean", 42, "SFYOLO")
        binding = self.cells[cell.key]
        self.assertEqual(finite.prerequisite_state(self.paths, cell)[0], "waiting")
        Path(binding["tam_checkpoint"]).write_bytes(b"cpu fixture")
        with patch("experiments.comparison.tam_artifacts.load_completed_tam") as load:
            self.assertEqual(finite.prerequisite_state(self.paths, cell), ("ready", ""))
            load.assert_called_once_with(Path(binding["tam_checkpoint"]), binding["tam_identity"], "cpu")
            load.side_effect = ValueError("encoder tensor mismatch")
            self.assertIn("encoder tensor mismatch", finite.prerequisite_state(self.paths, cell)[1])
        binding["tam_identity"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "seed42"):
            training.require_prerequisites(binding)

    def test_actual_tam_payload_rejects_partial_wrong_domain_and_preprocessing(self):
        from experiments.comparison.tam_artifacts import NORMALIZATION, SCHEMA

        cell = finite.Cell("DIOR", "clean", 42, "SFYOLO")
        binding = self.cells[cell.key]
        payload = dict(schema=SCHEMA, status="complete", training_steps=160000,
                       identity=binding["tam_identity"], normalization=NORMALIZATION)
        for overrides in (
                dict(status="smoke_not_formal"), dict(training_steps=2),
                dict(identity=dict(dataset="DIOR", domain="cloudy", seed=42)),
                dict(normalization={})):
            torch.save({**payload, **overrides}, binding["tam_checkpoint"])
            state, reason = finite.prerequisite_state(self.paths, cell)
            self.assertEqual(state, "waiting")
            self.assertTrue(reason)

    def test_waiting_prerequisites_do_not_stop_independent_b_f(self):
        scope = finite.load_cells([self.out / "train.list"], [])
        started = []
        finished = set()
        active = []
        backend = SimpleNamespace(
            queue=self.out, run_dir=self.root / "run", producer=nullcontext,
            discover_external_owners=lambda *args: [], discover=lambda *args: [],
            available=lambda: {4, 5, 6},
            evidence=lambda cell, phase: "complete" if (cell, phase) in finished else "pending",
            admission=lambda cell, phase: (finite.prerequisite_state(self.paths, cell)
                                           if phase == "train" else ("ready", "")),
            finish=lambda *args: ("complete", ""),
            wait=lambda: list(active), close=lambda handle: active.remove(handle),
        )

        def start(cell, phase, gpus):
            started.append((cell.method, phase, gpus))
            finished.add((cell, phase))
            handle = dict(cell=cell, phase=phase, gpus=gpus, pid=100, spec=None)
            active.append(handle)
            return handle
        backend.start = start
        state = finite.run_finite(scope, backend)
        self.assertEqual(state["status"], "blocked")
        self.assertEqual({method for method, _, _ in started}, {"B_REG", "F_text_only"})
        rows = {r["cell"]["method"]: r for r in state["cells"]}
        for method in ("B_REG", "F_text_only"):
            self.assertEqual((rows[method]["train"], rows[method]["eval"]), ("complete", "complete"))
        for method in ("AASFOD", "SFYOLO"):
            self.assertEqual(rows[method]["train"], "waiting")
            self.assertTrue(rows[method]["train_input_reason"])

    def test_worker_and_native_train_refuse_missing_prerequisite_before_gpu_or_output(self):
        for method in ("AASFOD", "SFYOLO"):
            cell = finite.Cell("DIOR", "clean", 42, method)
            job = self.root / f"{method}-job.json"
            receipt = self.root / f"{method}-receipt.json"
            write_json(job, dict(cell=finite.asdict(cell), phase="train", gpus=[6],
                                 queue=str(self.out), receipt=str(receipt), tmux=["tmux"]))
            with patch.object(finite, "idle_devices") as idle, \
                    patch.object(finite.subprocess, "run") as run:
                self.assertEqual(finite.worker(job), 75)
                idle.assert_not_called()
                self.assertEqual(json.loads(receipt.read_text())["status"], "blocked")
                self.assertEqual(run.call_count, 1)  # completion notification only
            with patch.dict("os.environ", {"IRAOD_GPU_LOCKED": "1"}), \
                    patch.object(training.subprocess, "run") as run:
                with self.assertRaises(ValueError):
                    training.train(self.out, 6, "DIOR", "clean", 42, method)
                run.assert_not_called()
        self.assertFalse((self.root / "outputs").exists())

    def test_completed_tsd_admits_formal_single_and_reuses_source_split(self):
        self.tsd()
        cell = finite.Cell("DIOR", "clean", 42, "AASFOD")
        self.assertEqual(finite.prerequisite_state(self.paths, cell), ("ready", ""))
        self.assertEqual(finite.runner_command(self.out, cell, "train", (6,))[2:],
                         ["6", "DIOR", "clean", "42", "AASFOD"])

    def test_adopts_original_queue_wrapper_and_preserves_its_terminal_evidence(self):
        cell = finite.Cell("DIOR", "clean", 42, "B_REG")
        origin = self.queues[0]
        command = finite.runner_command(origin, cell, "train", (6,))
        self.assertEqual(finite.pane_job(cell.session("train"), {
            "command": shlex.join(command)}, self.out), (cell, "train", (6,), None))
        binding = self.cells[cell.key]
        work = Path(binding["work_dir"])
        work.mkdir(parents=True)
        for name in ("checkpoint", "student_checkpoint"):
            Path(binding[name]).write_bytes(b"existing completed native checkpoint")
        Path(binding["terminal_status"]).write_text("tmux_wrap_exit=0\n")
        self.assertEqual(finite.train_state(self.out, self.paths, cell), "complete")
        (origin / f"wrap_{cell.session('train')}.status").write_text("wrap_exit=1\n")
        self.assertEqual(finite.train_state(self.out, self.paths, cell), "blocked")


class PortStudentQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queues = {}
        self.bindings = {}
        for method in training.METHODS:
            queue = self.root / method
            queue.mkdir()
            bindings = {}
            for ds, domains in DOMAINS.items():
                for domain in domains:
                    for seed in training.SEEDS:
                        cell = finite.Cell(ds, domain, seed, method)
                        work = self.root / "outputs" / cell.key / "work"
                        iteration = ({"RSAR": 531, "DIOR": 369} if method == "SFYOLO"
                                     else {"RSAR": 266, "DIOR": 185})[ds]
                        bindings[cell.key] = dict(
                            dataset=ds, domain=domain, seed=seed, method=method, role="ema",
                            source_checkpoint=f"/source/{ds}.pth", source_id=f"{ds}-source",
                            config=f"/training/{method}/{ds}.py", eval_config=f"/native331/{ds}.py",
                            training_code=f"/prepared/{method}", training_code_sha=f"prepared-{method}",
                            method_dir=str(work.parent), work_dir=str(work),
                            checkpoint=str(work / f"iter_{iteration}_ema.pth"),
                            student_checkpoint=str(work / f"iter_{iteration}.pth"),
                            eval_dir=str(work.parent / f"eval_full_{domain}_ids_v1"),
                            terminal_status=str(work.parent / "terminal_status"),
                            final_checkpoint_iteration=iteration)
            write_json(queue / "runtime.json", dict(
                cells=bindings, python="/python", training_code=f"/original/{method}",
                evaluation_code=f"/native331/{method}", evaluation_code_sha=finite.EVAL_SHA))
            (queue / "with_gpu_lock.sh").write_text(f"LOCKDIR={self.root / 'locks/gpu'}\n")
            rows = "".join(f"{c['dataset']} {c['domain']} {c['seed']} {method}\n"
                           for c in bindings.values())
            for phase in ("train", "eval"):
                (queue / f"{phase}.list").write_text(rows)
            self.queues[method] = queue
            self.bindings.update(bindings)

    def students(self):
        return mixed_queue.prepare(list(self.queues.values()), self.root / "students",
                                   role="student", methods=training.STUDENT_METHODS)

    def test_180_explicit_students_preserve_finals_native_suffixes_and_source_bindings(self):
        originals = {q / "runtime.json": (q / "runtime.json").read_bytes()
                     for q in self.queues.values()}
        out = self.students()
        paths = finite.load_paths(out)
        scope = finite.load_cells([out / "train.list"], [out / "eval.list"])
        self.assertEqual(len(scope), 180)
        self.assertFalse(any(scope.values()))
        self.assertEqual((paths.DATA["train_cells"], paths.DATA["eval_cells"]), (0, 180))
        self.assertEqual(len(paths.DATA["student_cells"]), 180)
        for cell in scope:
            self.assertEqual((cell.role, cell.width), ("student", 1))
            canonical = self.bindings[cell.model.key]
            runtime, binding = training.load_cell(out, cell.dataset, cell.domain, cell.seed,
                                                  cell.method, "student")
            self.assertEqual(paths.DATA["cells"][cell.model.key], canonical)
            self.assertEqual(binding["checkpoint"], canonical["student_checkpoint"])
            self.assertEqual(binding["config"], canonical["eval_config"])
            self.assertEqual(binding["training_code"], canonical["training_code"])
            self.assertEqual(binding["source_checkpoint"], canonical["source_checkpoint"])
            self.assertEqual(runtime["evaluation_code"], f"/native331/{cell.method}")
            expected = str(Path(canonical["eval_dir"]).with_name(
                f"eval_full_{cell.domain}_student_ids_v1"))
            self.assertEqual(binding["eval_dir"], expected)
            self.assertEqual(paths.eval_student_dir(cell.dataset, cell.domain, cell.seed, cell.method),
                             expected)
            if cell.method == "SFYOLO":
                self.assertEqual(Path(binding["checkpoint"]).name,
                                 f"iter_{531 if cell.dataset == 'RSAR' else 369}.pth")
        for path, original in originals.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "outputs").exists())
        with self.assertRaisesRegex(ValueError, "outside"):
            finite.load_cells([out / "eval.list"], [])

    def test_union_with_future_training_flattens_routes_without_duplicate_collectors(self):
        future = mixed_queue.prepare(
            [self.queues[m] for m in ("B_REG", *training.F_DELETIONS, "AASFOD", "SFYOLO")],
            self.root / "future")
        default = finite.load_paths(future).DATA
        self.assertEqual((default["train_cells"], default["eval_cells"]), (180, 180))
        out = mixed_queue.prepare([future, self.students()], self.root / "combined")
        data = finite.load_paths(out).DATA
        self.assertEqual((data["train_cells"], data["eval_cells"]), (180, 360))
        scope = finite.load_cells([out / "train.list"], [out / "eval.list"])
        self.assertEqual(sum(scope.values()), 180)
        self.assertFalse(any(train for c, train in scope.items() if c.method in training.PORT_METHODS))
        identities = [(b["dataset"], b["domain"], b["seed"], b["method"], b["role"])
                      for field in ("cells", "student_cells") for b in data[field].values()]
        self.assertEqual(len(identities), len(set(identities)))
        for cell in scope:
            self.assertEqual(data["source_queues"][cell.key], str(self.queues[cell.method]))
            runtime, _ = training.load_cell(out, cell.dataset, cell.domain, cell.seed,
                                            cell.method, cell.role)
            self.assertEqual(runtime["training_code"], f"/original/{cell.method}")
            command = finite.runner_command(self.queues[cell.method], cell,
                                             "train" if scope[cell] else "eval",
                                             (4, 5) if scope[cell] and cell.width == 2 else (6,))
            self.assertEqual(finite.pane_job(cell.session("train" if scope[cell] else "eval"),
                                            {"command": shlex.join(command)}, out)[0], cell)

    def test_student_evaluator_uses_actual_execution_and_never_prerequisites(self):
        out = self.students()
        for method in training.STUDENT_METHODS:
            cell = finite.Cell("DIOR", "clean", 42, method, "student")
            original = self.bindings[cell.model.key]
            Path(original["method_dir"]).mkdir(parents=True)
            write_json(Path(original["method_dir"]) / "execution.json",
                       dict(training_code=f"/actual/{method}", training_code_sha=f"actual-{method}"))
            with patch.object(training, "evaluate_binding") as native, \
                    patch.object(training, "require_prerequisites") as prerequisites:
                training.evaluate(out, 7, "DIOR", "clean", 42, method, "student")
            binding, runtime, gpu = native.call_args.args
            self.assertEqual(binding["role"], "student")
            self.assertEqual(binding["checkpoint"], original["student_checkpoint"])
            self.assertEqual(binding["config"], original["eval_config"])
            self.assertEqual(binding["training_code"], f"/actual/{method}")
            self.assertEqual(binding["training_code_sha"], f"actual-{method}")
            self.assertEqual(runtime["evaluation_code"], f"/native331/{method}")
            self.assertEqual(gpu, 7)
            prerequisites.assert_not_called()
            self.assertFalse(Path(binding["eval_dir"]).exists())
        with self.assertRaisesRegex(ValueError, "GPU4,5,6,7"):
            training.evaluate(out, 3, "DIOR", "clean", 42, "IRG", "student")

    def test_generated_wrapper_passes_optional_sixth_role(self):
        out = self.root / "wrappers"
        out.mkdir()
        training.write_runners(out, "/bin/echo", self.queues["IRG"] / "with_gpu_lock.sh")
        command = ["bash", str(out / "run_eval_full.sh"), "6", "DIOR", "clean", "42", "IRG"]
        for arguments, role in (([], "ema"), (["student"], "student")):
            result = subprocess.check_output(command + arguments, text=True)
            self.assertEqual(shlex.split(result)[-2:], ["--role", role])

    def test_student_worker_and_adopted_wrapper_share_model_lock(self):
        out = self.students()
        student = finite.Cell("DIOR", "clean", 42, "IRG", "student")
        lock_path = self.root / "locks/cell_locks" / (student.model.session("train") + ".lock")
        lock = finite.take_lock(lock_path)
        self.addCleanup(lock.close)
        job, receipt = self.root / "job.json", self.root / "receipt.json"
        write_json(job, dict(cell=finite.asdict(student), phase="eval", gpus=[6],
                             queue=str(out), receipt=str(receipt), tmux=["tmux"]))
        with patch.object(finite.subprocess, "run") as run:
            self.assertEqual(finite.worker(job), 75)
            run.assert_not_called()
        self.assertIn("cell lock busy", json.loads(receipt.read_text())["reason"])
        lock.close()
        backend = finite.TmuxBackend(out, self.root / "run", (4, 5, 6))
        self.addCleanup(backend.selector.close)
        import os
        handle = backend.observe(student, "eval", {"pid": os.getpid(), "dead": False}, (6,))
        try:
            self.assertIsNone(finite.take_lock(lock_path))
        finally:
            backend.close(handle)
        # A Student uses the original EMA training wrapper evidence, not a fictitious Student trainer.
        binding = self.bindings[student.model.key]
        for field in ("checkpoint", "student_checkpoint"):
            Path(binding[field]).parent.mkdir(parents=True, exist_ok=True)
            Path(binding[field]).write_bytes(b"completed model")
        Path(binding["terminal_status"]).write_text("tmux_wrap_exit=0\n")
        origin = self.queues["IRG"]
        (origin / f"wrap_{student.model.session('train')}.status").write_text("wrap_exit=1\n")
        self.assertEqual(finite.train_state(out, finite.load_paths(out), student), "blocked")

    def test_student_waits_for_model_prerequisites_and_exit_without_concurrent_roles(self):
        model = finite.Cell("DIOR", "clean", 42, "SFYOLO")
        student = finite.Cell("DIOR", "clean", 42, "SFYOLO", "student")
        independent = finite.Cell("DIOR", "clean", 42, "B_REG")
        active, started, finished, waiting = [], [], set(), []
        prereq_ready = False

        def evidence(cell, phase):
            identity = cell.model if phase == "train" else cell
            if (identity, phase) in finished:
                return "complete"
            # A visible EMA file but absent Student must not permanently block while its model waits.
            return "blocked" if cell == student and phase == "train" else "pending"

        def start(cell, phase, gpus):
            self.assertFalse(any(h["cell"].model == cell.model for h in active))
            self.assertFalse(cell.role == "student" and phase == "train")
            started.append((cell, phase))
            handle = dict(cell=cell, phase=phase, gpus=gpus, pid=100, spec=None)
            active.append(handle)
            return handle

        def wait():
            nonlocal prereq_ready
            snapshot = json.loads((self.root / "run/state.json").read_text())
            waiting.append(next(r for r in snapshot["cells"] if r["cell"]["role"] == "student"))
            prereq_ready = True
            for h in active:
                finished.add((h["cell"], h["phase"]))
            return list(active)

        backend = SimpleNamespace(
            queue=self.root, run_dir=self.root / "run", producer=nullcontext,
            discover_external_owners=lambda *args: [], discover=lambda *args: [],
            available=lambda: {4, 5, 6}, evidence=evidence,
            admission=lambda cell, phase: (("waiting", "TAM missing")
                                           if cell == model and phase == "train" and not prereq_ready
                                           else ("ready", "")),
            start=start, finish=lambda *args: ("complete", ""), wait=wait,
            close=lambda handle: active.remove(handle))
        result = finite.run_finite({student: False, model: True, independent: True}, backend)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(waiting[0]["train"], "waiting")
        self.assertEqual(started.count((model, "train")), 1)
        self.assertEqual(started.count((student, "eval")), 1)
        self.assertEqual(started.count((model, "eval")), 1)


if __name__ == "__main__":
    unittest.main()
