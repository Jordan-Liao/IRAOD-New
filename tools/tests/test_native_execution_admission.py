"""CPU regression for prepared-queue admission and the actual native eval entry."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite
from tools.tests.test_finite_resumer import PATHS_SOURCE


class NativeExecutionAdmissionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.gpu = training.ALLOWED_GPUS[0]
        self.execution = {
            "training_code": str(self.root / "actually executed training checkout"),
            "training_code_sha": "e" * 40,
            "status": "invoked_not_completion_evidence",
        }
        cells, students, origins = {}, {}, {}
        for method in ("IRG", "LPLD", "SFUT"):
            model = finite.Cell("RSAR", "clean", 43, method)
            method_dir = self.queue / "artifacts/RSAR/clean/43" / method
            work = method_dir / "work"
            work.mkdir(parents=True)
            ema, student = work / "iter_266_ema.pth", work / "iter_266.pth"
            ema.write_bytes(b"CPU fixture EMA")
            student.write_bytes(b"CPU fixture Student")
            (method_dir / "terminal_status").write_text("tmux_wrap_exit=0\n")
            annotations, images = method_dir / "annotations", method_dir / "images"
            annotations.mkdir()
            images.mkdir()
            binding = {
                **model.__dict__, "method_dir": str(method_dir),
                "checkpoint": str(ema), "student_checkpoint": str(student),
                "eval_config": str(self.root / "frozen_eval.py"),
                "ann_file": str(annotations), "img_prefix": str(images),
                "eval_dir": str(method_dir / "eval_full_clean_ids_v1"),
                "training_code": "/prepared/not-executed",
                "training_code_sha": "a" * 40,
            }
            cells[model.key] = binding
            students[model.key] = {
                **binding, "role": "student", "checkpoint": str(student),
                "eval_dir": str(method_dir / "eval_full_clean_student_ids_v1"),
            }
            for role in ("ema", "student"):
                origin = self.root / f"{method}-{role}-source"
                origin.mkdir()
                finite.write_json(origin / "runtime.json", {
                    "python": sys.executable, "evaluation_code": "/frozen/native-eval",
                    "evaluation_code_sha": training.EVALUATION_SHA,
                })
                origins[model.key + ("/student" if role == "student" else "")] = str(origin)
        finite.write_json(self.queue / "runtime.json", {
            "schema": "iraod-mixed-detector-queue-v1", "cells": cells,
            "student_cells": students, "source_queues": origins,
        })
        (self.queue / "paths.py").write_text(
            PATHS_SOURCE + "\nimport json\n"
            "DATA = json.loads((Path(__file__).parent / 'runtime.json').read_text())\n")
        (self.queue / "with_gpu_lock.sh").write_text(f"LOCKDIR={self.root / 'gpu_locks'}\n")
        self.paths = finite.load_paths(self.queue)

    def binding(self, cell):
        return training.load_cell(
            self.queue, cell.dataset, cell.domain, cell.seed, cell.method, cell.role)[1]

    def evaluate(self, cell):
        return training.evaluate(
            self.queue, self.gpu, cell.dataset, cell.domain, cell.seed, cell.method, cell.role)

    def test_missing_execution_waits_where_native_evaluate_raises(self):
        for method in ("IRG", "LPLD", "SFUT"):
            for role in ("ema", "student"):
                with self.subTest(method=method, role=role):
                    cell = finite.Cell("RSAR", "clean", 43, method, role)
                    binding = self.binding(cell)
                    self.assertEqual(finite.train_state(self.queue, self.paths, cell), "complete")
                    self.assertEqual(finite.eval_state(self.queue, self.paths, cell), "pending")
                    with patch.object(training, "evaluate_binding") as native:
                        with self.assertRaisesRegex(FileNotFoundError, "execution.json"):
                            self.evaluate(cell)
                    native.assert_not_called()
                    state, reason = finite.input_state(self.queue, self.paths, cell, "eval")
                    self.assertEqual(state, "waiting")
                    self.assertIn(str(Path(binding["method_dir"]) / "execution.json"), reason)
                    self.assertFalse(Path(binding["eval_dir"]).exists())

    def test_malformed_required_identity_is_not_admitted_or_evaluated(self):
        cases = [
            ("invalid-json", "{"),
            ("non-object", "[]"),
            *[(f"missing-{field}", json.dumps({k: v for k, v in self.execution.items()
                                              if k != field}))
              for field in ("training_code", "training_code_sha")],
            *[(f"{field}-{value!r}", json.dumps({**self.execution, field: value}))
              for field in ("training_code", "training_code_sha")
              for value in (None, "", "  ", 42)],
        ]
        for name, payload in cases:
            for role in ("ema", "student"):
                with self.subTest(case=name, role=role):
                    cell = finite.Cell("RSAR", "clean", 43, "IRG", role)
                    binding = self.binding(cell)
                    (Path(binding["method_dir"]) / "execution.json").write_text(payload)
                    with patch.object(training, "evaluate_binding") as native:
                        with self.assertRaises(ValueError):
                            self.evaluate(cell)
                    native.assert_not_called()
                    state, reason = finite.input_state(self.queue, self.paths, cell, "eval")
                    self.assertEqual(state, "waiting")
                    self.assertTrue(reason.startswith("eval input: "))
                    if name.startswith("missing-"):
                        self.assertIn(name.removeprefix("missing-"), reason)
                    self.assertFalse(Path(binding["eval_dir"]).exists())

    def test_valid_execution_preserves_actual_identity_for_both_native_roles(self):
        for method in ("IRG", "LPLD", "SFUT"):
            for role in ("ema", "student"):
                with self.subTest(method=method, role=role):
                    cell = finite.Cell("RSAR", "clean", 43, method, role)
                    binding = self.binding(cell)
                    metadata = Path(binding["method_dir"]) / "execution.json"
                    finite.write_json(metadata, self.execution)
                    original = metadata.read_bytes()
                    self.assertEqual(finite.input_state(self.queue, self.paths, cell, "eval"),
                                     ("ready", ""))
                    with patch.object(training, "evaluate_binding", return_value=17) as native:
                        self.assertEqual(self.evaluate(cell), 17)
                    actual, runtime, gpu = native.call_args.args
                    self.assertEqual(actual, {
                        **binding, "role": role, "config": binding["eval_config"],
                        "training_code": self.execution["training_code"],
                        "training_code_sha": self.execution["training_code_sha"],
                    })
                    self.assertEqual(runtime["evaluation_code_sha"], training.EVALUATION_SHA)
                    self.assertEqual(gpu, self.gpu)
                    self.assertEqual(metadata.read_bytes(), original)
                    self.assertFalse(Path(binding["eval_dir"]).exists())

    def test_waiting_producer_creates_no_attempt_job_output_or_gpu_activity(self):
        cells = {finite.Cell("RSAR", "clean", 43, "IRG", role): False
                 for role in ("ema", "student")}
        backend = finite.TmuxBackend(self.queue, self.root / "run", (self.gpu,))
        self.addCleanup(backend.selector.close)
        with patch.object(backend, "producer", return_value=nullcontext()), \
                patch.object(backend, "discover", return_value=[]), \
                patch.object(backend, "start") as start, \
                patch.object(finite, "idle_devices", return_value=set()) as probe, \
                patch.object(finite, "take_lock") as lock:
            result = finite.run_finite(cells, backend)
        self.assertEqual(result["status"], "blocked")
        for row in result["cells"]:
            self.assertEqual((row["train"], row["eval"], row["attempts"]),
                             ("complete", "waiting", []))
            self.assertIn("execution.json", row["eval_input_reason"])
        start.assert_not_called()
        probe.assert_not_called()
        lock.assert_not_called()
        self.assertFalse((backend.run_dir / "jobs").exists())
        self.assertFalse((backend.run_dir / "receipts").exists())
        for cell in cells:
            self.assertFalse(Path(self.binding(cell)["eval_dir"]).exists())

    def test_worker_rechecks_execution_before_gpu_lock_probe_or_runner(self):
        for role in ("ema", "student"):
            with self.subTest(role=role):
                cell = finite.Cell("RSAR", "clean", 43, "IRG", role)
                receipt = self.root / f"{role}-receipt.json"
                job = self.root / f"{role}-job.json"
                finite.write_json(job, {
                    "cell": cell.__dict__, "phase": "eval", "gpus": [self.gpu],
                    "queue": str(self.queue), "receipt": str(receipt), "tmux": ["tmux"],
                })
                with patch.object(finite, "idle_devices") as probe, \
                        patch.object(finite, "take_lock", wraps=finite.take_lock) as lock, \
                        patch.object(finite.subprocess, "run") as run:
                    self.assertEqual(finite.worker(job), 75)
                self.assertEqual(lock.call_count, 1)  # Model serialization, never a GPU lock.
                self.assertEqual(lock.call_args.args[0].parent.name, "cell_locks")
                probe.assert_not_called()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][:3], ["tmux", "wait-for", "-S"])
                result = json.loads(receipt.read_text())
                self.assertEqual(result["status"], "blocked")
                self.assertIn("execution.json", result["reason"])
                self.assertFalse(Path(self.binding(cell)["eval_dir"]).exists())

    def test_historical_nonruntime_queue_retains_admission(self):
        (self.queue / "paths.py").write_text(PATHS_SOURCE)
        paths = finite.load_paths(self.queue)
        cell = finite.Cell("RSAR", "clean", 43, "IRG")
        self.assertEqual(finite.input_state(self.queue, paths, cell, "eval"), ("ready", ""))


if __name__ == "__main__":
    unittest.main()
