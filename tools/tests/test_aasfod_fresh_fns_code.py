"""Fresh AASFOD finite-job -> CLI -> native stage selection, entirely on CPU."""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mmcv import Config

from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite, train_aasfod as stages, host_binding as host
from tools.tests import test_aasfod_config_import as configs
from tools.tests import test_aasfod_fns_continuation as recovery


FROZEN = "36c20531574a3d85d0dc6f76f18d0c52cfd011c3"


class FreshFNSCodeTest(unittest.TestCase):
    setUp = configs.AASFODConfigImportTest.setUp
    backend = recovery.FNSContinuationTest.backend
    model_checkout = recovery.FNSContinuationTest.model_checkout

    def fixture(self, dataset="RSAR"):
        cell = configs.AASFODConfigImportTest.fixture(self, dataset)
        original = self.model_checkout(FROZEN, "original36")
        accepted = self.model_checkout()
        method = Path(cell["work_dir"]).parent
        cell.update(method="AASFOD", method_dir=str(method), seed=43,
                    training_code=str(original), training_code_sha=FROZEN, world_size=1,
                    config=str(original / "configs/unbiased_teacher/sfod/extensions" /
                               f"aasfod_{dataset.lower()}.py"),
                    terminal_status=str(method / "terminal_status"))
        split = json.loads(Path(cell["tsd_split"]).read_text())
        split["identity"]["seed"] = 43
        finite.write_json(cell["tsd_split"], split)
        Path(cell["source_checkpoint"]).write_bytes(b"CPU source fixture")
        runtime = {"python": sys.executable, "training_code": str(original)}
        return runtime, cell, accepted

    def test_fresh_generated_job_routes_alignment_to36_and_fns_to_accepted_snapshot(self):
        self.check_fresh("RSAR")

    def test_fresh_dior_keeps_110_74_budget_and_185_final_labels(self):
        self.check_fresh("DIOR")

    def check_fresh(self, dataset):
        runtime, cell, accepted = self.fixture(dataset)
        original = Path(cell["training_code"])
        model = finite.Cell(cell["dataset"], cell["domain"], cell["seed"], "AASFOD")
        backend = self.backend(runtime, cell, None, "fresh")
        split = Path(cell["tsd_split"])
        retained = split.read_bytes(), split.stat().st_ino
        calls, handles, panes = [], [], {}

        @contextmanager
        def producer():
            backend.run_dir.mkdir()
            yield

        def cli(command, entry, main, env):
            with patch.dict(os.environ, env), patch.object(
                    sys, "argv", command[command.index(str(entry)):]):
                main()

        def native(command, **kwargs):
            if command[0] == "git":
                return recovery.NATIVE_RUN(command, **kwargs)
            if "wait-for" in command:
                return SimpleNamespace(returncode=0)
            if str(finite.SCRIPT.with_name("extension_training.py")) in command:
                self.assertNotIn("--fns-continuation", command)
                self.assertEqual(command[command.index("--aasfod-fns-code") + 1], str(accepted))
                cli(command, finite.SCRIPT.with_name("extension_training.py"), training.main, kwargs["env"])
                return SimpleNamespace(returncode=0)
            helper = configs.ROOT / "experiments/comparison/train_aasfod.py"
            if str(helper) in command:
                self.assertNotIn("--fns-continuation", command)
                self.assertEqual(command[command.index("--aasfod-fns-code") + 1], str(accepted))
                # Exercise the real native host shim and per-stage cwd selection;
                # only the helper runs in this scope, never host locks/probes.
                with patch.object(host, "is_target_host", return_value=True):
                    cli(command, helper, stages.main, kwargs["env"])
                return SimpleNamespace(returncode=0)
            code = Path(kwargs["cwd"])
            self.assertIn(code, (original, accepted))
            entry = str(code / "train.py")
            self.assertIn(entry, command)
            self.assertIn("native", command)
            cfg = configs.NATIVE_FROMFILE(command[command.index(entry) + 1],
                                         import_custom_modules=False)
            stage = cfg.model.cfg.aasfod_stage
            self.assertEqual(code, original if stage == "alignment" else accepted)
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(code))
            self.assertTrue(kwargs["check"])
            self.assertEqual(command[command.index("--seed") + 1], "43")
            spec = stages.stage_specs(cell, cell["work_dir"])[len(calls)]
            self.assertEqual(cfg.runner.max_iters, spec["updates"])
            self.assertEqual(cfg.data.samples_per_gpu, spec["samples_per_gpu"])
            self.assertEqual(cfg.data.train.unlabeled_epoch_size, cell["unlabeled_epoch_size"])
            self.assertEqual(cfg.data.train.tsd_split, cell["tsd_split"])
            self.assertEqual(cfg.data.train.stage, stage)
            self.assertEqual(cfg.load_from, spec["load_from"])
            self.assertEqual(cfg.model.ema_ckpt, spec["teacher_initialization"])
            self.assertIsNone(cfg.resume_from)
            self.assertEqual(cfg.optimizer, spec["optimizer"])
            self.assertEqual(cfg.lr_config.get("warmup_iters", 0), spec["warmup_iters"])
            self.assertEqual(cfg.model.cfg.aasfod_ema_interval, spec["ema_interval"])
            work = Path(cfg.work_dir)
            work.mkdir()
            for suffix in ("", "_ema"):
                (work / f"iter_{spec['updates']}{suffix}.pth").write_bytes(
                    f"CPU {stage} iter={spec['updates']} epoch=1 {suffix}".encode())
            calls.append(stage)
            return SimpleNamespace(returncode=0)

        def launch(*args, **kwargs):
            panes[model.session("train")] = dict(pid=os.getpid(), dead=False, command=args[-1])
            job = backend.run_dir / "jobs" / (model.session("train") + ".json")
            payload = json.loads(job.read_text())
            self.assertEqual(payload["aasfod_fns_code"], str(accepted))
            self.assertNotIn("fns_continuation", payload)
            self.assertEqual(finite.worker(job), 0)
            return SimpleNamespace(returncode=0)

        def observe(cell, phase, pane, gpus, spec=None):
            handle = dict(cell=cell, phase=phase, pid=pane["pid"], gpus=gpus, spec=spec)
            handles.append(handle)
            return handle

        with ExitStack() as stack:
            stack.enter_context(patch.object(Config, "fromfile", side_effect=lambda f, **kw:
                                configs.NATIVE_FROMFILE(f, import_custom_modules=False)))
            stack.enter_context(patch.object(backend, "producer", producer))
            stack.enter_context(patch.object(backend, "discover", return_value=[]))
            stack.enter_context(patch.object(backend, "available", return_value={4}))
            stack.enter_context(patch.object(backend, "panes", return_value=panes))
            stack.enter_context(patch.object(backend, "tmux_call", side_effect=launch))
            stack.enter_context(patch.object(backend, "observe", side_effect=observe))
            stack.enter_context(patch.object(backend, "wait", side_effect=lambda: handles[:]))
            stack.enter_context(patch.object(backend, "close"))
            stack.enter_context(patch.object(backend, "admission", side_effect=lambda c, p:
                                ("ready", "") if p == "train" else ("waiting", "CPU fixture: no TEST data")))
            stack.enter_context(patch.object(finite, "idle_devices", return_value={4}))
            stack.enter_context(patch.object(finite.subprocess, "run", side_effect=native))
            result = finite.run_finite({model: True}, backend, aasfod_fns_code=accepted)
        row = result["cells"][0]
        self.assertEqual((row["train"], row["eval"]), ("complete", "waiting"))
        self.assertEqual(calls, ["alignment", "fns"])
        self.assertEqual((split.read_bytes(), split.stat().st_ino), retained)
        self.assertFalse((backend.run_dir / "recovery").exists())
        plan = json.loads((Path(cell["work_dir"]) / "stages.json").read_text())
        self.assertEqual(plan["status"], "complete")
        self.assertEqual(plan["budget"], cell["aasfod_budget"])
        alignment, fns = plan["stages"]
        self.assertEqual(alignment["training_code"], str(original))
        self.assertEqual(alignment["training_code_sha"], FROZEN)
        self.assertEqual(fns["training_code"], str(accepted))
        self.assertEqual(fns["training_code_sha"], recovery.ACCEPTED)
        self.assertEqual(Path(cell["checkpoint"]).read_bytes(), Path(fns["teacher_checkpoint"]).read_bytes())
        execution = training.require_native_execution(cell)
        self.assertEqual(execution["training_code"], str(accepted))
        self.assertEqual(execution["training_code_sha"], recovery.ACCEPTED)
        self.assertEqual(execution["alignment_training_code"], str(original))
        self.assertEqual(execution["alignment_training_code_sha"], FROZEN)
        self.assertEqual(execution["config"], cell["config"])
        self.assertNotIn("preserved_alignment_updates", execution)
        self.assertEqual(Path(cell["checkpoint"]).name,
                         f"iter_{cell['aasfod_budget']['final_checkpoint_iteration']}_ema.pth")
        for role in ("ema", "student"):
            binding = {**cell, "eval_config": "/unchanged.py",
                       "checkpoint": cell["checkpoint" if role == "ema" else "student_checkpoint"]}
            with patch.object(training, "load_cell", return_value=(runtime, binding)), \
                    patch.object(training, "evaluate_binding") as evaluate:
                training.evaluate(backend.queue, 4, cell["dataset"], "clean", 43, "AASFOD", role)
                self.assertEqual(evaluate.call_args.args[0]["training_code_sha"], recovery.ACCEPTED)
                self.assertEqual(evaluate.call_args.args[0]["checkpoint"], binding["checkpoint"])

    def test_code_only_keeps_previous_failed_held_and_completed_rows_unchanged(self):
        runtime, cell, accepted = self.fixture()
        model = finite.Cell("RSAR", "clean", 43, "AASFOD")
        for phase in ("failed", "held", "complete"):
            with self.subTest(phase=phase):
                backend = self.backend(runtime, cell, None, phase)
                previous = self.root / (phase + "-previous.json")
                row = dict(cell=model.__dict__, train_requested=True, training_ownership="producer",
                           train="failed" if phase == "failed" else "complete" if phase == "complete" else "ready",
                           eval="blocked" if phase == "failed" else "complete" if phase == "complete" else "pending",
                           held=phase == "held", adopted=[], attempts=[],
                           reasons=["original history"])
                finite.write_json(previous, {"cells": [row]})
                old_bytes = previous.read_bytes()
                @contextmanager
                def producer():
                    backend.run_dir.mkdir()
                    yield
                with patch.object(backend, "producer", producer), \
                        patch.object(backend, "discover", return_value=[]), \
                        patch.object(backend, "start") as start:
                    result = finite.run_finite({model: True}, backend, previous_state=previous,
                                               aasfod_fns_code=accepted)
                start.assert_not_called()
                current = result["cells"][0]
                for key in ("train", "eval", "held", "attempts", "reasons"):
                    self.assertEqual(current[key], row[key])
                self.assertEqual(previous.read_bytes(), old_bytes)
                self.assertFalse((backend.run_dir / "recovery").exists())
        self.assertFalse(Path(cell["work_dir"]).exists())

    def test_new_fns_failure_retains_alignment_and_cannot_become_fresh_again(self):
        runtime, cell, accepted = self.fixture()
        backend = self.backend(runtime, cell, None)
        original = Path(cell["training_code"])
        calls = []

        def native(command, **kwargs):
            if command[0] == "git":
                return recovery.NATIVE_RUN(command, **kwargs)
            helper = configs.ROOT / "experiments/comparison/train_aasfod.py"
            if str(helper) in command:
                try:
                    stages.run(backend.queue, "RSAR", "clean", 43, aasfod_fns_code=accepted)
                except subprocess.CalledProcessError as error:
                    return SimpleNamespace(returncode=error.returncode)
                self.fail("The FNS failure must propagate through the helper")
            code = Path(kwargs["cwd"])
            cfg = configs.NATIVE_FROMFILE(command[command.index(str(code / "train.py")) + 1],
                                         import_custom_modules=False)
            stage = cfg.model.cfg.aasfod_stage
            calls.append(stage)
            work = Path(cfg.work_dir)
            work.mkdir()
            if stage == "fns":
                self.assertEqual(code, accepted)
                (work / "failure.log").write_text("CPU native FNS failure\n")
                raise subprocess.CalledProcessError(17, command)
            self.assertEqual(code, original)
            for suffix in ("", "_ema"):
                (work / f"iter_159{suffix}.pth").write_bytes(b"CPU completed alignment")
            return SimpleNamespace(returncode=0)

        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                patch.object(Config, "fromfile", side_effect=lambda f, **kw:
                             configs.NATIVE_FROMFILE(f, import_custom_modules=False)), \
                patch.object(training.subprocess, "run", side_effect=native):
            with self.assertRaises(subprocess.CalledProcessError):
                training.train(backend.queue, 4, "RSAR", "clean", 43, "AASFOD",
                               aasfod_fns_code=accepted)
            retained = {p: p.read_bytes() for p in Path(cell["method_dir"]).rglob("*") if p.is_file()}
            with self.assertRaises(FileExistsError):
                training.train(backend.queue, 4, "RSAR", "clean", 43, "AASFOD",
                               aasfod_fns_code=accepted)
        self.assertEqual(calls, ["alignment", "fns"])
        self.assertEqual({p: p.read_bytes() for p in retained}, retained)
        self.assertEqual(Path(cell["terminal_status"]).read_text(), "tmux_wrap_exit=17\n")
        self.assertFalse(Path(cell["checkpoint"]).exists())
        model = finite.Cell("RSAR", "clean", 43, "AASFOD")
        self.assertEqual(finite.train_state(backend.queue, backend.paths, model), "blocked")
        self.assertEqual(training.require_native_execution(cell)["training_code_sha"], recovery.ACCEPTED)

    def test_retained_alignment_needs_capture_even_with_code_and_selected_retry(self):
        owner = recovery.FNSContinuationTest()
        owner.root = self.root
        owner.addCleanup = self.addCleanup
        runtime, cell, evidence = owner.fixture()
        accepted = self.model_checkout()
        backend = self.backend(runtime, cell, None)
        model = finite.Cell("RSAR", "clean", 42, "AASFOD")
        previous = self.root / "previous.json"
        finite.write_json(previous, {"cells": [dict(
            cell=model.__dict__, train_requested=True, training_ownership="producer",
            train="failed", eval="blocked", held=False, adopted=[], attempts=[], reasons=["old failure"])]})
        retained = {p: p.read_bytes() for p in Path(cell["method_dir"]).rglob("*") if p.is_file()}
        @contextmanager
        def producer():
            backend.run_dir.mkdir()
            yield
        with patch.object(backend, "producer", producer), \
                patch.object(backend, "discover", return_value=[]):
            result = finite.run_finite({model: True}, backend, previous_state=previous,
                                       retry_cells=(model.key + ":train",), aasfod_fns_code=accepted)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("retained checkpoint", result["reason"])
        self.assertEqual({p: p.read_bytes() for p in retained}, retained)
        self.assertFalse((backend.run_dir / "recovery").exists())
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), self.assertRaises(FileExistsError):
            training.train(backend.queue, 4, "RSAR", "clean", 42, "AASFOD",
                           aasfod_fns_code=accepted)

    def test_bad_code_blocks_before_fresh_alignment_and_other_routes_are_unchanged(self):
        runtime, cell, accepted = self.fixture()
        backend = self.backend(runtime, cell, None)
        (accepted / "changed.py").write_text("changed = True\n")
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), self.assertRaises(ValueError):
            training.train(backend.queue, 4, "RSAR", "clean", 43, "AASFOD",
                           aasfod_fns_code=accepted)
        self.assertFalse(Path(cell["work_dir"]).exists())
        self.assertFalse((Path(cell["method_dir"]) / "execution.json").exists())
        backend.fns_code = str(accepted)
        backend.run_dir.mkdir()
        for model, phase in (
                (finite.Cell("RSAR", "clean", 43, "IRG"), "train"),
                (finite.Cell("RSAR", "clean", 43, "AASFOD"), "eval"),
                (finite.Cell("RSAR", "clean", 43, "AASFOD", "student"), "eval")):
            with patch.object(backend, "tmux_call", return_value=SimpleNamespace(returncode=0)), \
                    patch.object(backend, "panes", return_value={}):
                backend.start(model, phase, (4,))
            job = json.loads((backend.run_dir / "jobs" / (model.session(phase) + ".json")).read_text())
            self.assertNotIn("aasfod_fns_code", job)
            self.assertNotIn("fns_continuation", job)
            self.assertEqual(finite.runner_command(backend.queue, model, phase, (4,))[0], "bash")
            with self.assertRaises(ValueError):
                finite.runner_command(backend.queue, model, phase, (4,), aasfod_fns_code=accepted)


if __name__ == "__main__":
    unittest.main()
