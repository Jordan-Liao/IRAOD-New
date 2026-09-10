"""CPU continuation at the real stage/config and finite recovery boundaries."""

from contextlib import ExitStack, contextmanager
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mmcv import Config

from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite
from experiments.comparison import host_binding as host, train_aasfod as stages
from tools.tests import test_aasfod_config_import as config_tests

NATIVE_FROMFILE, ROOT = config_tests.NATIVE_FROMFILE, config_tests.ROOT


ACCEPTED = "cbd0f75ab147fea6728f61d6fd696325cc195296"
NATIVE_RUN = subprocess.run


class FNSContinuationTest(unittest.TestCase):
    setUp = config_tests.AASFODConfigImportTest.setUp
    entry_scope = config_tests.AASFODConfigImportTest.entry_scope

    def fixture(self, dataset="RSAR"):
        cell = config_tests.AASFODConfigImportTest.fixture(self, dataset)
        method = Path(cell["work_dir"]).parent
        cell.update(method="AASFOD", method_dir=str(method),
                    training_code_sha="36c20531574a3d85d0dc6f76f18d0c52cfd011c3",
                    terminal_status=str(method / "terminal_status"), world_size=1)
        Path(cell["source_checkpoint"]).write_bytes(b"source fixture")
        runtime = {"python": sys.executable, "training_code": str(ROOT)}
        with ExitStack() as stack:
            self.entry_scope(stack, cell)

            def native(command, **kwargs):
                cfg = NATIVE_FROMFILE(
                    command[command.index(str(ROOT / "train.py")) + 1],
                    import_custom_modules=False)
                work = Path(cfg.work_dir)
                work.mkdir()
                if cfg.data.train.stage == "alignment":
                    for suffix in ("", "_ema"):
                        (work / f"iter_159{suffix}.pth").write_bytes(
                            f"small CPU alignment payload {suffix}".encode())
                    (work / "20260910_061657.log").write_text(
                        "Iter [150/159] loss: 0.4982\nSaving checkpoint at 159 iterations\n")
                else:
                    (work / "20260910_062147.log").write_text(
                        "workflow: [('train', 1)], max: 106 iters\n")
                    raise subprocess.CalledProcessError(1, command)
            stack.enter_context(patch.object(stages.subprocess, "run", side_effect=native))
            with self.assertRaises(subprocess.CalledProcessError):
                stages.run("/queue", dataset, "clean", 42)
        finite.write_json(method / "execution.json", {
            **cell, "orchestration_code_sha": "02fb860-old-attempt",
            "status": "invoked_not_completion_evidence"})
        (method / "terminal_status").write_text("tmux_wrap_exit=1\n")
        (method / "train.log").write_text("NameError: name 'ColorJitter' is not defined\n")
        evidence = self.root / "evidence"
        evidence.mkdir()
        for path in (method / "work").rglob("*"):
            if path.is_file() and path.suffix != ".pth":
                target = evidence / path.relative_to(method)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        specs = json.loads((method / "work/stages.json").read_text())["stages"]
        checkpoints = []
        for role, count in (("student", 419), ("teacher", 396)):
            path = Path(specs[0][role + "_checkpoint"])
            checkpoints.append(dict(
                path=str(path), bytes=path.stat().st_size,
                meta=dict(iter=159, epoch=1, seed=42, exp_name="alignment.py"),
                state_tensors=count, nonfinite_state_tensors=[]))
        finite.write_json(evidence / "checkpoint_evidence.json", {"checkpoints": checkpoints})
        finite.write_json(evidence / "owner_console_evidence.json", [{
            "path": str(self.root / "prereq_runs_20260909/aasfod-RSAR-clean-42.recover-5427fdd.log"),
            "tail": "features = self.extract_feat(torch.stack(images))\n"
                    "RuntimeError: stack expects each tensor to be equal size, but got "
                    "[3, 256, 256] at entry 0 and [3, 800, 800] at entry 7\n"
                    "subprocess.CalledProcessError: Command "
                    + repr(specs[1]["command"]) + " returned non-zero exit status 1.\n"}])
        split = Path(cell["tsd_split"])
        finite.write_json(evidence / "EVIDENCE.json", {
            "method_dir": str(method), "json": {
                "execution.json": json.loads((method / "execution.json").read_text()),
                "tsd.json": {
                    "identity": json.loads(split.read_text())["identity"],
                    "images": 8467, "stochastic_roi_passes": 169340,
                    "bytes": split.stat().st_size,
                    "sha256": hashlib.sha256(split.read_bytes()).hexdigest()}}})
        return runtime, cell, evidence

    def test_completed_alignment_evidence_is_accepted_without_stage_exit_files(self):
        _, cell, evidence = self.fixture()
        proof = stages.validate_fns_evidence(cell, evidence)
        self.assertEqual(proof["stages"][0]["updates"], 159)
        self.assertEqual(proof["stages"][1]["updates"], 106)

    def backend(self, runtime, cell, evidence, name="continuation"):
        queue = self.root / "queue"
        queue.mkdir(exist_ok=True)
        key = "/".join(str(cell[k]) for k in ("dataset", "domain", "seed", "method"))
        finite.write_json(queue / "runtime.json", {**runtime, "cells": {
            key: cell}})
        (queue / "with_gpu_lock.sh").write_text(
            f"LOCKDIR={shlex.quote(str(queue / 'gpu_locks'))}\n")
        (queue / "paths.py").write_text(
            "import json\nfrom pathlib import Path\n"
            "C=json.loads((Path(__file__).parent/'runtime.json').read_text())['cells']"
            f"[{key!r}]\n"
            "def binding(*a): return C\n"
            "def method_dir(*a): return C['method_dir']\n"
            "def ema_path(*a): return C['checkpoint']\n"
            "def student_path(*a): return C['student_checkpoint']\n"
            "def eval_full_dir(*a): return str(Path(C['method_dir'])/'eval')\n")
        backend = finite.TmuxBackend(queue, self.root / name, (4,))
        self.addCleanup(backend.selector.close)
        backend.fns_options = (dict(evidence_dir=str(evidence), model_code=str(self.root / "model-code"))
                               if evidence else None)
        return backend

    def model_checkout(self, sha=ACCEPTED, name="model-code"):
        code = self.root / name
        NATIVE_RUN(["git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(code)],
                   check=True)
        NATIVE_RUN(["git", "-C", str(code), "checkout", "--quiet", "--detach", sha], check=True)
        return code

    def test_selected_previous_state_worker_native_stage_and_eval_identity(self):
        runtime, cell, evidence = self.fixture()
        code = self.model_checkout()
        backend = self.backend(runtime, cell, evidence)
        model = finite.Cell("RSAR", "clean", 42, "AASFOD")
        held = finite.Cell("RSAR", "noise_suppression", 44, "IRG", "student")
        other = finite.Cell("DIOR", "clean", 44, "SFUT")
        previous = self.root / "previous.json"
        rows = [
            dict(cell=c.__dict__, train_requested=requested,
                 training_ownership="producer" if requested else "eval_only",
                 train="failed" if requested else "blocked", eval="blocked",
                 reasons=["original failure"], attempts=[{"phase": "train", "status": "failed"}]
                 if requested else [], adopted=[], held=c == held)
            for c, requested in ((model, True), (held, False), (other, True))]
        finite.write_json(previous, {"queue": str(backend.queue), "cells": rows})
        old_ledger = previous.read_bytes()
        split = Path(cell["tsd_split"])
        preserved = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                     for p in [split, *Path(cell["work_dir"]).glob("alignment/*.pth"),
                               Path(cell["work_dir"]) / "alignment.py"]}
        old_fns = (Path(cell["work_dir"]) / "fns/20260910_062147.log").read_bytes()
        calls, configs = [], []
        panes, handles = {}, []

        @contextmanager
        def producer():
            backend.run_dir.mkdir()
            yield

        def command_run(command, **kwargs):
            if command[0] == "git":
                return NATIVE_RUN(command, **kwargs)
            if "wait-for" in command:
                return SimpleNamespace(returncode=0)
            if str(finite.SCRIPT.with_name("extension_training.py")) in command:
                self.assertIn("--fns-continuation", command)
                with patch.dict(os.environ, kwargs["env"]):
                    training.train(str(backend.queue), 4, "RSAR", "clean", 42, "AASFOD",
                                   command[command.index("--fns-continuation") + 1],
                                   command[command.index("--aasfod-fns-code") + 1])
                return SimpleNamespace(returncode=0)
            if str(ROOT / "experiments/comparison/train_aasfod.py") in command:
                self.assertEqual(kwargs["cwd"], code)
                with patch.dict(os.environ, kwargs["env"]):
                    stages.run(str(backend.queue), "RSAR", "clean", 42,
                               fns_continuation=command[command.index("--fns-continuation") + 1],
                               aasfod_fns_code=command[command.index("--aasfod-fns-code") + 1])
                return SimpleNamespace(returncode=0)
            self.assertIn(str(code / "train.py"), command)
            self.assertEqual(kwargs["cwd"], code)
            self.assertEqual(kwargs["env"]["PYTHONPATH"], str(code))
            probe = (
                "from pathlib import Path\n"
                "from types import SimpleNamespace\n"
                "from unittest.mock import patch\n"
                "from mmcv.utils import ext_loader\n"
                "def unused(*a, **k): raise AssertionError('no compiled ops on CPU')\n"
                "with patch.object(ext_loader, 'load_ext', side_effect=lambda n, fs: "
                "SimpleNamespace(**{f: unused for f in fs})):\n"
                " import sfod.extensions.aasfod as model\n"
                f" assert Path(model.__file__).resolve().is_relative_to(Path({str(code)!r}))\n")
            NATIVE_RUN([sys.executable, "-c", probe], cwd=code,
                       env={**kwargs["env"], "CUDA_VISIBLE_DEVICES": "",
                            "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            cfg = NATIVE_FROMFILE(command[command.index(str(code / "train.py")) + 1],
                                 import_custom_modules=False)
            configs.append(cfg)
            calls.append(cfg.data.train.stage)
            self.assertEqual(cfg.runner.max_iters, 106)
            self.assertEqual(cfg.optimizer, dict(type="SGD", lr=.02, momentum=.9, weight_decay=.0001))
            self.assertEqual(cfg.lr_config, dict(policy="Fixed", by_epoch=False))
            self.assertIsNone(cfg.resume_from)
            self.assertEqual(cfg.data.samples_per_gpu, 32)
            self.assertEqual(cfg.load_from, str(Path(cell["work_dir"]) / "alignment/iter_159.pth"))
            self.assertEqual(cfg.model.ema_ckpt, cfg.load_from)
            self.assertEqual(cfg.model.cfg.aasfod_ema_interval, 1)
            with patch.object(host, "is_target_host", return_value=True):
                generated = host.native_command(command, code / "train.py")
            self.assertIn("native", generated)
            self.assertIn(str(code / "train.py"), generated)
            work = Path(cfg.work_dir)
            work.mkdir()
            import torch
            for suffix in ("", "_ema"):
                torch.save({"meta": {"iter": 106, "epoch": 1},
                            "state_dict": {"cpu_fixture": torch.ones(1)}},
                           work / f"iter_106{suffix}.pth")
            return SimpleNamespace(returncode=0)

        def tmux(*args, **kwargs):
            panes[model.session("train")] = {
                "pid": os.getpid(), "dead": False, "command": args[-1]}
            job_file = backend.run_dir / "jobs" / (model.session("train") + ".json")
            self.assertEqual(finite.worker(job_file), 0)
            return SimpleNamespace(returncode=0)

        def observe(cell, phase, pane, gpus, spec=None):
            handle = dict(cell=cell, phase=phase, gpus=gpus, spec=spec, pid=pane["pid"])
            handles.append(handle)
            return handle

        with ExitStack() as stack:
            stack.enter_context(patch.object(Config, "fromfile", side_effect=lambda f, **kw:
                                NATIVE_FROMFILE(f, import_custom_modules=False)))
            stack.enter_context(patch.object(backend, "producer", producer))
            stack.enter_context(patch.object(backend, "discover", return_value=[]))
            stack.enter_context(patch.object(backend, "available", return_value={4}))
            stack.enter_context(patch.object(backend, "panes", return_value=panes))
            stack.enter_context(patch.object(backend, "observe", side_effect=observe))
            stack.enter_context(patch.object(backend, "wait", side_effect=lambda: handles[:]))
            stack.enter_context(patch.object(backend, "close"))
            stack.enter_context(patch.object(backend, "tmux_call", side_effect=tmux))
            # No eval data is synthesized; valid training reaches native eval admission.
            stack.enter_context(patch.object(backend, "admission", side_effect=lambda c, p:
                                ("ready", "") if p == "train" else ("waiting", "CPU test has no TEST data")))
            stack.enter_context(patch.object(finite, "idle_devices", return_value={4}))
            stack.enter_context(patch.object(finite.subprocess, "run", side_effect=command_run))
            result = finite.run_finite(
                {model: True, held: False, other: True}, backend, previous_state=previous,
                retry_cells=(model.key + ":train",), aasfod_fns_evidence=evidence,
                aasfod_fns_code=code)
        selected = next(r for r in result["cells"] if r["cell"] == model.__dict__)
        self.assertEqual(selected["train"], "complete")
        self.assertEqual(selected["eval"], "waiting")
        self.assertEqual(calls, ["fns"])
        self.assertEqual([a["status"] for a in selected["attempts"]],
                         ["failed", "retry_authorized", "complete"])
        self.assertEqual(previous.read_bytes(), old_ledger)
        for expected in rows[1:]:
            self.assertEqual(next(r for r in result["cells"] if r["cell"] == expected["cell"]), expected)
        for path, snapshot in preserved.items():
            self.assertEqual((path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns), snapshot)
        record = json.loads(Path(backend.fns_recovery).read_text())
        archive = next(Path(m["archive"]) for m in record["moves"] if m["source"].endswith("/fns"))
        self.assertEqual((archive / "20260910_062147.log").read_bytes(), old_fns)
        plan = json.loads((Path(cell["work_dir"]) / "stages.json").read_text())
        self.assertEqual(plan["status"], "complete")
        self.assertEqual(plan["budget"]["total_updates"], 265)
        self.assertTrue(plan["stages"][0]["preserved"])
        self.assertEqual(plan["stages"][0]["training_code_sha"], cell["training_code_sha"])
        self.assertEqual(plan["stages"][1]["training_code_sha"], ACCEPTED)
        self.assertEqual(Path(cell["checkpoint"]).read_bytes(),
                         (Path(cell["work_dir"]) / "fns/iter_106_ema.pth").read_bytes())
        self.assertEqual(finite.train_state(backend.queue, backend.paths, model), "complete")
        execution = training.require_native_execution(cell)
        self.assertEqual(execution["training_code"], str(code))
        self.assertEqual(execution["training_code_sha"], ACCEPTED)
        for role in ("ema", "student"):
            binding = {**cell, "eval_config": "/unchanged-eval.py",
                       "checkpoint": cell["checkpoint" if role == "ema" else "student_checkpoint"]}
            with patch.object(training, "load_cell", return_value=(runtime, binding)), \
                    patch.object(training, "evaluate_binding") as evaluate:
                training.evaluate(backend.queue, 4, "RSAR", "clean", 42, "AASFOD", role)
                actual = evaluate.call_args.args[0]
                self.assertEqual(actual["training_code_sha"], ACCEPTED)
                self.assertEqual(actual["checkpoint"], binding["checkpoint"])
        receipt = json.loads((backend.run_dir / "results" /
                              (model.session("train") + ".json")).read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["exit_code"], 0)
        with self.assertRaises((ValueError, FileExistsError)):
            stages.load_fns_continuation(backend.fns_recovery, cell)

    def test_invalid_proof_budget_binding_and_retained_outputs_are_blocked(self):
        _, cell, evidence = self.fixture()
        cases = [
            ("missing invocation", evidence / "work/stages.json",
             lambda d: d["stages"][1].pop("command")),
            ("completed stage", evidence / "work/stages.json",
             lambda d: d.update(status="complete")),
            ("wrong budget", evidence / "work/stages.json",
             lambda d: d["budget"].update(fns_updates=107)),
            ("invalid metadata", evidence / "checkpoint_evidence.json",
             lambda d: d["checkpoints"][0]["meta"].update(iter=158)),
            ("nonfinite metadata", evidence / "checkpoint_evidence.json",
             lambda d: d["checkpoints"][0].update(nonfinite_state_tensors=["bad"])),
            ("wrong source", evidence / "EVIDENCE.json",
             lambda d: d["json"]["execution.json"].update(source_checkpoint="/wrong.pth")),
            ("wrong TSD hash", evidence / "EVIDENCE.json",
             lambda d: d["json"]["tsd.json"].update(sha256="wrong")),
        ]
        for name, path, mutate in cases:
            with self.subTest(name=name):
                original = path.read_bytes()
                value = json.loads(original)
                mutate(value)
                finite.write_json(path, value)
                with self.assertRaises((ValueError, KeyError)):
                    stages.validate_fns_evidence(cell, evidence)
                path.write_bytes(original)
        for relative in ("work/fns/iter_106.pth", "work/fns/iter_106_ema.pth",
                         "work/iter_266.pth", "work/iter_266_ema.pth"):
            path = Path(cell["method_dir"]) / relative
            path.write_bytes(b"retained checkpoint")
            with self.subTest(path=relative), self.assertRaises(ValueError):
                stages.validate_fns_evidence(cell, evidence)
            path.unlink()
        for changed in (dict(seed=43), dict(unlabeled_epoch_size=8466),
                        dict(aasfod_budget={}), dict(training_code_sha=ACCEPTED)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                stages.validate_fns_evidence({**cell, **changed}, evidence)
        log = evidence / "work/alignment/20260910_061657.log"
        log.unlink()
        with self.assertRaises(ValueError):
            stages.validate_fns_evidence(cell, evidence)

    def test_pre_stage_guard_stays_closed_without_explicit_continuation(self):
        runtime, cell, evidence = self.fixture()
        backend = self.backend(runtime, cell, evidence)
        backend.fns_options = None
        original = {p: p.read_bytes() for p in Path(cell["method_dir"]).rglob("*") if p.is_file()}
        with self.assertRaisesRegex(finite.Blocked, "retained checkpoint"):
            backend.archive_retry(finite.Cell("RSAR", "clean", 42, "AASFOD"), "train")
        self.assertEqual({p: p.read_bytes() for p in original}, original)

    def test_dirty_or_bound_model_code_is_not_accepted(self):
        _, cell, _ = self.fixture()
        code = self.model_checkout()
        self.assertEqual(stages.validate_fns_code(code, cell), code)
        with self.assertRaises(ValueError):
            stages.validate_fns_code(code, {**cell, "training_code": str(code)})
        (code / "unexpected.py").write_text("changed = True\n")
        with self.assertRaises(ValueError):
            stages.validate_fns_code(code, cell)

    def test_holds_live_jobs_busy_locks_and_missing_selection_never_archive(self):
        runtime, cell, evidence = self.fixture()
        model = finite.Cell("RSAR", "clean", 42, "AASFOD")
        original = {p: p.read_bytes() for p in Path(cell["method_dir"]).rglob("*") if p.is_file()}
        for case in ("held", "live", "busy"):
            with self.subTest(case=case):
                backend = self.backend(runtime, cell, evidence, case)
                previous = self.root / (case + ".json")
                finite.write_json(previous, {"cells": [{
                    "cell": model.__dict__, "train_requested": True, "training_ownership": "producer",
                    "train": "failed", "eval": "blocked", "held": case == "held",
                    "reasons": ["preserved failure"], "attempts": [], "adopted": []}]})
                @contextmanager
                def producer():
                    backend.run_dir.mkdir()
                    yield
                live = [dict(cell=model, phase="train", pid=os.getpid(), gpus=(4,), spec=None)]
                lock = None
                if case == "busy":
                    lock = finite.take_lock(backend.lock_dir.parent / "cell_locks" /
                                            (model.session("train") + ".lock"))
                try:
                    with patch.object(backend, "producer", producer), \
                            patch.object(backend, "discover", return_value=live if case == "live" else []):
                        result = finite.run_finite(
                            {model: True}, backend, previous_state=previous,
                            retry_cells=(model.key + ":train",), aasfod_fns_evidence=evidence,
                            aasfod_fns_code=self.root / "not-even-inspected")
                    self.assertEqual(result["status"], "blocked")
                    self.assertIn({"held": "Retry cell held", "live": "no live canonical",
                                   "busy": "lock busy"}[case], result["reason"])
                    self.assertFalse((backend.run_dir / "recovery").exists())
                finally:
                    if lock:
                        lock.close()
                self.assertEqual({p: p.read_bytes() for p in original}, original)
        with self.assertRaisesRegex(finite.Blocked, "explicit TRAIN retry"):
            finite.run_finite({model: True}, backend, previous_state=previous,
                              aasfod_fns_evidence=evidence, aasfod_fns_code="/not-inspected")
        student = finite.Cell("RSAR", "clean", 42, "AASFOD", "student")
        with self.assertRaisesRegex(ValueError, "selected EMA"):
            finite.runner_command(backend.queue, student, "train", (4,), "/record.json")

    def test_failed_new_fns_attempt_is_retained_and_not_replayed(self):
        runtime, cell, evidence = self.fixture()
        self.model_checkout()
        backend = self.backend(runtime, cell, evidence)
        backend.run_dir.mkdir()
        model = finite.Cell("RSAR", "clean", 42, "AASFOD")
        backend.archive_retry(model, "train")
        with self.assertRaisesRegex(ValueError, "differs from the archived recovery record"):
            stages.load_fns_continuation(backend.fns_recovery, cell, self.root / "different-code")
        retained = {p: p.read_bytes() for p in Path(cell["work_dir"]).glob("alignment/*") if p.is_file()}

        def fail(command, **kwargs):
            if command[0] == "git":
                return NATIVE_RUN(command, **kwargs)
            fns = Path(cell["work_dir"]) / "fns"
            fns.mkdir()
            (fns / "failure.log").write_text("new native failure retained\n")
            return SimpleNamespace(returncode=17)

        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                patch.object(training.subprocess, "run", side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                training.train(backend.queue, 4, "RSAR", "clean", 42, "AASFOD", backend.fns_recovery)
            original = {p: p.read_bytes() for p in Path(cell["method_dir"]).rglob("*") if p.is_file()}
            with self.assertRaises(ValueError):
                training.train(backend.queue, 4, "RSAR", "clean", 42, "AASFOD", backend.fns_recovery)
        self.assertEqual({p: p.read_bytes() for p in original}, original)
        self.assertEqual({p: p.read_bytes() for p in retained}, retained)
        self.assertFalse(Path(cell["checkpoint"]).exists())
        self.assertEqual(Path(cell["terminal_status"]).read_text(), "tmux_wrap_exit=17\n")


if __name__ == "__main__":
    unittest.main()
