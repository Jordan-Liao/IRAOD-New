"""One CPU bootstrap/trajectory/epoch-checkpoint test; no research detector."""

import importlib.util
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from mmcv.runner import BaseRunner, EpochBasedRunner, Hook, OptimizerHook

from experiments.comparison import nonfinite_a39_epoch as entry


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "experiments/comparison/numerical_integrity.py"


class TinyFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.7, -.2]))
        self.ema = self.weight.detach().clone()
        self.losses = []

    def train_step(self, data, optimizer, **kwargs):
        loss = (self.weight - torch.rand_like(self.weight)).square().sum()
        self.losses.append(loss.item())
        return {"loss": loss, "log_vars": {}, "num_samples": 1}


class NaturalCheckpointFixture(Hook):
    def __init__(self):
        self.saved = {}
        self.after_run_seen = False

    def after_train_iter(self, runner):
        runner.model.ema.mul_(.999).add_(runner.model.weight.detach(), alpha=.001)

    def after_train_epoch(self, runner):
        for role, tensor in (("student", runner.model.weight), ("ema", runner.model.ema)):
            buffer = io.BytesIO()
            torch.save({"weight": tensor.detach().clone(), "iteration": runner.iter + 1}, buffer)
            buffer.seek(0)
            self.saved[role] = torch.load(buffer, weights_only=True)

    def after_run(self, runner):
        self.after_run_seen = True


class NaturalEpochTest(unittest.TestCase):
    def test_bootstrap_and_checked_natural_epoch_preserve_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            science = root / "a39"
            science.mkdir()
            runtime_file = science / "iraod_runtime.py"
            runtime_file.write_bytes(subprocess.check_output([
                "git", "-C", str(ROOT), "show", entry.SOURCE_SHA + ":iraod_runtime.py"]))
            spec = importlib.util.spec_from_file_location("iraod_runtime", runtime_file)
            runtime = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runtime)
            previous_cwd = Path.cwd()

            def exec_guard(executable, argv, env):
                self.assertEqual(argv[1:], [
                    str(Path(entry.__file__).resolve()), "--code-root", str(science),
                    "--finite-helper", str(HELPER), "--", "config.py", "--work-dir", "fresh"])
                self.assertEqual(Path.cwd(), science)
                self.assertEqual(env["PYTHONPATH"], str(science))
                self.assertEqual(env["IRAOD_RUNTIME_READY"], "1")
                raise RuntimeError("actual runtime reexec seam")

            try:
                with patch.dict(os.environ, {
                        "IRAOD_CONDA_PREFIX": str(Path(sys.executable).parent.parent),
                        "MPLCONFIGDIR": str(root / "mpl"), "XDG_CACHE_HOME": str(root / "cache")}), \
                        patch.dict(sys.modules, {"iraod_runtime": runtime}), \
                        patch.object(sys, "path", sys.path.copy()), \
                        patch.object(sys, "argv", sys.argv.copy()), \
                        patch.object(runtime.os, "execvpe", exec_guard):
                    os.environ.pop("IRAOD_RUNTIME_READY", None)
                    with self.assertRaisesRegex(RuntimeError, "actual runtime reexec"):
                        entry.bootstrap(science, HELPER, ["config.py", "--work-dir", "fresh"])
                    os.environ["IRAOD_RUNTIME_READY"] = "1"
                    entry.bootstrap(science, HELPER, ["config.py", "--work-dir", "fresh"])
            finally:
                os.chdir(previous_cwd)
            checks = entry.load_finite_helper(HELPER)
            self.assertEqual(Path(checks.__file__).resolve(), HELPER)
            self.assertEqual(checks.__name__, "_attempt6_finite_checks")
            original_hook = BaseRunner.call_hook
            original_save = torch.save
            anomaly_before = torch.is_anomaly_enabled()
            results = []
            for checked in (False, True):
                torch.manual_seed(44)
                model = TinyFixture()
                initial = model.weight.detach().clone()
                optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9, weight_decay=.0001)
                runner = EpochBasedRunner(
                    model, optimizer=optimizer, max_epochs=1,
                    work_dir=str(root / str(checked)), logger=logging.getLogger("cpu_epoch_fixture"))
                runner.register_hook(OptimizerHook())
                checkpoint = NaturalCheckpointFixture()
                runner.register_hook(checkpoint, priority="LOW")
                events = []
                if checked:
                    with checks.native_boundaries(evaluate=False), \
                            entry.observe_natural_epoch(BaseRunner, events.append) as state:
                        self.assertIsNot(torch.save, original_save)
                        runner.run([list(range(265))], [("train", 1)])
                    self.assertEqual(state, {
                        "completed_iterations": 265, "completed_epoch_hooks": 1, "after_run": True})
                    self.assertEqual(sum(e["event"] == "iteration_hooks_complete" for e in events), 265)
                    self.assertEqual(events[-1]["event"], "epoch_hooks_complete")
                else:
                    runner.run([list(range(265))], [("train", 1)])
                self.assertTrue(checkpoint.after_run_seen)
                self.assertFalse(torch.equal(model.weight, initial))
                self.assertGreater(model.losses[0], 0)
                self.assertEqual(set(checkpoint.saved), {"student", "ema"})
                self.assertTrue(all(saved["iteration"] == 266 for saved in checkpoint.saved.values()))
                results.append((model, optimizer, checkpoint, torch.get_rng_state().clone()))
            plain, checked = results
            self.assertTrue(torch.equal(plain[0].weight, checked[0].weight))
            self.assertTrue(torch.equal(plain[0].ema, checked[0].ema))
            self.assertEqual(plain[0].losses, checked[0].losses)
            self.assertEqual(plain[1].state_dict()["param_groups"], checked[1].state_dict()["param_groups"])
            self.assertTrue(torch.equal(
                plain[1].state_dict()["state"][0]["momentum_buffer"],
                checked[1].state_dict()["state"][0]["momentum_buffer"]))
            for role in ("student", "ema"):
                self.assertTrue(torch.equal(
                    plain[2].saved[role]["weight"], checked[2].saved[role]["weight"]))
            self.assertTrue(torch.equal(plain[3], checked[3]))
            self.assertEqual(torch.is_anomaly_enabled(), anomaly_before)
            self.assertIs(BaseRunner.call_hook, original_hook)
            self.assertIs(torch.save, original_save)


if __name__ == "__main__":
    unittest.main()
