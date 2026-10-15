"""CPU regressions through real MMCV boundaries; no native detector/GPU imports."""

from contextlib import nullcontext
import copy
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmcv.runner import OptimizerHook, load_checkpoint

from experiments.comparison import host_binding as host
from experiments.comparison import numerical_integrity as integrity
from experiments.comparison import smoke_frozen_training as smoke
from experiments.comparison import train_tam


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "integrity_semi_runner", ROOT / "mmdet_extension/core/runner/semi_runner.py")
semi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semi)


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, bad):
        ctx.bad = bad
        return value.square().sum()

    @staticmethod
    def backward(ctx, grad):
        return grad.new_full((1, 2), ctx.bad), None


def runner_fixture():
    model = torch.nn.Linear(2, 1, bias=False)
    model.register_buffer("running", torch.tensor([1.]))
    # Teacher is deliberately outside registered student parameters, as in native.
    teacher = copy.deepcopy(model)
    object.__setattr__(model, "ema_model", teacher)
    optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9)
    return SimpleNamespace(model=model, optimizer=optimizer, iter=0, epoch=0, meta=None)


class IntegrityTest(unittest.TestCase):
    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second):
                self.assert_tree_equal(left, right)
        else:
            self.assertEqual(first, second)

    def test_loss_fails_before_native_zero_grad_and_backward(self):
        for bad in (float("nan"), float("inf")):
            runner = runner_fixture()
            runner.model.weight.grad = torch.ones_like(runner.model.weight)
            runner.outputs = {"loss": runner.model.weight.sum() * bad}
            original = runner.model.weight.detach().clone()
            with integrity.native_boundaries():
                with self.assertRaisesRegex(FloatingPointError, "loss before backward"):
                    OptimizerHook().after_train_iter(runner)
            self.assertTrue(torch.equal(runner.model.weight.grad, torch.ones_like(original)))
            self.assertTrue(torch.equal(runner.model.weight, original))
            self.assertEqual(runner.optimizer.state, {})

    def test_finite_loss_bad_gradient_fails_before_real_sgd_and_adam_steps(self):
        for optimizer_type in (torch.optim.SGD, torch.optim.Adam):
            for bad in (float("nan"), float("inf")):
                runner = runner_fixture()
                runner.optimizer = optimizer_type(runner.model.parameters(), lr=.02)
                original = runner.model.weight.detach().clone()
                runner.outputs = {"loss": BadGradient.apply(runner.model.weight, bad)}
                self.assertTrue(torch.isfinite(runner.outputs["loss"]))
                with integrity.native_boundaries():
                    with self.assertRaisesRegex(FloatingPointError, "gradient before optimizer.step"):
                        OptimizerHook().after_train_iter(runner)
                self.assertTrue(torch.equal(runner.model.weight, original))
                self.assertEqual(runner.optimizer.state, {})

    def test_finite_two_update_smoke_is_bitwise_identical_including_rng_and_ema(self):
        def execute(guarded, directory, optimizer_type):
            torch.manual_seed(44)
            random.seed(44)
            np.random.seed(44)
            runner = runner_fixture()
            options = dict(momentum=.9, weight_decay=.0001) if optimizer_type is torch.optim.SGD else {}
            runner.optimizer = optimizer_type(runner.model.parameters(), lr=.02, **options)
            observer = smoke.SmokeEvidence(2, Path(directory) / "rank.json")
            # Match native: instrumentation installed before smoke wraps step.
            with integrity.native_boundaries() if guarded else nullcontext():
                observer.before_run(runner)
                for iteration in range(2):
                    runner.iter = iteration
                    with torch.no_grad():
                        runner.model.ema_model.weight.mul_(.998).add_(
                            runner.model.weight, alpha=.002)
                    loss = runner.model(torch.randn(1, 2)).square().sum()
                    runner.outputs = dict(loss=loss, log_vars={"loss": float(loss.detach())})
                    OptimizerHook().after_train_iter(runner)
                    if iteration == 1:
                        with self.assertRaises(smoke.SmokeFinished):
                            observer.after_train_iter(runner)
                    else:
                        observer.after_train_iter(runner)
            return (
                copy.deepcopy(runner.model.state_dict()),
                copy.deepcopy(runner.model.ema_model.state_dict()),
                copy.deepcopy(runner.optimizer.state_dict()), runner.model.weight.grad.clone(),
                torch.get_rng_state(), random.getstate(), np.random.get_state()[1].tolist(),
                json.loads(observer.output.read_text()))

        with tempfile.TemporaryDirectory() as directory:
            for optimizer_type in (torch.optim.SGD, torch.optim.Adam):
                baseline = execute(False, directory, optimizer_type)
                guarded = execute(True, directory, optimizer_type)
                self.assert_tree_equal(baseline, guarded)
                self.assertEqual(guarded[-1]["optimizer_updates"], 2)

    def test_native_student_ema_save_preserves_finite_payload_and_rejects_bad_payloads(self):
        for runner_type in (semi.SemiEpochBasedRunner, semi.SemiIterBasedRunner):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runner = runner_fixture()
                runner.outputs = {"loss": runner.model.weight.square().sum()}
                OptimizerHook().after_train_iter(runner)
                with patch("mmcv.runner.checkpoint.time.asctime", return_value="fixed time"):
                    runner_type.save_checkpoint(runner, str(root / "old"))
                    with integrity.native_boundaries():
                        runner_type.save_checkpoint(runner, str(root / "new"))
                for filename in ("iter_1.pth", "iter_1_ema.pth"):
                    self.assertEqual((root / "old" / filename).read_bytes(),
                                     (root / "new" / filename).read_bytes())
                self.assertEqual(os.readlink(root / "new/latest.pth"), "iter_1.pth")

                for location in ("student", "ema", "optimizer"):
                    bad = runner_fixture()
                    bad.outputs = {"loss": bad.model.weight.square().sum()}
                    OptimizerHook().after_train_iter(bad)
                    if location == "optimizer":
                        bad.optimizer.state[bad.model.weight]["momentum_buffer"].fill_(float("inf"))
                    else:
                        model = bad.model if location == "student" else bad.model.ema_model
                        model.running.fill_(float("nan"))
                    # Red-capable: the same real old runner publishes these tensors.
                    old = root / ("old_bad_" + location)
                    runner_type.save_checkpoint(bad, str(old))
                    self.assertTrue((old / "latest.pth").exists())
                    out = root / ("bad_" + location)
                    filename = "iter_1_ema.pth" if location == "ema" else "iter_1.pth"
                    with integrity.native_boundaries():
                        with self.assertRaisesRegex(FloatingPointError, "checkpoint before publication"):
                            runner_type.save_checkpoint(bad, str(out))
                    self.assertFalse((out / filename).exists())
                    self.assertFalse((out / "latest.pth").exists())

    def test_failed_save_does_not_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture()
            semi.SemiEpochBasedRunner.save_checkpoint(runner, directory)
            filename = Path(directory) / "iter_1.pth"
            original = filename.read_bytes()
            runner.model.running.fill_(float("inf"))
            with integrity.native_boundaries():
                with self.assertRaises(FloatingPointError):
                    semi.SemiEpochBasedRunner.save_checkpoint(runner, directory)
            self.assertEqual(filename.read_bytes(), original)

    def test_actual_mmcv_load_rejects_bad_parameters_and_buffers_before_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            for key in ("weight", "running"):
                for bad in (float("nan"), float("inf")):
                    runner = runner_fixture()
                    state = runner.model.state_dict()
                    state[key].fill_(bad)
                    filename = str(Path(directory) / "bad.pth")
                    torch.save({"state_dict": state}, filename)
                    # Old path loads successfully and can silently yield zero detections.
                    load_checkpoint(runner.model, filename, logger=logging.getLogger(__name__))
                    detections = runner.model(torch.ones(1, 2)).detach()
                    if key == "weight":
                        self.assertEqual(detections[torch.isfinite(detections)].numel(), 0)
                    with integrity.native_boundaries(evaluate=True):
                        with self.assertRaisesRegex(FloatingPointError, "loaded model before evaluation"):
                            load_checkpoint(runner.model, filename, logger=logging.getLogger(__name__))

    def test_finite_eval_load_and_context_cleanup_preserve_state(self):
        from mmcv.runner import checkpoint
        originals = (torch.save, OptimizerHook.after_train_iter, checkpoint.load_state_dict)
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_fixture()
            state = copy.deepcopy(runner.model.state_dict())
            filename = str(Path(directory) / "finite.pth")
            torch.save({"state_dict": state}, filename)
            rng = torch.get_rng_state()
            with integrity.native_boundaries(evaluate=True):
                loaded = load_checkpoint(runner.model, filename, logger=logging.getLogger(__name__))
            self.assert_tree_equal(loaded["state_dict"], state)
            self.assert_tree_equal(runner.model.state_dict(), state)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(originals,
                         (torch.save, OptimizerHook.after_train_iter, checkpoint.load_state_dict))
        # Removed step hook must not affect later uninstrumented execution.
        parameter = torch.nn.Parameter(torch.ones(1))
        parameter.grad = torch.full_like(parameter, float("nan"))
        torch.optim.SGD([parameter], lr=.02).step()
        self.assertTrue(torch.isnan(parameter).all())

    def test_current_tam_loop_guards_both_actual_optimizer_boundaries(self):
        class TinyTAM(torch.nn.Module):
            def __init__(self, bad_update):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 2))
                self.bad_update = bad_update
                self.completed = 0

            def make_optimizers(self):
                return (torch.optim.Adam([self.weight], lr=.01),
                        torch.optim.Adam([self.weight], lr=.01))

            def alternating_step(self, content, style, first, second, iteration):
                for index, optimizer in enumerate((first, second)):
                    optimizer.zero_grad()
                    loss = (BadGradient.apply(self.weight, float("nan"))
                            if index == self.bad_update else self.weight.square().sum())
                    loss.backward()
                    optimizer.step()
                    self.completed += 1
                return {"decoder": loss.detach(), "moments": loss.detach(), "lr": .01}

        for bad_update in (0, 1):
            module = TinyTAM(bad_update)
            with self.assertRaisesRegex(FloatingPointError, "gradient before optimizer.step"):
                train_tam.train_loop(module, [torch.ones(1)], [torch.ones(1)], 1, "cpu", io.StringIO())
            self.assertEqual(module.completed, bad_update)

    def test_cold_native_wrapper_reaches_original_source_boundaries(self):
        # Native detector libraries are absent locally. Use actual MMCV + the real
        # runner source in an isolated frozen-shaped checkout, never GPU/model data.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sfod").mkdir()
            (root / "sfod/__init__.py").write_text("SOURCE = 'unchanged-frozen-model'\n")
            common = (
                "import torch, sfod\n"
                "assert sfod.SOURCE == 'unchanged-frozen-model'\n"
                "from mmcv.runner import OptimizerHook, load_checkpoint\n"
                "from types import SimpleNamespace\n"
                "model=torch.nn.Linear(2,1,bias=False)\n"
                "optimizer=torch.optim.SGD(model.parameters(),lr=.02)\n")
            cases = {
                "loss": "runner=SimpleNamespace(model=model,optimizer=optimizer,iter=0,"
                        "outputs={'loss':model.weight.sum()*float('nan')})\n"
                        "OptimizerHook().after_train_iter(runner)\n",
                "save": "torch.save({'state_dict':{'weight':torch.tensor(float('nan'))}},"
                        f"{str(root / 'bad-save.pth')!r})\n",
                "eval": f"load_checkpoint(model,{str(root / 'bad-load.pth')!r})\n"
                        "raise AssertionError('evaluation must not succeed')\n",
            }
            torch.save({"state_dict": {"weight": torch.full((1, 2), float("nan"))}},
                       root / "bad-load.pth")
            for mode, body in cases.items():
                entry = root / ("test.py" if mode == "eval" else "train.py")
                entry.write_text(common + body)
                before = entry.read_bytes()
                bootstrap = (
                    "import runpy,socket,sys; "
                    f"socket.gethostname=lambda:{host.TARGET_HOST!r}; "
                    f"sys.argv={[host.__file__, 'native', str(entry)]!r}; "
                    f"runpy.run_path({host.__file__!r},run_name='__main__')")
                result = subprocess.run(
                    [sys.executable, "-c", bootstrap], cwd=root,
                    env={**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "",
                         "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                         "PYTHONDONTWRITEBYTECODE": "1"},
                    text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FloatingPointError: Nonfinite", result.stderr)
                self.assertEqual(entry.read_bytes(), before)
            self.assertFalse((root / "bad-save.pth").exists())

    def test_cold_aasfod_helper_uses_current_shim_with_bound_pythonpath(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "experiments/comparison"
            package.mkdir(parents=True)
            # Match the native checkout's experiments namespace packages.
            (package / "host_binding.py").write_text(
                "raise AssertionError('must not use old bound orchestration')\n")
            entry = ROOT / "experiments/comparison/train_aasfod.py"
            bootstrap = (
                "import runpy,socket; "
                f"socket.gethostname=lambda:{host.TARGET_HOST!r}; "
                f"scope=runpy.run_path({str(entry)!r}); "
                "host=scope['host']; "
                f"assert host.__file__ == {host.__file__!r}; "
                "command=host.native_command(['python','/bound/train.py'],'/bound/train.py'); "
                f"assert command == ['python',{host.__file__!r},'native','/bound/train.py']")
            result = subprocess.run(
                [sys.executable, "-c", bootstrap], cwd=root,
                env={**os.environ, "PYTHONPATH": str(root), "CUDA_VISIBLE_DEVICES": "",
                     "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "PYTHONDONTWRITEBYTECODE": "1"},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
