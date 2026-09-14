"""CPU preservation/red-stop control using the supplied actual native parser source."""

import argparse
import ast
from collections import OrderedDict
import io
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from mmcv.parallel import DataContainer
from mmcv.runner import BaseRunner, EpochBasedRunner, Hook, OptimizerHook

from experiments.comparison import nonfinite_a39_epoch as epoch
from experiments.comparison import nonfinite_named_losses as named


ROOT = Path(__file__).resolve().parents[2]
NATIVE = {}


def direct_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": []}


class LossFixture(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.7, -.2]))
        self.ema = self.weight.detach().clone()
        self.mode = mode
        self.forward_calls = self.parse_calls = 0
        self.loss_values = []

    def forward(self, img, img_metas, ordinal):
        self.last_pre_forward_rng = direct_rng()
        self.forward_calls += 1
        random_value = random.random() + float(np.random.random()) + torch.rand(())
        losses = OrderedDict([
            ("loss_main", (self.weight - img.data - .01 * random_value).square()),
            ("accuracy", self.weight.new_tensor(float("nan"))),
            ("teacher_aux_loss", [self.weight.square().mean() * .1]),
        ])
        if ordinal == 2:
            if self.mode == "named":
                losses["teacher_aux_loss"] = [self.weight.new_tensor(-1.).sqrt()]
            elif self.mode == "aggregate":
                huge = self.weight.new_tensor(torch.finfo(torch.float32).max * .75)
                losses["loss_main"] = huge
                losses["teacher_aux_loss"] = [huge]
            elif self.mode == "empty_reduction":
                losses["loss_main"] = self.weight[:0]
        return losses

    def _parse_losses(self, losses):
        self.parse_calls += 1
        return NATIVE["_parse_losses"](self, losses)

    def train_step(self, data, optimizer, **kwargs):
        output = NATIVE["train_step"](self, data, optimizer)
        self.loss_values.append(output["loss"].item())
        return output


class EpochFixture(Hook):
    def __init__(self):
        self.updates = 0
        self.saved = {}
        self.after_run_seen = False
        self.prefix = []

    def after_train_iter(self, runner):
        self.updates += 1
        runner.model.ema.mul_(.999).add_(runner.model.weight.detach(), alpha=.001)
        if self.updates <= 2:
            self.prefix.append((
                runner.model.weight.detach().clone(), runner.model.ema.clone(),
                runner.optimizer.state_dict()["state"][0]["momentum_buffer"].clone()))

    def after_train_epoch(self, runner):
        for role, tensor in (("student", runner.model.weight), ("ema", runner.model.ema)):
            stream = io.BytesIO()
            torch.save({"weight": tensor.detach().clone(), "iteration": runner.iter + 1}, stream)
            stream.seek(0)
            self.saved[role] = torch.load(stream, weights_only=True)

    def after_run(self, runner):
        self.after_run_seen = True


class NamedLossTest(unittest.TestCase):
    def test_native_preservation_and_failure_only_red_stops(self):
        helper = ROOT / "experiments/comparison/numerical_integrity.py"
        checks = epoch.load_finite_helper(helper)
        original_save = torch.save
        original_hook = BaseRunner.call_hook
        anomaly = torch.is_anomaly_enabled()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            science = root / "a39"
            science.mkdir()
            runtime_path = science / "iraod_runtime.py"
            runtime_path.write_bytes(subprocess.check_output([
                "git", "-C", str(ROOT), "show", epoch.SOURCE_SHA + ":iraod_runtime.py"]))
            spec = named.importlib.util.spec_from_file_location("iraod_runtime", runtime_path)
            runtime = named.importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runtime)
            old_cwd = Path.cwd()

            def stop_exec(executable, argv, env):
                self.assertEqual(argv[1:], [
                    str(Path(named.__file__).resolve()), "--code-root", str(science),
                    "--finite-helper", str(helper), "--epoch-adapter", str(Path(epoch.__file__).resolve()),
                    "--failure-dir", str(root / "bootstrap_failure"), "--", "config.py"])
                self.assertEqual(env["PYTHONPATH"], str(science))
                self.assertEqual(env["IRAOD_RUNTIME_READY"], "1")
                raise RuntimeError("named entry survives actual bootstrap")

            try:
                with patch.dict(os.environ, {
                        "IRAOD_CONDA_PREFIX": str(Path(sys.executable).parent.parent),
                        "MPLCONFIGDIR": str(root / "mpl"), "XDG_CACHE_HOME": str(root / "cache")}), \
                        patch.dict(sys.modules, {"iraod_runtime": runtime}), \
                        patch.object(sys, "path", sys.path.copy()), \
                        patch.object(sys, "argv", sys.argv.copy()), \
                        patch.object(runtime.os, "execvpe", stop_exec):
                    os.environ.pop("IRAOD_RUNTIME_READY", None)
                    with self.assertRaisesRegex(RuntimeError, "survives actual bootstrap"):
                        named.bootstrap(science, helper, epoch.__file__, root / "bootstrap_failure",
                                        ["config.py"])
                    os.environ["IRAOD_RUNTIME_READY"] = "1"
                    named.bootstrap(science, helper, epoch.__file__, root / "bootstrap_failure",
                                    ["config.py"])
            finally:
                os.chdir(old_cwd)

            baseline = None
            for mode in ("baseline", "finite", "named", "aggregate", "empty_reduction"):
                with self.subTest(mode=mode):
                    random.seed(44)
                    np.random.seed(44)
                    torch.manual_seed(44)
                    model = LossFixture(mode)
                    initial = model.weight.detach().clone()
                    optimizer = torch.optim.SGD(model.parameters(), lr=.02, momentum=.9, weight_decay=.0001)
                    runner = EpochBasedRunner(
                        model, optimizer=optimizer, max_epochs=1, work_dir=str(root / mode),
                        logger=logging.getLogger("named_loss_cpu"))
                    runner.register_hook(OptimizerHook())
                    observer = EpochFixture()
                    runner.register_hook(observer, priority="LOW")
                    batches = [{"img": DataContainer(torch.tensor([.1, .3])),
                                "img_metas": [{"ori_filename": f"fixture_{i}.png", "shape": (1, 2)}],
                                "ordinal": i} for i in range(265)]
                    failure_dir = root / (mode + "_failure")
                    saves = []

                    def save_batch(payload, path):
                        self.assertIsNot(torch.save, original_save)
                        with self.assertRaises(FloatingPointError):
                            torch.save({"state_dict": {"bad": torch.tensor(float("nan"))}}, io.BytesIO())
                        saves.append(path)
                        original_save(payload, path)

                    probe = named.NamedLossProbe(failure_dir, checks.require_finite, save_batch)
                    with patch("mmcv.runner.epoch_based_runner.time.sleep", return_value=None):
                        if mode == "baseline":
                            with checks.native_boundaries(evaluate=False):
                                runner.run([batches], [("train", 1)])
                        elif mode == "finite":
                            with checks.native_boundaries(evaluate=False), \
                                    epoch.observe_natural_epoch(BaseRunner, lambda _: None) as state, \
                                    probe.install(BaseRunner):
                                runner.run([batches], [("train", 1)])
                            self.assertEqual(state, {
                                "completed_iterations": 265, "completed_epoch_hooks": 1, "after_run": True})
                        else:
                            with checks.native_boundaries(evaluate=False), \
                                    epoch.observe_natural_epoch(BaseRunner, lambda _: None), \
                                    probe.install(BaseRunner), self.assertRaises(FloatingPointError):
                                runner.run([batches], [("train", 1)])
                    self.assertIs(BaseRunner.call_hook, original_hook)
                    self.assertIs(torch.save, original_save)
                    self.assertNotIn("_parse_losses", model.__dict__)
                    self.assertEqual(torch.is_anomaly_enabled(), anomaly)
                    self.assertFalse(torch.cuda.is_initialized())
                    if mode == "baseline":
                        baseline = (model, optimizer, observer, direct_rng())
                    elif mode == "finite":
                        self.assertFalse(failure_dir.exists())
                        self.assertEqual(saves, [])
                        self.assertEqual(model.parse_calls, 265)
                        self.assertFalse(torch.equal(model.weight, initial))
                        self.assertGreater(model.loss_values[0], 0)
                        self.assertTrue(torch.equal(model.weight, baseline[0].weight))
                        self.assertTrue(torch.equal(model.ema, baseline[0].ema))
                        self.assertEqual(model.loss_values, baseline[0].loss_values)
                        self.assertEqual(optimizer.state_dict()["param_groups"],
                                         baseline[1].state_dict()["param_groups"])
                        self.assertTrue(torch.equal(
                            optimizer.state_dict()["state"][0]["momentum_buffer"],
                            baseline[1].state_dict()["state"][0]["momentum_buffer"]))
                        self.assertTrue(observer.after_run_seen)
                        for role in ("student", "ema"):
                            self.assertEqual(observer.saved[role]["iteration"], 266)
                            self.assertTrue(torch.equal(
                                observer.saved[role]["weight"], baseline[2].saved[role]["weight"]))
                        current_rng = direct_rng()
                        self.assertEqual(current_rng["python"], baseline[3]["python"])
                        np.testing.assert_equal(current_rng["numpy"], baseline[3]["numpy"])
                        self.assertTrue(torch.equal(current_rng["torch_cpu"], baseline[3]["torch_cpu"]))
                    else:
                        self.assertEqual(observer.updates, 2)
                        self.assertEqual(model.forward_calls, 3)
                        self.assertEqual(model.parse_calls, 2 if mode == "named" else 3)
                        self.assertFalse(observer.after_run_seen)
                        self.assertEqual(observer.saved, {})
                        self.assertTrue(torch.equal(model.weight, baseline[2].prefix[1][0]))
                        self.assertTrue(torch.equal(model.ema, baseline[2].prefix[1][1]))
                        self.assertTrue(torch.equal(
                            optimizer.state_dict()["state"][0]["momentum_buffer"], baseline[2].prefix[1][2]))
                        self.assertEqual(len(saves), 1)
                        metadata = json.loads((failure_dir / "failure.json").read_text())
                        self.assertEqual(metadata["stage"],
                                         "named_loss_before_aggregate" if mode == "named" else "aggregate_only")
                        self.assertEqual(metadata["runner_iter_zero_based"], 2)
                        self.assertEqual(metadata["update_one_based"], 3)
                        self.assertEqual(metadata["ordered_loss_names"], ["loss_main", "teacher_aux_loss"])
                        self.assertEqual(metadata["capture_status"], "complete")
                        self.assertTrue(metadata["not_a_model_or_resume_checkpoint"])
                        self.assertEqual(set(p.name for p in failure_dir.iterdir()),
                                         {"failure.json", "failure_batch_rng.pt"})
                        payload = torch.load(failure_dir / "failure_batch_rng.pt",
                                             map_location="cpu", weights_only=False)
                        self.assertEqual(set(payload), {
                            "schema", "diagnostic_only", "not_a_model_or_resume_checkpoint",
                            "batch", "pre_forward_rng"})
                        self.assertEqual(payload["batch"]["ordinal"], 2)
                        self.assertTrue(torch.equal(payload["batch"]["img"].data, batches[2]["img"].data))
                        self.assertEqual(payload["batch"]["img_metas"], batches[2]["img_metas"])
                        self.assertEqual(payload["pre_forward_rng"]["python"],
                                         model.last_pre_forward_rng["python"])
                        np.testing.assert_equal(payload["pre_forward_rng"]["numpy"],
                                                model.last_pre_forward_rng["numpy"])
                        self.assertTrue(torch.equal(payload["pre_forward_rng"]["torch_cpu"],
                                                    model.last_pre_forward_rng["torch_cpu"]))
                        self.assertEqual(payload["pre_forward_rng"]["torch_cuda"], [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-parser-source", required=True)
    args = parser.parse_args()
    source = Path(args.native_parser_source).resolve()
    tree = ast.parse(source.read_text())
    detector = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseDetector")
    methods = [node for node in detector.body
               if isinstance(node, ast.FunctionDef) and node.name in ("_parse_losses", "train_step")]
    namespace = {"torch": torch, "dist": torch.distributed, "OrderedDict": OrderedDict}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(source), "exec"), namespace)
    NATIVE.update({name: namespace[name] for name in ("_parse_losses", "train_step")})
    unittest.main(argv=[sys.argv[0]], verbosity=2)
