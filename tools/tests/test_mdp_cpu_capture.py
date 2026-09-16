"""CPU-checked observation preserves updates and keeps the original finite gates."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from mmcv.runner import OptimizerHook
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/comparison'))
from mdp_cpu_capture import CpuFirstInvalidCapture, DiagnosticLimitReached
import numerical_integrity as finite
from test_mdp_first_invalid import ForwardFailure, GradientFailure, ToyStep, UpdateFailure


def step(model, optimizer, batch):
    runner = SimpleNamespace(model=model, optimizer=optimizer)
    runner.outputs = model.train_step(batch, optimizer)
    OptimizerHook().after_train_iter(runner)


class CpuCaptureTest(unittest.TestCase):
    def test_cpu_checks_preserve_updates_without_torch_finite_or_max_reductions(self):
        torch.manual_seed(42)
        plain = ToyStep()
        optimizer = torch.optim.SGD(plain.parameters(), lr=.1, momentum=.9)
        batch = {'x': torch.tensor([1., 2.])}
        step(plain, optimizer, batch)
        step(plain, optimizer, batch)
        expected = deepcopy(plain.state_dict())
        expected_rng = torch.get_rng_state()
        torch.manual_seed(42)
        observed = ToyStep()
        observed_optimizer = torch.optim.SGD(observed.parameters(), lr=.1, momentum=.9)
        with tempfile.TemporaryDirectory() as directory:
            capture = CpuFirstInvalidCapture(Path(directory) / 'capture', finite, 2)
            with patch.object(torch, 'isfinite', side_effect=AssertionError('Torch finite check')), \
                    patch.object(torch, 'stack', side_effect=AssertionError('Torch maximum reduction')):
                with capture.install(ToyStep, OptimizerHook):
                    step(observed, observed_optimizer, batch)
                    step(observed, observed_optimizer, batch)
                    with self.assertRaises(DiagnosticLimitReached):
                        step(observed, observed_optimizer, batch)
            for key, value in expected.items():
                torch.testing.assert_close(observed.state_dict()[key], value, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
            torch.testing.assert_close(
                observed_optimizer.state[observed.weight]['momentum_buffer'],
                optimizer.state[plain.weight]['momentum_buffer'], rtol=0, atol=0)
            self.assertEqual(capture.completed, 2)
            self.assertIsNone(capture.anomaly)
            self.assertTrue((capture.output / 'initial.pt').is_file())
            self.assertTrue((capture.output / 'capture.pt').is_file())
            self.assertGreater(capture.logs['gradient_max_abs'], 0)

    def test_invalid_gradient_still_stops_before_optimizer_update(self):
        model = GradientFailure()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            capture = CpuFirstInvalidCapture(Path(directory) / 'capture', finite, 2)
            with capture.install(GradientFailure, OptimizerHook):
                with self.assertRaises(FloatingPointError):
                    step(model, optimizer, {'x': torch.tensor([2.])})
            torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
            summary = json.loads((capture.output / 'summary.json').read_text())
            self.assertEqual(summary['stage'], 'before_optimizer')
            self.assertEqual(capture.completed, 0)

    def test_invalid_loss_still_stops_before_backward(self):
        model = ForwardFailure()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        before = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            capture = CpuFirstInvalidCapture(Path(directory) / 'capture', finite, 2)
            with capture.install(ForwardFailure, OptimizerHook):
                with self.assertRaises(FloatingPointError):
                    step(model, optimizer, {'x': torch.tensor([2000.])})
            torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
            payload = torch.load(capture.output / 'capture.pt', weights_only=False)
            self.assertEqual(payload['stage'], 'forward_or_loss')
            self.assertFalse(payload['backward_started'])
            self.assertEqual(payload['gradients'], {})

    def test_finite_gradient_overflowing_optimizer_is_still_captured(self):
        model = UpdateFailure()
        with torch.no_grad():
            model.weight.fill_(3e38)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        with tempfile.TemporaryDirectory() as directory:
            capture = CpuFirstInvalidCapture(Path(directory) / 'capture', finite, 2)
            with capture.install(UpdateFailure, OptimizerHook):
                with self.assertRaises(FloatingPointError):
                    step(model, optimizer, {
                        'x': torch.tensor([-3e38]), 'baseline': torch.tensor([3e38])})
            summary = json.loads((capture.output / 'summary.json').read_text())
            self.assertEqual(summary['stage'], 'after_optimizer')
            self.assertEqual(capture.completed, 0)


if __name__ == '__main__':
    unittest.main()
