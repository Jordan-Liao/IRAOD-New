"""CPU checks for an observer, not a reproduction of the real MDP failure."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from mmcv.runner import OptimizerHook

ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load('mdp_capture_cpu', 'experiments/comparison/mdp_first_invalid.py')
finite = load('finite_capture_cpu', 'experiments/comparison/numerical_integrity.py')


class ToyStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.5]))
        self.register_buffer('running', torch.zeros(1))

    def train_step(self, batch, optimizer):
        self.running.add_(1)
        value = self.weight * (batch['x'] + torch.rand_like(batch['x']) * .1)
        loss = value.square().mean()
        return dict(loss=loss, log_vars={'loss': loss.item()}, num_samples=len(batch['x']))


class BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.square()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float('nan'))


class GradientFailure(ToyStep):
    def train_step(self, batch, optimizer):
        self.running.add_(1)
        loss = BadBackward.apply(self.weight * batch['x']).sum()
        return dict(loss=loss, log_vars={'loss': loss.item()}, num_samples=len(batch['x']))


class Exponential(nn.Module):
    def forward(self, value):
        return value.exp()


class ForwardFailure(ToyStep):
    def __init__(self):
        super().__init__()
        self.mdp = nn.Module()
        self.mdp.afsp = Exponential()

    def train_step(self, batch, optimizer):
        self.running.add_(1)
        loss = self.mdp.afsp(self.weight * batch['x']).sum()
        return dict(loss=loss, log_vars={'loss': loss.item()}, num_samples=len(batch['x']))


class UpdateFailure(ToyStep):
    def train_step(self, batch, optimizer):
        loss = ((self.weight - batch['baseline']) * batch['x']).sum()
        return dict(loss=loss, log_vars={'loss': loss.item()}, num_samples=len(batch['x']))


class FirstInvalidTest(unittest.TestCase):
    def setUp(self):
        state = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, state)

    def step(self, model, optimizer, batch):
        runner = SimpleNamespace(model=model, optimizer=optimizer)
        runner.outputs = model.train_step(batch, optimizer)
        OptimizerHook().after_train_iter(runner)

    def test_cli_keeps_training_tail_and_explicit_diagnostic_limit(self):
        argv = ['observer.py', '--code-root', str(ROOT),
                '--finite-helper', 'finite.py', '--output', 'capture',
                '--max-updates', '265', '--', 'config.py', '--seed', '42']
        with patch.object(diagnostic.sys, 'argv', argv):
            options = diagnostic.arguments()
        self.assertEqual(options.max_updates, 265)
        self.assertEqual(options.train_arguments, ['--', 'config.py', '--seed', '42'])

    def test_finite_updates_rng_and_optimizer_are_unchanged(self):
        torch.manual_seed(93)
        plain = ToyStep()
        optimizer = torch.optim.SGD(plain.parameters(), lr=.1, momentum=.9)
        batch = {'x': torch.tensor([1., 2.]), 'meta': [{'name': 'synthetic'}]}
        self.step(plain, optimizer, batch)
        self.step(plain, optimizer, batch)
        rng = torch.get_rng_state()
        torch.manual_seed(93)
        observed = ToyStep()
        observed_optimizer = torch.optim.SGD(observed.parameters(), lr=.1, momentum=.9)
        with tempfile.TemporaryDirectory() as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, max_updates=2, anomaly_from=1)
            with capture.install(ToyStep, OptimizerHook):
                self.step(observed, observed_optimizer, batch)
                with self.assertRaises(diagnostic.DiagnosticLimitReached):
                    self.step(observed, observed_optimizer, batch)
            torch.testing.assert_close(observed.weight, plain.weight, rtol=0, atol=0)
            torch.testing.assert_close(observed.running, plain.running, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            torch.testing.assert_close(
                observed_optimizer.state[observed.weight]['momentum_buffer'],
                optimizer.state[plain.weight]['momentum_buffer'], rtol=0, atol=0)
            summary = json.loads((capture.output / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'DIAGNOSTIC_LIMIT_WITHOUT_NONFINITE')
            self.assertFalse(summary['valid_full_budget_model'])

    def test_finite_limit_retains_initial_and_last_actual_step_for_replay(self):
        torch.manual_seed(42)
        model = ToyStep()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9)
        batches = [{'x': torch.tensor([1., 2.])}, {'x': torch.tensor([3., 4.])}]
        expected = []
        with tempfile.TemporaryDirectory() as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, max_updates=2, anomaly_from=1)
            with capture.install(ToyStep, OptimizerHook):
                for index, batch in enumerate(batches):
                    if index == 1:
                        with self.assertRaises(diagnostic.DiagnosticLimitReached):
                            self.step(model, optimizer, batch)
                    else:
                        self.step(model, optimizer, batch)
                    expected.append({
                        'model': diagnostic.copy_tree(model.state_dict(), to_cpu=True),
                        'optimizer': diagnostic.copy_tree(optimizer.state_dict(), to_cpu=True),
                        'rng': torch.get_rng_state().clone(),
                    })
            for index, name in ((1, 'capture.pt'), (0, 'initial.pt')):
                payload = torch.load(capture.output / name, weights_only=False)
                self.assertEqual(payload['update'], index + 1)
                self.assertEqual(payload['stage'], 'initial' if index == 0 else 'finite_limit')
                torch.testing.assert_close(payload['actual_batch']['x'], batches[index]['x'])
                replay = ToyStep()
                replay_optimizer = torch.optim.SGD(replay.parameters(), lr=.1, momentum=.9)
                replay.load_state_dict(payload['pre_forward']['model'])
                replay_optimizer.load_state_dict(payload['pre_forward']['optimizer'])
                torch.set_rng_state(payload['pre_forward']['rng']['torch'])
                self.step(replay, replay_optimizer, payload['actual_batch'])
                for key, value in expected[index]['model'].items():
                    torch.testing.assert_close(replay.state_dict()[key], value, rtol=0, atol=0)
                torch.testing.assert_close(
                    replay_optimizer.state[replay.weight]['momentum_buffer'],
                    expected[index]['optimizer']['state'][0]['momentum_buffer'], rtol=0, atol=0)
                torch.testing.assert_close(torch.get_rng_state(), expected[index]['rng'])
            summary = json.loads((capture.output / 'summary.json').read_text())
            self.assertEqual(summary['capture'], str(capture.output / 'capture.pt'))
            self.assertFalse(summary['valid_full_budget_model'])
            self.assertFalse(capture.failure_saved)

    def test_bad_gradient_stops_before_update_and_preserves_actual_prestate(self):
        model = GradientFailure()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        batch = {'x': torch.tensor([2.]), 'meta': [{'id': 'finite-input'}]}
        original = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, anomaly_from=99)
            with capture.install(GradientFailure, OptimizerHook):
                with self.assertRaises(FloatingPointError):
                    self.step(model, optimizer, batch)
            payload = torch.load(capture.output / 'capture.pt', weights_only=False)
            torch.testing.assert_close(model.weight, original)
            torch.testing.assert_close(payload['pre_forward']['model']['weight'], original)
            torch.testing.assert_close(payload['pre_forward']['model']['running'], torch.zeros(1))
            torch.testing.assert_close(payload['actual_batch']['x'], batch['x'])
            self.assertTrue(torch.isnan(payload['gradients']['weight']).all())
            self.assertEqual(capture.completed, 0)
            self.assertTrue(capture.failure_saved)

    def test_finite_gradient_overflowing_update_retains_pre_update_state(self):
        model = UpdateFailure()
        with torch.no_grad():
            model.weight.fill_(3e38)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.)
        batch = {'x': torch.tensor([-3e38]), 'baseline': torch.tensor([3e38])}
        with tempfile.TemporaryDirectory() as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, anomaly_from=99)
            with capture.install(UpdateFailure, OptimizerHook):
                with self.assertRaises(FloatingPointError):
                    self.step(model, optimizer, batch)
            payload = torch.load(capture.output / 'capture.pt', weights_only=False)
            self.assertEqual(payload['stage'], 'after_optimizer')
            self.assertTrue(torch.isfinite(payload['pre_forward']['model']['weight']).all())
            self.assertTrue(torch.isfinite(payload['gradients']['weight']).all())
            self.assertTrue(torch.isinf(model.weight).all())
            self.assertEqual(capture.completed, 0)

    def test_anomaly_backward_keeps_its_error_and_capture(self):
        model = GradientFailure()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with tempfile.TemporaryDirectory() as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, anomaly_from=1)
            with capture.install(GradientFailure, OptimizerHook):
                with self.assertRaisesRegex(RuntimeError, 'returned nan values'):
                    self.step(model, optimizer, {'x': torch.tensor([2.])})
            summary = json.loads((capture.output / 'summary.json').read_text())
            self.assertEqual(summary['status'], 'CAPTURED_NONFINITE')
            self.assertEqual(summary['stage'], 'backward_or_update')
            self.assertEqual(capture.completed, 0)

    def test_first_bad_forward_saves_operands_without_running_optimizer(self):
        model = ForwardFailure()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with tempfile.TemporaryDirectory() as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, anomaly_from=1)
            with capture.install(ForwardFailure, OptimizerHook):
                with self.assertRaises(FloatingPointError):
                    self.step(model, optimizer, {'x': torch.tensor([2000.])})
            payload = torch.load(capture.output / 'capture.pt', weights_only=False)
            self.assertEqual(payload['operands']['module'], 'mdp.afsp')
            self.assertTrue(torch.isinf(payload['operands']['output']).all())
            self.assertEqual(payload['gradients'], {})
            self.assertEqual(capture.completed, 0)
            self.assertEqual(len(model.mdp.afsp._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
