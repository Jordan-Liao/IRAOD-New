"""Observer passthrough on the actual MDP train_step and native CPU loss stack."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F
from mmcv.runner import OptimizerHook

from sfod.extensions.mdp import MDPOBB
import test_mdp_integration as mdp_fixture
from test_mdp_first_invalid import diagnostic, finite

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/comparison'))
from mdp_cpu_capture import CpuFirstInvalidCapture


class MDPCaptureNativeTest(unittest.TestCase):
    def test_cpu_checked_observer_preserves_actual_native_losses_and_update(self):
        fixture = mdp_fixture.MDPIntegrationTest()
        fixture.setUp()
        baseline, _ = fixture.initialized_model()
        observed, _ = fixture.initialized_model()
        observed.ema_model.load_state_dict(baseline.ema_model.state_dict())
        observed.load_state_dict(baseline.state_dict())
        images = torch.randn(2, 3, 64, 64)
        data = {
            'img': F.interpolate(images, (128, 128), mode='bilinear', align_corners=False),
            'img_metas': [mdp_fixture.meta(f'{i}.png', 128, 2.) for i in range(2)],
            'gt_bboxes': [torch.empty(0, 5) for _ in range(2)],
            'gt_labels': [torch.empty(0, dtype=torch.long) for _ in range(2)],
            'img_unlabeled_1': images,
            'img_metas_unlabeled_1': [mdp_fixture.meta(f'{i}.png', 64) for i in range(2)],
            'gt_bboxes_unlabeled_1': [torch.empty(0, 5) for _ in range(2)],
            'gt_labels_unlabeled_1': [torch.empty(0, dtype=torch.long) for _ in range(2)],
        }
        rng = diagnostic.rng_state()
        optimizer = torch.optim.SGD(baseline.parameters(), lr=.001, momentum=.9)
        runner = SimpleNamespace(model=baseline, optimizer=optimizer)
        runner.outputs = baseline.train_step(deepcopy(data), optimizer)
        OptimizerHook().after_train_iter(runner)
        expected_rng = diagnostic.rng_state()
        torch.set_rng_state(rng['torch'])
        np.random.set_state(rng['numpy'])
        diagnostic.random.setstate(rng['python'])
        observed_optimizer = torch.optim.SGD(observed.parameters(), lr=.001, momentum=.9)
        observed_runner = SimpleNamespace(model=observed, optimizer=observed_optimizer)
        with tempfile.TemporaryDirectory() as directory:
            capture = CpuFirstInvalidCapture(Path(directory) / 'capture', finite, 1)
            with capture.install(MDPOBB, OptimizerHook):
                observed_runner.outputs = observed.train_step(deepcopy(data), observed_optimizer)
                OptimizerHook().after_train_iter(observed_runner)
            self.assertEqual(observed_runner.outputs['log_vars'], runner.outputs['log_vars'])
            self.assertGreater(capture.logs['loss'], 0)
            self.assertEqual(capture.completed, 1)
            self.assertIsNone(capture.anomaly)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(observed.state_dict()[name], value, rtol=0, atol=0)
            baseline_parameters = dict(baseline.named_parameters())
            for name, parameter in observed.named_parameters():
                if parameter.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad, baseline_parameters[name].grad, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng['torch'], rtol=0, atol=0)
        self.assertFalse(torch.cuda.is_initialized())

    def test_actual_train_step_has_identical_update_and_restores_hooks(self):
        fixture = mdp_fixture.MDPIntegrationTest()
        fixture.setUp()
        baseline, _ = fixture.initialized_model()
        observed, _ = fixture.initialized_model()
        observed.ema_model.load_state_dict(baseline.ema_model.state_dict())
        observed.load_state_dict(baseline.state_dict())
        images = torch.randn(2, 3, 64, 64)
        data = {
            'img': F.interpolate(images, (128, 128), mode='bilinear', align_corners=False),
            'img_metas': [mdp_fixture.meta(f'{i}.png', 128, 2.) for i in range(2)],
            'gt_bboxes': [torch.empty(0, 5) for _ in range(2)],
            'gt_labels': [torch.empty(0, dtype=torch.long) for _ in range(2)],
            'img_unlabeled_1': images,
            'img_metas_unlabeled_1': [mdp_fixture.meta(f'{i}.png', 64) for i in range(2)],
            'gt_bboxes_unlabeled_1': [torch.empty(0, 5) for _ in range(2)],
            'gt_labels_unlabeled_1': [torch.empty(0, dtype=torch.long) for _ in range(2)],
        }
        state = diagnostic.rng_state()
        optimizer = torch.optim.SGD(baseline.parameters(), lr=.001, momentum=.9)
        runner = SimpleNamespace(model=baseline, optimizer=optimizer)
        runner.outputs = baseline.train_step(deepcopy(data), optimizer)
        OptimizerHook().after_train_iter(runner)
        expected = deepcopy(baseline.state_dict())
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['numpy'])
        diagnostic.random.setstate(state['python'])
        observed_optimizer = torch.optim.SGD(observed.parameters(), lr=.001, momentum=.9)
        observed_runner = SimpleNamespace(model=observed, optimizer=observed_optimizer)
        inherited = 'train_step' not in MDPOBB.__dict__
        with tempfile.TemporaryDirectory(prefix='mdp-observer-native-') as directory:
            capture = diagnostic.FirstInvalidCapture(
                Path(directory) / 'capture', finite, max_updates=1, anomaly_from=1)
            with capture.install(MDPOBB, OptimizerHook):
                observed_runner.outputs = observed.train_step(deepcopy(data), observed_optimizer)
                with self.assertRaises(diagnostic.DiagnosticLimitReached):
                    OptimizerHook().after_train_iter(observed_runner)
            for key, value in expected.items():
                torch.testing.assert_close(observed.state_dict()[key], value, rtol=0, atol=0)
            self.assertEqual(capture.completed, 1)
            self.assertFalse(capture.failure_saved)
            self.assertIsNotNone(capture.teacher_state)
            payload = torch.load(capture.output / 'capture.pt', weights_only=False)
            self.assertEqual(payload['stage'], 'finite_limit')
            replay, _ = fixture.initialized_model()
            replay.ema_model.load_state_dict(payload['teacher_state'])
            replay.load_state_dict(payload['pre_forward']['model'])
            for name, value in payload['pre_forward']['python_state'].items():
                setattr(replay, name, value)
            replay_optimizer = torch.optim.SGD(replay.parameters(), lr=.001, momentum=.9)
            replay_optimizer.load_state_dict(payload['pre_forward']['optimizer'])
            rng = payload['pre_forward']['rng']
            torch.set_rng_state(rng['torch'])
            np.random.set_state(rng['numpy'])
            diagnostic.random.setstate(rng['python'])
            replay_runner = SimpleNamespace(model=replay, optimizer=replay_optimizer)
            replay_runner.outputs = replay.train_step(payload['actual_batch'], replay_optimizer)
            self.assertEqual(replay_runner.outputs['log_vars'], observed_runner.outputs['log_vars'])
            OptimizerHook().after_train_iter(replay_runner)
            for key, value in expected.items():
                torch.testing.assert_close(replay.state_dict()[key], value, rtol=0, atol=0)
            for name, parameter in replay.named_parameters():
                if parameter.grad is not None:
                    torch.testing.assert_close(
                        parameter.grad, payload['gradients'][name], rtol=0, atol=0)
        self.assertEqual('train_step' not in MDPOBB.__dict__, inherited)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
