"""Observer passthrough on the actual MDP train_step and native CPU loss stack."""

from copy import deepcopy
from pathlib import Path
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


class MDPCaptureNativeTest(unittest.TestCase):
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
        self.assertEqual('train_step' not in MDPOBB.__dict__, inherited)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
