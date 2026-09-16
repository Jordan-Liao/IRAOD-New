"""Verify native epoch-final EMA, budget metadata and paired checkpoint saving."""

from copy import deepcopy
import logging
from pathlib import Path
import sys
import tempfile
import unittest

import torch

import test_mdp_integration as mdp_fixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/comparison'))
import mdp_recover_final_pair as recovery


class MDPRecoveryTest(unittest.TestCase):
    def test_native_finalization_updates_ema_once_and_saves_budgeted_pair(self):
        fixture = mdp_fixture.MDPIntegrationTest()
        fixture.setUp()
        model, _ = fixture.initialized_model()
        teacher_before = deepcopy(model.ema_model.state_dict())
        with torch.no_grad():
            next(model.parameters()).add_(.1)
        model.cur_iter = 265
        expected_student = deepcopy(model.state_dict())
        expected_teacher = {
            name: (expected_student[name] * (1 - .9) + value * .9).to(value.dtype)
            for name, value in teacher_before.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            runner = recovery.SemiEpochBasedRunner(
                max_epochs=1,
                model=model, optimizer=torch.optim.SGD(model.parameters(), lr=.02),
                work_dir=directory, logger=logging.getLogger('mdp-recovery-test'))
            runner._iter = runner._inner_iter = 264
            runner._epoch = 0
            recovery.finalize_epoch(runner, 265)
            for suffix, expected in (('', expected_student), ('_ema', expected_teacher)):
                payload = torch.load(
                    Path(directory) / f'iter_266{suffix}.pth', weights_only=False)
                self.assertEqual(payload['meta']['epoch'], 1)
                self.assertEqual(payload['meta']['iter'], 266)
                self.assertEqual(payload['optimizer']['param_groups'][0]['lr'], .02)
                self.assertEqual(payload['state_dict'].keys(), expected.keys())
                for name, value in expected.items():
                    torch.testing.assert_close(payload['state_dict'][name], value, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, 'prescribed epoch budget'):
                recovery.finalize_epoch(runner, 265)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
