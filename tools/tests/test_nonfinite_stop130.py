"""One CPU trajectory/stop check; not a reproduction of the research NaNs."""

import copy
import unittest

import torch

from experiments.comparison.nonfinite_stop130 import (
    STOP_AFTER, WindowComplete, stop_after_iteration,
)
from experiments.comparison.numerical_integrity import optimizer_boundary, require_finite


class RunnerFixture:
    def __init__(self):
        self.iter = 0
        self.parameter = torch.nn.Parameter(torch.tensor([.7, -.2], dtype=torch.float32))
        self.optimizer = torch.optim.SGD([self.parameter], lr=.02, momentum=.9, weight_decay=.0001)
        self.ema = self.parameter.detach().clone()
        self.updates = 0
        self.losses = []

    def call_hook(self, name):
        if name == "after_train_iter":
            self.optimizer.zero_grad()
            loss = (self.parameter - torch.tensor([.1, .3])).square().sum()
            require_finite(loss, "fixture loss before backward")
            self.losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.ema.mul_(.999).add_(self.parameter.detach(), alpha=.001)
            self.updates += 1
        return "original_hook_result"


class StopWindowTest(unittest.TestCase):
    def test_preserves_updates_rng_and_exact_stop_without_masking_failure(self):
        original_method = RunnerFixture.call_hook
        anomaly_before = torch.is_anomaly_enabled()
        rng_before = torch.get_rng_state().clone()
        baseline = RunnerFixture()
        initial = baseline.parameter.detach().clone()
        with optimizer_boundary():
            for iteration in range(STOP_AFTER):
                baseline.iter = iteration
                baseline.call_hook("before_train_iter")
                baseline.call_hook("after_train_iter")
        expected_state = copy.deepcopy(baseline.optimizer.state_dict())
        candidate, events = RunnerFixture(), []
        with self.assertRaises(WindowComplete), optimizer_boundary(), \
                stop_after_iteration(RunnerFixture, events.append):
            for iteration in range(STOP_AFTER + 1):
                candidate.iter = iteration
                self.assertEqual(candidate.call_hook("before_train_iter"), "original_hook_result")
                candidate.call_hook("after_train_iter")
        self.assertEqual(candidate.updates, 130)
        self.assertEqual([event["completed_iterations"] for event in events], list(range(1, 131)))
        self.assertTrue(torch.equal(candidate.parameter, baseline.parameter))
        self.assertTrue(torch.equal(candidate.ema, baseline.ema))
        self.assertEqual(candidate.losses, baseline.losses)
        self.assertGreater(candidate.losses[0], 0)
        self.assertFalse(torch.equal(candidate.parameter, initial))
        self.assertEqual(candidate.optimizer.state_dict()["param_groups"], expected_state["param_groups"])
        self.assertTrue(torch.equal(candidate.optimizer.state_dict()["state"][0]["momentum_buffer"],
                                    expected_state["state"][0]["momentum_buffer"]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        self.assertEqual(torch.is_anomaly_enabled(), anomaly_before)
        self.assertIs(RunnerFixture.call_hook, original_method)

        class FailingRunner:
            iter = 0

            def call_hook(self, name):
                raise FloatingPointError("actual native finite-boundary failure")

        failed_events = []
        with stop_after_iteration(FailingRunner, failed_events.append), \
                self.assertRaisesRegex(FloatingPointError, "native finite-boundary"):
            FailingRunner().call_hook("after_train_iter")
        self.assertEqual(failed_events, [])


if __name__ == "__main__":
    unittest.main()
