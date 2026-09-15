"""CPU-only final-pair/budget publication checks with tiny synthetic checkpoints."""

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.comparison.nonfinite_recovery import load_checks, observe_budget, validate_final_pair


class RecoveryTest(unittest.TestCase):
    def test_natural_budgets_and_written_model_optimizer_finiteness(self):
        root = Path(__file__).resolve().parents[2]
        checks = load_checks(root / "experiments/comparison/numerical_integrity.py")

        class Runner:
            def call_hook(self, name):
                return name

        original = Runner.call_hook
        for epochs in (1, 2):
            state = {"completed_updates": 0, "completed_epochs": 0, "after_run": False}
            runner = Runner()
            runner.max_epochs, runner.data_loader = epochs, range(265)
            with observe_budget(Runner, epochs, state):
                for epoch in range(epochs):
                    runner.call_hook("before_train_epoch")
                    for step in range(265):
                        runner.iter = epoch * 265 + step
                        runner.call_hook("after_train_iter")
                    runner.call_hook("after_train_epoch")
                self.assertEqual(runner.call_hook("after_run"), "after_run")
            self.assertEqual(state, {"completed_updates": epochs * 265,
                                     "completed_epochs": epochs, "after_run": True})
            self.assertIs(Runner.call_hook, original)
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                iteration = epochs * 265 + 1
                case = {"work_dir": str(work), "epochs": epochs, "terminal_iteration": iteration,
                        "final_pair": {"student": f"iter_{iteration}.pth",
                                       "ema": f"iter_{iteration}_ema.pth"}}
                payload = {
                    "meta": {"epoch": epochs, "iter": iteration},
                    "state_dict": {"weight": torch.tensor([1., 2.])},
                    "optimizer": {"state": {0: {"momentum_buffer": torch.tensor([.1, .2])}},
                                  "param_groups": [{"params": [0], "lr": .02, "momentum": .9}]},
                }
                for filename in case["final_pair"].values():
                    torch.save(payload, work / filename)
                self.assertEqual(set(validate_final_pair(case, checks)), {"student", "ema"})
                for kind in ("model", "optimizer", "budget", "missing_optimizer"):
                    bad = copy.deepcopy(payload)
                    if kind == "model":
                        bad["state_dict"]["weight"][0] = float("nan")
                    elif kind == "optimizer":
                        bad["optimizer"]["state"][0]["momentum_buffer"][0] = float("inf")
                    elif kind == "budget":
                        bad["meta"]["iter"] -= 1
                    else:
                        del bad["optimizer"]
                    torch.save(bad, work / case["final_pair"]["ema"])
                    with self.assertRaises((FloatingPointError, ValueError)):
                        validate_final_pair(case, checks)
                    self.assertFalse((work / "recovery_result.json").exists())


if __name__ == "__main__":
    unittest.main()
