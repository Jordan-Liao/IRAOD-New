"""CPU-only effective-step accounting and real frozen command/lock construction."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite
from experiments.comparison import smoke_frozen_training as smoke
from experiments.comparison.b_regression import TRAINING_CODE_SHA
from experiments.comparison.result_completion import write_json


class SmokeEvidenceTest(unittest.TestCase):
    def test_counts_actual_optimizer_updates_and_records_native_loss_delta_and_ema(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1, bias=False)
            # As in frozen SemiBaseDetector, teacher is not part of the optimizer.
            teacher = torch.nn.Linear(2, 1, bias=False)
            teacher.load_state_dict(model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=.02)
            model.ema_model = teacher
            runner = SimpleNamespace(model=model, optimizer=optimizer, iter=0)
            hook = smoke.SmokeEvidence(2, Path(directory) / "rank_0.json")
            hook.before_run(runner)
            for index in range(3):
                with torch.no_grad():
                    teacher.weight.mul_(.998).add_(model.weight, alpha=.002)
                loss = model(torch.ones(1, 2)).square().sum() + model.weight.sum()
                runner.outputs = dict(loss=loss, log_vars={"native_loss": float(loss.detach())})
                runner.iter = index
                if index != 0:  # A train iteration is not an effective optimizer step.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if index == 2:
                    with self.assertRaises(smoke.SmokeFinished):
                        hook.after_train_iter(runner)
                else:
                    hook.after_train_iter(runner)
                    self.assertFalse(hook.output.exists())
            evidence = json.loads(hook.output.read_text())
            self.assertEqual(evidence["status"], "NON_RESULT")
            self.assertTrue(evidence["bounded_pass"])
            self.assertEqual(evidence["optimizer_updates"], 2)
            self.assertEqual(len(evidence["losses"]), 3)
            self.assertGreater(evidence["student_parameter_l1_delta"], 0)
            self.assertGreater(evidence["ema_parameter_l1_delta"], 0)
            self.assertEqual(list(Path(directory).glob("*.pth")), [])


class FrozenCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        (self.queue / "with_gpu_lock.sh").write_text(f"LOCKDIR={self.root / 'locks/gpu'}\n")
        source = self.root / "source.pth"
        source.write_bytes(b"fixture source")
        cells = {}
        for method in ("B_REG", "F_text_only"):
            cells[f"DIOR/clean/42/{method}"] = dict(
                dataset="DIOR", domain="clean", seed=42, method=method,
                source_checkpoint=str(source), training_code="/frozen0f98",
                training_code_sha=TRAINING_CODE_SHA, config=f"/preserved/{method}.py",
                world_size=1 if method == "B_REG" else 2,
                use_bbox_reg=method == "B_REG", target_val="/image-only/val",
                unlabeled_epoch_size=5863, method_dir=str(self.root / "formal" / method),
                model_environment={"CGA_BLEND_DET_WEIGHT": "0.7"},
                wrapper_environment={"SARCLIP_BASE": "/frozen/sarclip.pt"},
            )
        write_json(self.queue / "runtime.json", {
            "cells": cells, "python": "/venv/bin/python", "training_code": "/orchestration"})

    def test_real_legacy_b_reg_binding_without_new_topology_fields(self):
        # Exact cell from b_reg_manifests_0e5f8f7 and release_manifests_752a139.
        binding = json.loads((Path(__file__).parent / "fixtures"
                              / "b_reg_0e5f8f7_dior_clean_42.json").read_text())
        self.assertNotIn("world_size", binding)
        self.assertNotIn("samples_per_gpu", binding)
        self.assertNotIn("use_bbox_reg", binding)
        key = "DIOR/clean/42/B_REG"
        write_json(self.queue / "runtime.json", {
            "cells": {key: binding}, "python": "/venv/bin/python",
            "training_code": "/original/orchestration",
        })
        release = self.root / "release"
        release.mkdir()
        (release / "with_gpu_lock.sh").symlink_to(self.queue / "with_gpu_lock.sh")
        write_json(release / "runtime.json", {
            "cells": {key: binding}, "source_queues": {key: str(self.queue)},
        })
        out = self.root / "legacy_smoke"

        def launch(command, **kwargs):
            self.assertEqual(kwargs["cwd"], Path(binding["training_code"]))
            self.assertIn("data.samples_per_gpu=32", command)
            self.assertIn("model.cfg.use_bbox_reg=True", command)
            self.assertIn("optimizer.lr=0.02", command)
            self.assertIn("load_from=" + binding["source_checkpoint"], command)
            self.assertIn(binding["config"], command)
            self.assertIn("model.ema_ckpt=" + binding["source_checkpoint"], command)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "6")
            self.assertNotIn("--nproc_per_node=2", command)
            self.assertIsNone(finite.take_lock(self.root / "locks/gpu/gpu6.lock"))
            write_json(out / "rank_0.json", {
                "status": "NON_RESULT", "bounded_pass": True, "optimizer_updates": 2,
            })

        with patch.object(training, "code_sha", return_value=TRAINING_CODE_SHA), \
                patch.object(training, "require_file", side_effect=Path), \
                patch.object(finite, "idle_devices", return_value={6}), \
                patch.object(smoke.subprocess, "run", side_effect=launch) as native:
            self.assertEqual(smoke.run(release, "DIOR", "clean", 42, "B_REG", "6", out), out)
        native.assert_called_once()
        self.assertEqual(json.loads((release / "runtime.json").read_text())["cells"][key], binding)
        evidence = json.loads((out / "invocation.json").read_text())
        self.assertEqual(evidence["world_size"], 1)
        self.assertIn("finite_resumer.Cell.width", evidence["topology_basis"])

    def test_outer_gpu_lock_blocks_smoke_even_with_owned_environment_flag(self):
        outer = finite.take_lock(self.root / "locks/gpu/gpu4.lock")
        self.assertIsNotNone(outer)
        try:
            with patch.object(training, "code_sha", return_value=TRAINING_CODE_SHA), \
                    patch.object(finite, "idle_devices", return_value={4, 5}), \
                    patch.object(smoke.subprocess, "run") as native, \
                    patch.dict(smoke.os.environ, {"IRAOD_GPU_LOCKED": "1"}):
                with self.assertRaisesRegex(RuntimeError, "Actual shared lock busy"):
                    smoke.run(self.queue, "DIOR", "clean", 42, "F_text_only", "4,5",
                              self.root / "nested")
            native.assert_not_called()
            self.assertFalse((self.root / "nested").exists())
        finally:
            outer.close()

    def test_b_reg_and_one_f_control_use_real_frozen_training_with_fresh_nonresult_outputs(self):
        for method, gpus in (("B_REG", "6"), ("B_REG", "7"),
                             ("F_text_only", "4,5"), ("F_text_only", "6,7")):
            out = self.root / f"smoke_{method}_{gpus.replace(',', '_')}"

            def launch(command, **kwargs):
                self.assertEqual(kwargs["cwd"], Path("/frozen0f98"))
                self.assertEqual(kwargs["env"]["PYTHONPATH"], "/frozen0f98")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], gpus)
                self.assertIn("checkpoint_config=None", command)
                self.assertNotIn("checkpoint_config.save_last=True", command)
                self.assertEqual(command[command.index("--frozen-train") + 1], "/frozen0f98/train.py")
                self.assertIn(f"/preserved/{method}.py", command)
                self.assertIn("model.ema_ckpt=" + str(self.root / "source.pth"), command)
                self.assertIn("optimizer.lr=0.02", command)
                self.assertNotIn("runner.max_epochs=1", command)
                if method == "F_text_only":
                    port = 29804 if gpus == "4,5" else 29806
                    self.assertEqual(command[1:5], [
                        "-m", "torch.distributed.launch", "--nproc_per_node=2", f"--master_port={port}"])
                    self.assertIn("data.samples_per_gpu=16", command)
                    self.assertEqual(kwargs["env"]["CGA_BLEND_DET_WEIGHT"], "0.7")
                    self.assertEqual(kwargs["env"]["SARCLIP_BASE"], "/frozen/sarclip.pt")
                else:
                    self.assertIn("data.samples_per_gpu=32", command)
                    self.assertIn("model.cfg.use_bbox_reg=True", command)
                # Actual locks are held for the entire frozen subprocess lifetime.
                for gpu in map(int, gpus.split(",")):
                    self.assertIsNone(finite.take_lock(self.root / f"locks/gpu/gpu{gpu}.lock"))
                for rank in range(len(gpus.split(","))):
                    write_json(out / f"rank_{rank}.json",
                               dict(status="NON_RESULT", bounded_pass=True, optimizer_updates=2))

            with patch.object(training, "code_sha", return_value=TRAINING_CODE_SHA), \
                    patch.object(finite, "idle_devices", return_value={4, 5, 6, 7}), \
                    patch.object(smoke.subprocess, "run", side_effect=launch) as run:
                self.assertEqual(smoke.run(self.queue, "DIOR", "clean", 42, method, gpus, out), out)
                run.assert_called_once()
            self.assertEqual(json.loads((out / "invocation.json").read_text())["status"], "NON_RESULT")
            self.assertFalse((self.root / "formal").exists())
            self.assertEqual(list(self.root.rglob("terminal_status")), [])
            with self.assertRaises(FileExistsError):
                smoke.run(self.queue, "DIOR", "clean", 42, method, gpus, out)

    def test_rejects_unapproved_gpu_wrong_pair_unbounded_or_formal_outputs(self):
        for method, gpus, updates in (
                ("B_REG", "3", 2), ("F_text_only", "4,6", 2),
                ("F_text_only", "4", 2), ("B_REG", "6", 5)):
            with self.assertRaises(ValueError):
                smoke.run(self.queue, "DIOR", "clean", 42, method, gpus, self.root / "smoke", updates)
        with self.assertRaisesRegex(ValueError, "separate"):
            smoke.run(self.queue, "DIOR", "clean", 42, "B_REG", "6", self.root / "formal/B_REG/smoke")

    def test_second_pair_still_rejects_foreign_occupancy(self):
        with patch.object(training, "code_sha", return_value=TRAINING_CODE_SHA), \
                patch.object(finite, "idle_devices", return_value={6}), \
                patch.object(smoke.subprocess, "run") as launch:
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                smoke.run(self.queue, "DIOR", "clean", 42, "F_text_only", "6,7", self.root / "busy")
        launch.assert_not_called()
        self.assertFalse((self.root / "busy").exists())


if __name__ == "__main__":
    unittest.main()
