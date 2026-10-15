"""CPU finite/DDP dispatch for the two approved single-coordinate F deletions."""

import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from experiments.comparison import extension_training as training, f_deletion
from experiments.comparison.finite_resumer import Cell, load_cells, runner_command, train_state
from experiments.comparison.collect_report_manifest import load_resolver
from tools.tests import test_extension_training as fixtures


class FDeletionExecutionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExtensionTrainingTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.root / "sarclip.safetensors"
        self.base.touch()
        (self.fixture.core / "train.py").write_text("# frozen F; never executed\n")

    def spec(self, paths, dataset, method, base_weights, overrides):
        self.assertEqual(overrides["data.samples_per_gpu"], 16)
        self.assertTrue(overrides["find_unused_parameters"])
        self.assertNotIn("model.cfg.use_bbox_reg", overrides)
        environment = {"SARCLIP_PRETRAINED": str(self.base),
                       "CGA_BLEND_DET_WEIGHT": "1.0" if method == "F_veto_only" else "0.7"}
        return {
            "training_code": str(self.fixture.core), "training_code_sha": f_deletion.TRAINING_CODE_SHA,
            "config_text": "model = dict(cfg=dict(use_bbox_reg=False))\n",
            "config_diff": ([{"path": f_deletion.ALPHA_PATH, "before": .5, "after": 1.}]
                            if method == "F_text_only" else []),
            "environment_diff": ([{"path": f_deletion.BLEND_KEY, "before": "0.7", "after": "1.0"}]
                                 if method == "F_veto_only" else []),
            "effective_model_environment": environment,
            "wrapper_environment": {"HOME": "/tmp/iraod_clip_home",
                                    "CLIP_DOWNLOAD_ROOT": f_deletion.CLIP_CACHE},
        }

    def test_exact72_cells_ddp2_flags_and_final_evidence(self):
        with patch.object(f_deletion, "build_f_deletion_spec", side_effect=self.spec):
            self.fixture.prepare(training.F_DELETIONS, sarclip_base=self.base)
        queue = self.fixture.queue
        cells = load_cells([queue / "train.list"], [queue / "eval.list"])
        self.assertEqual(len(cells), 72)
        self.assertTrue(all(c.width == 2 for c in cells))
        subprocess.run(["bash", "-n", str(queue / "run_train_2gpu.sh")], check=True)
        runtime = self.fixture.runtime()
        audits = json.loads((queue / "f_deletion_config_diff.json").read_text())
        self.assertEqual(len(audits), 72)
        for method in training.F_DELETIONS:
            cell = runtime["cells"][f"DIOR/cloudy/44/{method}"]
            self.assertTrue(cell["work_dir"].endswith("/ddp2/work"))
            self.assertTrue(cell["terminal_status"].endswith("/ddp2/terminal_status"))
            self.assertTrue(cell["eval_dir"].endswith("/ddp2/eval_full_cloudy_ids_v1"))
            self.assertFalse(cell["use_bbox_reg"])
            with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "SARCLIP_LORA": "/bad.pth"}), \
                    patch.object(training, "code_sha", return_value=f_deletion.TRAINING_CODE_SHA), \
                    patch.object(training.subprocess, "run",
                                 side_effect=lambda *a, **kw: self.fixture.complete(cell)) as run:
                training.train_ddp(queue, "4,5", 29804, "DIOR", "cloudy", 44, method)
            command, env = run.call_args.args[0], run.call_args.kwargs["env"]
            self.assertEqual(command[1:3], ["-m", "torch.distributed.launch"])
            self.assertIn("--nproc_per_node=2", command)
            self.assertIn("--master_port=29804", command)
            self.assertIn("data.samples_per_gpu=16", command)
            self.assertIn("model.cfg.use_bbox_reg=False", command)
            self.assertIn("find_unused_parameters=True", command)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4,5")
            self.assertEqual(env["HOME"], "/tmp/iraod_clip_home")
            self.assertNotIn("SARCLIP_LORA", env)
            self.assertEqual(env["CGA_BLEND_DET_WEIGHT"],
                             "1.0" if method == "F_veto_only" else "0.7")
            self.assertEqual(train_state(
                queue, load_resolver(queue / "paths.py"), Cell("DIOR", "cloudy", 44, method)), "complete")

    def test_rejects_wrong_gpu_pair_and_wrong_single_topology(self):
        with patch.object(f_deletion, "build_f_deletion_spec", side_effect=self.spec):
            self.fixture.prepare(("F_text_only",), sarclip_base=self.base)
        with self.assertRaisesRegex(ValueError, "approved pair"):
            training.train_ddp(self.fixture.queue, "4,6", 29804, "RSAR", "clean", 42, "F_text_only")
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                self.assertRaisesRegex(ValueError, "topology"):
            training.train(self.fixture.queue, 4, "RSAR", "clean", 42, "F_text_only")
        command = runner_command(self.fixture.queue, Cell("RSAR", "clean", 42, "F_text_only"),
                                 "train", (6, 7))
        self.assertEqual(command[2:4], ["6,7", "29806"])


if __name__ == "__main__":
    unittest.main()
