"""DIOR24 CPU preparation through actual config/admission/queue consumer seams."""

from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from mmcv import Config
import torch

from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite
from experiments.comparison import host_binding as host
from experiments.comparison import mixed_queue, oracle_training as oracle
from experiments.comparison.b_regression import config_diff
from experiments.comparison.labels import CLASSES, ORACLE_TRAINING_TEMPLATES
from experiments.comparison.result_completion import DOMAINS
from tools.tests import test_extension_training, test_oracle_adapters


class OracleTrainingTest(unittest.TestCase):
    def setUp(self):
        test_extension_training.ExtensionTrainingTest.setUp(self)
        adapter_fixture = test_oracle_adapters.OracleAdapterTest()
        adapter_fixture.setUp()
        self.addCleanup(adapter_fixture.doCleanups)
        self.adapter = adapter_fixture.path
        self.weights = adapter_fixture.base
        self.weights.write_bytes(b"synthetic base; no native model loading")
        self.payload = deepcopy(adapter_fixture.payload)
        self.payload.update(
            dataset="DIOR", classes=list(CLASSES["DIOR"]),
            templates=[ORACLE_TRAINING_TEMPLATES["DIOR"]],
            metadata_row_count=3,
            class_counts={name: 3 if i == 0 else 0
                          for i, name in enumerate(CLASSES["DIOR"])},
            crop_modes={"aabb": 3}, crop_expansions={"0.4": 3},
            corruptions={domain: 1 for domain in DOMAINS["DIOR"][1:]},
            sampler=dict(type="inverse_class_weighted", replacement=True,
                         num_samples=3, seed=42))
        torch.save(self.payload, self.adapter)
        self.teacher = self.core / "dior_source_config.py"
        self.teacher.write_text(
            "model = dict(type='OrientedRCNN', backbone=dict(type='OrthoNet', depth=50),\n"
            "    roi_head=dict(bbox_head=dict(num_classes=20)), test_cfg=dict(rcnn=dict(score_thr=.05)))\n")
        self.originals = {}
        for baseline in ("D", "F"):
            config = self.core / f"dior_{baseline}.py"
            cfg = dict(strict_source_free=True, weight_l=0., weight_u=1.,
                       score_thr=.7, use_bbox_reg=False)
            if baseline == "F":
                cfg.update(
                    vlst_enabled=True, vlst_strict=True, vlst_lora_path=None,
                    vlst_text_visual_alpha=.5, vlst_loss_weight=.1,
                    vlst_temperature=.07, vlst_prototype_momentum=.9,
                    vlst_pretrained="/stale/base", vlst_cache_dir="/stale/cache")
            config.write_text(
                "import os\n"
                "custom_imports = dict(imports=['sfod', 'mmdet_extension'], allow_failed_imports=False)\n"
                f"model = {dict(type='UnbiasedTeacher' if baseline == 'D' else 'UnbiasedTeacherVLST', ema_config=str(self.teacher), cfg=cfg, roi_head=dict(bbox_head=dict(num_classes=20)))!r}\n"
                f"data = {dict(samples_per_gpu=32, train=dict(type='StrictSourceFreeDOTADataset', classes=CLASSES['DIOR'], unlabeled_epoch_size=5863, unlabeled_subset_seed=42))!r}\n"
                "optimizer = dict(type='SGD', lr=.02, momentum=.9)\n"
                "runner = dict(type='SemiEpochBasedRunner', max_epochs=1)\n"
                "checkpoint_config = dict(interval=1)\n"
                "custom_hooks = [dict(type='NumClassCheckHook')]\n"
                "os.environ.update(CGA_SCORER='sarclip', CGA_BACKEND='sarclip', CGA_STRICT='1',\n"
                "    CGA_FILTER_MODE='veto_soft', CGA_DROP_SCORE='0.0', CGA_VETO_PRED_THR='0.7',\n"
                "    CGA_VETO_LABEL_THR='0.1', CGA_PROTECT_DET_SCORE='0.9', CGA_BLEND_DET_WEIGHT='0.7')\n"
                + ("os.environ['VLST_BACKEND'] = 'sarclip'\n" if baseline == "F" else ""))
            self.originals[baseline] = config
        with self.paths.open("a") as stream:
            stream.write(
                "def dior_cfg(method): return f'{ROOT}/dior_{method}.py'\n"
                "def rsar_cfg(method): raise AssertionError('RSAR must not be requested')\n")

    def prepare(self):
        return oracle.prepare(
            self.base, self.report, self.paths, self.queue, self.artifacts,
            self.eval_code, self.python, self.adapter, self.weights)

    def runtime(self):
        return json.loads((self.queue / "runtime.json").read_text())

    def test_exact24_executable_configs_preserve_science_and_source(self):
        before = {p: p.read_bytes() for p in (
            self.base, self.report, self.paths, self.teacher, self.adapter,
            self.core / "DIOR_source.pth", *self.originals.values())}
        require_file = training.require_file

        def dior_only(path):
            self.assertNotIn("RSAR", str(path))
            return require_file(path)

        with patch.object(training, "require_file", side_effect=dior_only), patch.dict(
                os.environ, {"SARCLIP_LORA": "/strictAF/leaked.pth",
                             "CGA_BLEND_DET_WEIGHT": ".123",
                             "VLST_BACKEND": "poison"}):
            self.prepare()
            self.assertEqual(os.environ["SARCLIP_LORA"], "/strictAF/leaked.pth")
        runtime = self.runtime()
        self.assertEqual((runtime["train_cells"], runtime["eval_cells"]), (24, 24))
        self.assertEqual(runtime["domains"], {"DIOR": list(DOMAINS["DIOR"])})
        self.assertEqual(runtime["source_training_cells"], 0)
        self.assertNotIn("student_cells", runtime)
        self.assertFalse(self.artifacts.exists())
        cells = runtime["cells"].values()
        self.assertEqual(Counter(c["method"] for c in cells),
                         {"LoRA-CGA": 12, "LoRA-CGA+VLST": 12})
        self.assertEqual(Counter(c["seed"] for c in cells), {42: 8, 43: 8, 44: 8})
        self.assertEqual(Counter(c["domain"] for c in cells),
                         {domain: 6 for domain in DOMAINS["DIOR"]})
        for cell in cells:
            width = 2 if cell["method"].endswith("+VLST") else 1
            self.assertEqual((cell["world_size"], cell["samples_per_gpu"]), (width, 32 // width))
            self.assertFalse(cell["use_bbox_reg"])
            self.assertTrue(cell["checkpoint"].endswith("/iter_185_ema.pth"))
            self.assertTrue(cell["student_checkpoint"].endswith("/iter_185.pth"))
            self.assertEqual(cell["fairness_group"], "Target-supervised")
            self.assertTrue(cell["appendix_only"])
            self.assertEqual(cell["source_checkpoint"], str(self.core / "DIOR_source.pth"))
            self.assertEqual(cell["eval_config"], str(self.eval_code / "DIOR_source.py"))
            self.assertTrue(cell["target_val"].endswith(f"{cell['domain']}/val"))
        for method, baseline in zip(training.ORACLE_METHODS, ("D", "F")):
            cell = next(c for c in cells if c["method"] == method)
            with oracle.model_environment(self.weights):
                old = Config.fromfile(str(self.originals[baseline]), import_custom_modules=False)
                new = Config.fromfile(cell["config"], import_custom_modules=False)
                teacher = Config.fromfile(str(self.teacher), import_custom_modules=False)
                self.assertNotIn("SARCLIP_LORA", os.environ)
                self.assertEqual(os.environ["CGA_BLEND_DET_WEIGHT"], "0.7")
                self.assertEqual(os.environ["CGA_VETO_PRED_THR"], "0.7")
                self.assertEqual(os.environ["CGA_PROTECT_DET_SCORE"], "0.9")
            self.assertEqual(config_diff(teacher.model, new.model.ema_config["model"]),
                             [dict(path="type", before="OrientedRCNN", after="OrientedRCNN_CGA")])
            changes = {d["path"] for d in config_diff(old, new)}
            allowed = {"model.type", "model.ema_config", "custom_imports.imports",
                       "model.cfg.oracle_dataset", "model.cfg.oracle_adapter",
                       "model.cfg.oracle_base_weights", "model.cfg.vlst_pretrained",
                       "model.cfg.vlst_cache_dir"}
            self.assertFalse(changes - allowed, changes)
            if baseline == "F":
                self.assertEqual(new.model.cfg.vlst_pretrained, str(self.weights))
                self.assertEqual(new.model.cfg.vlst_text_visual_alpha, .5)
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data)

    def test_native_invocation_admission_and_mixed_origin_routing(self):
        self.prepare()
        mixed = self.root / "mixed"
        mixed_queue.prepare([self.queue], mixed)
        scope = finite.load_cells([mixed / "train.list"], [mixed / "eval.list"])
        self.assertEqual((len(scope), sum(scope.values())), (24, 24))
        paths = finite.load_paths(mixed)
        for cell in scope:
            runtime, binding = training.load_cell(
                mixed, cell.dataset, cell.domain, cell.seed, cell.method)
            self.assertEqual(runtime["artifact_root"], str(self.artifacts))
            self.assertEqual(finite.prerequisite_state(paths, cell), ("ready", ""))
            gpus = (4, 5) if cell.width == 2 else (4,)
            with patch.dict(os.environ, {"SARCLIP_LORA": "/poison.pth"}):
                _, _, command, env = training.training_invocation(
                    mixed, runtime, binding, gpus, 29804 if cell.width == 2 else None,
                    Path(binding["work_dir"]))
            self.assertIn("model.cfg.use_bbox_reg=False", command)
            self.assertIn("optimizer.lr=0.02", command)
            self.assertIn(f"data.samples_per_gpu={32 // cell.width}", command)
            self.assertIn("load_from=" + binding["source_checkpoint"], command)
            self.assertIn("--no-validate", command)
            self.assertNotIn("SARCLIP_LORA", env)
            self.assertEqual(env["SARCLIP_PRETRAINED"], str(self.weights))
            self.assertEqual(env["CGA_BLEND_DET_WEIGHT"], "0.7")
            with patch.object(training, "evaluate_binding") as evaluate, patch.object(
                    host, "read_json", wraps=host.read_json) as read:
                execution = Path(binding["method_dir"]) / "execution.json"
                # Execution receipt is the only unavailable producer artifact.
                read.side_effect = lambda path: (
                    dict(training_code_sha="producer", training_code=str(training.ROOT))
                    if Path(path) == execution else read._mock_wraps(path))
                training.evaluate(mixed, 4, cell.dataset, cell.domain, cell.seed, cell.method)
            eval_cell, eval_runtime, _ = evaluate.call_args.args
            self.assertEqual(eval_cell["checkpoint"], binding["checkpoint"])
            self.assertEqual(eval_cell["config"], binding["eval_config"])
            self.assertEqual(eval_runtime["evaluation_code"], str(self.eval_code))
        bad = deepcopy(self.payload)
        bad["epochs_completed"] = 9
        torch.save(bad, self.adapter)
        state, reason = finite.prerequisite_state(paths, next(iter(scope)))
        self.assertEqual(state, "waiting")
        self.assertIn("frozen dataset/TRAIN/final-epoch", reason)

    def test_selected_host_alias_admission_and_rank_handler_commands(self):
        self.prepare()
        cell = finite.Cell("DIOR", "cloudy", 44, "LoRA-CGA+VLST")
        old_base = "/mnt/shared/zechuan/iraod_weights/base.safetensors"
        self.payload["sarclip_pretrained"] = old_base
        torch.save(self.payload, self.adapter)
        with patch.object(host, "is_target_host", return_value=True), patch.object(
                host, "PREFIXES", ((old_base, str(self.weights)),)):
            training.require_prerequisites(self.runtime()["cells"][cell.key])
            command = finite.runner_command(self.queue, cell, "train", (0, 1))
            self.assertIn("train-ddp", command)
            self.assertEqual(command[command.index("--port") + 1], "29804")
            runtime, binding = training.load_cell(
                self.queue, cell.dataset, cell.domain, cell.seed, cell.method)
            _, _, command, env = training.training_invocation(
                self.queue, runtime, binding, (0, 1), 29804, Path(binding["work_dir"]))
            self.assertIn(str(Path(host.__file__).absolute()), command)
            self.assertIn("native", command)
            self.assertIn("--nproc_per_node=2", command)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(env["MASTER_PORT"], "29804")
        self.payload["sarclip_pretrained"] = "/unrelated/base.safetensors"
        torch.save(self.payload, self.adapter)
        with self.assertRaisesRegex(ValueError, "declared base SARCLIP"):
            training.require_prerequisites(self.runtime()["cells"][cell.key])

    def test_wrong_payload_blocks_before_metadata_and_no_extra_scopes(self):
        self.payload["dataset"] = "RSAR"
        torch.save(self.payload, self.adapter)
        with self.assertRaisesRegex(ValueError, "frozen dataset/TRAIN/final-epoch"):
            self.prepare()
        self.assertFalse(self.queue.exists())
        self.assertFalse(self.artifacts.exists())
        for row in ("RSAR clean 42 LoRA-CGA", "DIOR clean 45 LoRA-CGA",
                    "DIOR clean 42 LoRA-CGA student"):
            listing = self.root / "invalid.list"
            listing.write_text(row + "\n")
            with self.assertRaisesRegex(ValueError, "outside the approved"):
                finite.load_cells([], [listing])

    def test_source_teacher_architecture_mismatch_is_not_replaced(self):
        self.teacher.write_text("model = dict(type='DifferentDetector', roi_head=dict(bbox_head=dict(num_classes=20)))\n")
        with self.assertRaisesRegex(ValueError, "matching 20-class DIOR"):
            self.prepare()
        self.assertFalse(self.queue.exists())

    def test_producer_does_not_mark_unadmitted_oracle_ready(self):
        cell = finite.Cell("DIOR", "clean", 42, "LoRA-CGA")
        start = Mock(side_effect=AssertionError("unadmitted oracle must not launch"))
        backend = SimpleNamespace(
            queue=self.root, run_dir=self.root / "finite_state", producer=lambda: nullcontext(),
            evidence=lambda *args: "pending", available=lambda: {4},
            admission=lambda c, phase: ("waiting", "adapter consumer admission pending"),
            discover_external_owners=lambda *args: [], discover=lambda *args: [],
            start=start)
        result = finite.run_finite({cell: True}, backend)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["cells"][0]["train"], "waiting")
        self.assertEqual(result["cells"][0]["eval"], "waiting")
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
