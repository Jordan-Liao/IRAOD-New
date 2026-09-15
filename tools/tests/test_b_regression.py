"""CPU unittest coverage for the frozen-B-only regression control."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mmcv import Config, ConfigDict

from experiments.comparison import b_regression as regression


class Transform:
    """Like torchvision transforms: deterministic repr, identity equality."""

    def __init__(self, strength=0.4):
        self.strength = strength

    def __repr__(self):
        return f"Transform(strength={self.strength!r})"


class BRegressionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="frozen B ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.code = self.root / "code 0f98"
        self.rsar = self.code / "configs/unbiased_teacher/sfod/rsar_B.py"
        self.rsar.parent.mkdir(parents=True)
        (self.code / "train.py").write_text("# frozen training entry; never executed\n")
        self.rsar.write_text(
            "from tools.tests.test_b_regression import Transform\n"
            "custom_imports = dict(imports=['must_not_import_gpu_models'])\n"
            "model = dict(type='UnbiasedTeacher', ema_ckpt='rsar_source.pth',\n"
            "    cfg=dict(use_bbox_reg=False, strict_source_free=True,\n"
            "             weight_l=0., weight_u=1.))\n"
            # No EMA momentum/timing override: the frozen model owns defaults.
            "data = dict(samples_per_gpu=2, workers_per_gpu=2, train=dict(\n"
            "    type='StrictSourceFreeDOTADataset', img_prefix='rsar/val/images',\n"
            "    unlabeled_epoch_size=8467, unlabeled_subset_seed=42,\n"
            "    classes=('ship', 'aircraft'), pipeline_share=[dict(type='Flip')],\n"
            "    pipeline_weak=[dict(type='Resize', img_scale=(800, 800))],\n"
            "    pipeline_strong=[dict(type='Apply', operations=[Transform()])]))\n"
            "custom_hooks = [dict(type='NumClassCheckHook')]\n"
            "optimizer = dict(type='SGD', lr=.001, momentum=.9, weight_decay=.0001)\n"
            "runner = dict(type='SemiEpochBasedRunner', max_epochs=1)\n"
            "load_from = 'rsar_source.pth'\n"
            "checkpoint_config = dict(interval=1)\n"
        )
        self.source = self.root / "dior_source/resolved_config.py"
        self.source.parent.mkdir()
        self.source.write_text("source_identity = 'original DIOR source'\n")
        self.dior = self.root / "queue_0f98a48/configs/dior_B_strict.py"
        self.dior.parent.mkdir(parents=True)
        self.dior.write_text(
            f"_base_ = [{str(self.rsar)!r}, {str(self.source)!r}]\n"
            "model = dict(ema_ckpt='dior_source.pth')\n"
            "load_from = 'dior_source.pth'\n"
            "data = dict(train=dict(img_prefix='dior/val', unlabeled_epoch_size=5863,\n"
            "                       classes=('airplane', 'airport', 'baseballfield')))\n"
        )
        self.paths = Mock()
        self.paths.rsar_cfg.return_value = str(self.rsar)
        self.paths.dior_cfg.return_value = str(self.dior)
        self.git = patch.object(
            regression.subprocess, "check_output",
            return_value=regression.TRAINING_CODE_SHA + "\n").start()
        self.addCleanup(patch.stopall)
        self.overrides = {
            "data.samples_per_gpu": 32, "optimizer.lr": .02,
            "model.cfg.strict_source_free": True, "model.cfg.weight_l": 0,
            "model.cfg.weight_u": 1, "data.train.type": "StrictSourceFreeDOTADataset",
        }
        self.expected = [{"path": "model.cfg.use_bbox_reg", "before": False, "after": True}]

    def build(self, dataset="RSAR", overrides=None):
        return regression.build_b_regression_spec(
            self.paths, dataset, self.overrides if overrides is None else overrides)

    def test_exact_diff_and_frozen_producer_for_each_dataset(self):
        for dataset, reference in (("RSAR", self.rsar), ("DIOR", self.dior)):
            with self.subTest(dataset=dataset):
                spec = self.build(dataset)
                self.assertEqual(spec["method"], "B_REG")
                self.assertEqual(spec["config_diff"], self.expected)
                self.assertEqual(spec["reference_config"], str(reference))
                self.assertEqual(spec["training_code"], str(self.code))
                self.assertEqual(spec["training_code_sha"], regression.TRAINING_CODE_SHA)
                self.assertEqual(spec["overlay_text"],
                                 f"_base_ = {str(reference)!r}\n"
                                 "model = dict(cfg=dict(use_bbox_reg=True))\n")
        self.paths.rsar_cfg.assert_called_with("B")
        self.paths.dior_cfg.assert_called_once_with("B")
        self.git.assert_called_with(
            ["git", "-C", str(self.code), "rev-parse", "HEAD"], text=True)

    def test_executable_overlay_preserves_original_science_and_common_options(self):
        for dataset, reference, size in (
                ("RSAR", self.rsar, 8467), ("DIOR", self.dior, 5863)):
            for seed in (42, 43, 44):
                with self.subTest(dataset=dataset, seed=seed):
                    overrides = dict(self.overrides, **{
                        "data.train.img_prefix": f"/targets/{dataset}/val",
                        "data.train.unlabeled_epoch_size": size,
                        "load_from": f"/source/{dataset}.pth",
                        "model.ema_ckpt": f"/source/{dataset}.pth",
                        "seed": seed, "work_dir": f"/new/B_REG/{dataset}/{seed}",
                        "checkpoint_config.max_keep_ckpts": 2,
                        "checkpoint_config.save_last": True,
                    })
                    spec = self.build(dataset, overrides)
                    overlay = self.root / "candidate.py"
                    overlay.write_text(spec["overlay_text"])
                    baseline = Config.fromfile(reference, import_custom_modules=False)
                    candidate = Config.fromfile(overlay, import_custom_modules=False)
                    baseline.merge_from_dict(deepcopy(overrides))
                    candidate.merge_from_dict(deepcopy(overrides))
                    self.assertEqual(regression.config_diff(baseline, candidate), self.expected)
                    self.assertEqual(candidate.model.type, "UnbiasedTeacher")
                    self.assertEqual(candidate.model.cfg,
                                     dict(use_bbox_reg=True, strict_source_free=True,
                                          weight_l=0, weight_u=1))
                    # B's absent momentum override selects .998 in frozen code;
                    # no after-optimizer or epoch-final teacher hook is added.
                    self.assertNotIn("momentum", candidate.model.cfg)
                    self.assertEqual(candidate.model.cfg.get("momentum", .998), .998)
                    self.assertEqual(candidate.custom_hooks, [dict(type="NumClassCheckHook")])
                    self.assertEqual(candidate.data.train.unlabeled_epoch_size, size)
                    self.assertEqual(candidate.data.samples_per_gpu, 32)
                    self.assertEqual(candidate.optimizer,
                                     dict(type="SGD", lr=.02, momentum=.9, weight_decay=.0001))
                    self.assertEqual(candidate.runner,
                                     dict(type="SemiEpochBasedRunner", max_epochs=1))
                    self.assertEqual(candidate.model.ema_ckpt, candidate.load_from)
                    self.assertEqual(candidate.seed, seed)
                    self.assertEqual(candidate.work_dir, overrides["work_dir"])
                    self.assertIn("historical B", spec["operational_note"])
                    if dataset == "DIOR":
                        self.assertEqual(candidate.source_identity, "original DIOR source")

    def test_no_writes_or_override_mutation(self):
        def snapshot():
            return {str(path.relative_to(self.root)): path.read_bytes()
                    for path in self.root.rglob("*") if path.is_file()}

        before = snapshot()
        overrides = deepcopy(self.overrides)
        self.build("DIOR")
        self.assertEqual(snapshot(), before)
        self.assertEqual(self.overrides, overrides)

    def test_semantic_containers_and_identity_only_transforms(self):
        before = Config(dict(pipeline=[ConfigDict(operations=(Transform(),))]))
        after = deepcopy(before)
        self.assertIsNot(before.pipeline[0].operations[0], after.pipeline[0].operations[0])
        self.assertEqual(regression.config_diff(before, after), [])
        after.pipeline[0].operations[0].strength = .8
        self.assertEqual(regression.config_diff(before, after), [{
            "path": "pipeline.0.operations.0",
            "before": "Transform(strength=0.4)", "after": "Transform(strength=0.8)",
        }])

    def test_rejects_wrong_sha_or_missing_training_entry(self):
        self.git.return_value = "new-sfut-producer\n"
        with self.assertRaisesRegex(ValueError, "requires training code"):
            self.build()
        self.git.return_value = regression.TRAINING_CODE_SHA
        (self.code / "train.py").unlink()
        with self.assertRaisesRegex(ValueError, "lacks train.py"):
            self.build()

    def test_rejects_non_b_and_already_enabled_regression(self):
        original = self.rsar.read_text()
        for old, new, error in (
                ("UnbiasedTeacher", "SFUTOBB", "UnbiasedTeacher"),
                ("use_bbox_reg=False", "use_bbox_reg=True", "must be False")):
            with self.subTest(new=new):
                self.rsar.write_text(original.replace(old, new))
                with self.assertRaisesRegex(ValueError, error):
                    self.build()

    def test_rejects_regression_overrides_even_false_or_nested(self):
        for override in (
                {"model.cfg.use_bbox_reg": True},
                {"model.cfg.use_bbox_reg": False},
                {"model": {"cfg": {"use_bbox_reg": True}}},
                {"model.cfg": {"use_bbox_reg": False}}):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "must not contain use_bbox_reg"):
                    self.build(overrides=dict(self.overrides, **override))

    def test_rejects_changed_algorithm_or_unknown_option(self):
        for override in (
                {"model.type": "SFUTOBB"},
                {"model.cfg.momentum": .9996},
                {"model.cfg.weight_u": 2},
                {"custom_hooks": [dict(type="AfterOptimizerTeacherHook")]},
                {"data.train.pipeline_weak": [dict(type="NewAugmentation")]},
                {"unknown_launch_option": True}):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "change B algorithm fields"):
                    self.build(overrides=dict(self.overrides, **override))

    def test_requires_authorized_batch_lr_and_budget(self):
        for key, value in (("data.samples_per_gpu", 16), ("optimizer.lr", .01),
                           ("runner.max_epochs", 2)):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "one epoch"):
                    self.build(overrides=dict(self.overrides, **{key: value}))

    def test_rejects_unknown_dataset(self):
        with self.assertRaisesRegex(ValueError, "Unsupported B dataset"):
            self.build("SFUT")


if __name__ == "__main__":
    unittest.main()
