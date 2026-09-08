"""CPU-only execution tests for the two exact frozen-F deletions."""

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mmcv import Config

from experiments.comparison import f_deletion as deletion
from experiments.comparison.b_regression import config_diff


class FDeletionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import the actual scalar implementation without sfod's model registry.
        source = Path(__file__).resolve().parents[2] / "sfod/cga.py"
        spec = importlib.util.spec_from_file_location("f_deletion_cpu_cga", source)
        cls.cga = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cga)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="frozen F ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.code = self.root / "core 0f98"
        self.rsar = self.code / "configs/unbiased_teacher/sfod/F_strict.py"
        self.rsar.parent.mkdir(parents=True)
        (self.code / "train.py").write_text("# frozen entry, never executed\n")
        self.weights = self.root / "base SARCLIP.safetensors"
        self.weights.write_bytes(b"mock checkpoint, never loaded")
        self.rsar.write_text(
            "import os\n"
            "from torchvision.transforms import ColorJitter\n"
            "custom_imports = dict(imports=['must_not_import_gpu_models'])\n"
            "model = dict(type='UnbiasedTeacherVLST', ema_ckpt='rsar_source.pth',\n"
            "    cfg=dict(use_bbox_reg=False, strict_source_free=True,\n"
            "      weight_l=0., weight_u=1., score_thr=.7,\n"
            "      vlst_enabled=True, vlst_strict=True, vlst_text_visual_alpha=.5,\n"
            "      vlst_loss_weight=.1, vlst_temperature=.07,\n"
            "      vlst_prototype_momentum=.9, vlst_lora_path=None,\n"
            "      vlst_pretrained=os.environ['SARCLIP_PRETRAINED']))\n"
            "data = dict(samples_per_gpu=2, workers_per_gpu=2, train=dict(\n"
            "    type='StrictSourceFreeDOTADataset', img_prefix='rsar/val',\n"
            "    unlabeled_epoch_size=8467, unlabeled_subset_seed=42,\n"
            "    classes=('ship', 'aircraft'), pipeline_weak=[dict(type='Resize')],\n"
            "    pipeline_strong=[dict(type='Apply',\n"
            "      operations=[ColorJitter(.4, .4, .4, .1)])]))\n"
            "custom_hooks = [dict(type='NumClassCheckHook')]\n"
            "optimizer = dict(type='SGD', lr=.001, momentum=.9)\n"
            "runner = dict(type='SemiEpochBasedRunner', max_epochs=1)\n"
            "load_from = 'rsar_source.pth'\n"
            "checkpoint_config = dict(interval=1)\n"
            "os.environ['CGA_SCORER'] = 'sarclip'\n"
            "os.environ['CGA_BACKEND'] = 'sarclip'\n"
            "os.environ['CGA_STRICT'] = '1'\n"
            "os.environ['CGA_FILTER_MODE'] = 'veto_soft'\n"
            "os.environ['CGA_DROP_SCORE'] = '0.0'\n"
            "os.environ['CGA_FILTER_LOG_EVERY'] = '500'\n"
            "os.environ['CGA_VETO_PRED_THR'] = '0.7'\n"
            "os.environ['CGA_VETO_LABEL_THR'] = '0.1'\n"
            "os.environ['CGA_PROTECT_DET_SCORE'] = '0.9'\n"
            # An outer environment override cannot defeat this assignment.
            "os.environ['CGA_BLEND_DET_WEIGHT'] = '0.7'\n"
            "os.environ['VLST_BACKEND'] = 'sarclip'\n"
            "del ColorJitter\n"
        )
        self.dior = self.root / "remotequeue/configs/dior_F_strict.py"
        self.dior.parent.mkdir(parents=True)
        self.dior.write_text(
            f"_base_ = {str(self.rsar)!r}\n"
            "model = dict(ema_ckpt='dior_source.pth')\n"
            "load_from = 'dior_source.pth'\n"
            "data = dict(train=dict(img_prefix='dior/val', unlabeled_epoch_size=5863,\n"
            "                       classes=('airplane', 'airport')))\n"
        )
        self.paths = Mock()
        self.paths.rsar_cfg.return_value = str(self.rsar)
        self.paths.dior_cfg.return_value = str(self.dior)
        git_patch = patch.object(
            deletion.subprocess, "check_output",
            return_value=deletion.TRAINING_CODE_SHA + "\n")
        self.git = git_patch.start()
        self.addCleanup(git_patch.stop)
        env_patch = patch.dict(os.environ, {
            "CGA_BLEND_DET_WEIGHT": ".222",
            "CGA_VETO_PRED_THR": ".111",
            "CGA_FILTER_MODE": "legacy",
            "CGA_POISON": "bad",
            "SARCLIP_LORA": "bad-lora.pth",
            "SARCLIP_PRETRAINED": "bad-base.pth",
            "VLST_BACKEND": "bad",
            "HOME": "/caller/home",
            "CLIP_DOWNLOAD_ROOT": "/caller/cache",
            "CGA_CLIP_CACHE": "/caller/cache",
        })
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.overrides = {
            "data.samples_per_gpu": 16,
            "optimizer.lr": .02,
            "runner.max_epochs": 1,
            "find_unused_parameters": True,
            "data.train.img_prefix": "/image-only/val",
            "data.train.unlabeled_epoch_size": 32,
            "model.ema_ckpt": "/same/source.pth",
            "load_from": "/same/source.pth",
            "corrupt": "fog",
            "seed": 43,
            "work_dir": "/new/output",
            "checkpoint_config.max_keep_ckpts": 1,
            "checkpoint_config.save_last": True,
        }

    def build(self, dataset="RSAR", method="F_veto_only"):
        return deletion.build_f_deletion_spec(
            self.paths, dataset, method, self.weights, self.overrides)

    def test_executable_exact_deletions_and_actual_cga_scalars(self):
        before_env = dict(os.environ)
        before_bytes = {p: p.read_bytes() for p in (self.rsar, self.dior, self.weights)}
        overrides_before = deepcopy(self.overrides)
        real_fromfile = Config.fromfile
        for dataset in ("RSAR", "DIOR"):
            reference = self.rsar if dataset == "RSAR" else self.dior
            with deletion._f_environment(self.weights):
                baseline = real_fromfile(str(reference), import_custom_modules=False)
                original_env = deletion._model_environment()
            baseline.merge_from_dict(self.overrides)
            baseline_before = deepcopy(baseline)
            for method in ("F_text_only", "F_veto_only"):
                with self.subTest(dataset=dataset, method=method):
                    with patch.object(Config, "fromfile", wraps=real_fromfile) as load:
                        spec = self.build(dataset, method)
                    # Helper-owned validation files are deleted, custom imports disabled.
                    temp_loads = [
                        Path(call.args[0]) for call in load.call_args_list
                        if Path(call.args[0]) not in (self.rsar, self.dior)
                    ]
                    self.assertTrue(temp_loads)
                    self.assertTrue(all(not p.exists() for p in temp_loads))
                    self.assertTrue(all(
                        call.kwargs["import_custom_modules"] is False
                        for call in load.call_args_list))
                    self.assertEqual(spec["reference_config"], str(reference))
                    self.assertEqual(spec["training_code"], str(self.code))
                    self.assertEqual(spec["training_code_sha"], deletion.TRAINING_CODE_SHA)
                    self.assertEqual(spec["topology"], {
                        "world_size": 2, "samples_per_gpu": 16, "global_batch_size": 32})
                    self.assertNotIn("_base_", spec["config_text"])
                    generated = self.root / "candidate.py"
                    generated.write_text(spec["config_text"])
                    with patch.dict(os.environ):
                        loaded = real_fromfile(str(generated), import_custom_modules=False)
                        effective = deletion._model_environment()
                        self.assertEqual(effective, spec["effective_model_environment"])
                        self.assertNotIn("SARCLIP_LORA", effective)
                        self.assertNotIn("CGA_POISON", effective)
                        self.assertEqual(effective["SARCLIP_PRETRAINED"], str(self.weights))
                        self.assertEqual(os.environ["HOME"], before_env["HOME"])
                        self.assertEqual(
                            os.environ["CLIP_DOWNLOAD_ROOT"], before_env["CLIP_DOWNLOAD_ROOT"])
                        mixin = self.cga.TestMixins()
                        with patch.object(self.cga, "CGA") as constructor, patch.object(
                                self.cga, "_log_cga_info"):
                            mixin._build_cga(6 if dataset == "RSAR" else 20)
                        constructor.assert_called_once()
                        self.assertEqual(
                            constructor.call_args.kwargs["pretrained"], str(self.weights))
                        self.assertTrue(constructor.call_args.kwargs["strict"])
                        self.assertEqual(mixin.cga_filter_mode, "veto_soft")
                        self.assertEqual(mixin.cga_veto_pred_thr, .7)
                        self.assertEqual(mixin.cga_veto_label_thr, .1)
                        self.assertEqual(mixin.cga_protect_det_score, .9)
                        self.assertEqual(mixin.cga_drop_score, 0.)
                        self.assertEqual(
                            mixin.cga_blend_detector_weight,
                            1. if method == "F_veto_only" else .7)
                    expected_cfg = ([{
                        "path": deletion.ALPHA_PATH, "before": .5, "after": 1.
                    }] if method == "F_text_only" else [])
                    expected_env = ([{
                        "path": deletion.BLEND_KEY, "before": "0.7", "after": "1.0"
                    }] if method == "F_veto_only" else [])
                    self.assertEqual(config_diff(baseline, loaded), expected_cfg)
                    self.assertEqual(spec["config_diff"], expected_cfg)
                    self.assertEqual(config_diff(original_env, effective), expected_env)
                    self.assertEqual(spec["environment_diff"], expected_env)
                    self.assertEqual(config_diff(baseline_before, baseline), [])
                    self.assertEqual(dict(os.environ), before_env)
                    self.assertEqual(self.overrides, overrides_before)
                    for path, content in before_bytes.items():
                        self.assertEqual(path.read_bytes(), content)
        self.paths.rsar_cfg.assert_called_with("F")
        self.paths.dior_cfg.assert_called_with("F")
        self.git.assert_called_with(
            ["git", "-C", str(self.code), "rev-parse", "HEAD"], text=True)

    def test_original_unconditional_assignment_defeats_outer_and_child_override(self):
        child = self.root / "old_inheritance.py"
        child.write_text(
            f"_base_ = {str(self.rsar)!r}\n"
            "import os\n"
            "os.environ['CGA_BLEND_DET_WEIGHT'] = '1.0'\n")
        with deletion._f_environment(self.weights):
            os.environ[deletion.BLEND_KEY] = "1.0"
            Config.fromfile(str(child), import_custom_modules=False)
            self.assertEqual(os.environ[deletion.BLEND_KEY], "0.7")

    def test_rejects_wrong_f_baselines_and_restores_caller(self):
        original = self.rsar.read_text()
        mutations = (
            ("type='UnbiasedTeacherVLST'", "type='UnbiasedTeacher'"),
            ("vlst_text_visual_alpha=.5", "vlst_text_visual_alpha=.8"),
            ("use_bbox_reg=False", "use_bbox_reg=True"),
            ("vlst_enabled=True", "vlst_enabled=False"),
            ("vlst_strict=True", "vlst_strict=False"),
            ("strict_source_free=True", "strict_source_free=False"),
            ("vlst_lora_path=None", "vlst_lora_path='lora.pth'"),
            ("vlst_pretrained=os.environ['SARCLIP_PRETRAINED']", "vlst_pretrained='other'"),
            ("['CGA_BLEND_DET_WEIGHT'] = '0.7'", "['CGA_BLEND_DET_WEIGHT'] = '1.0'"),
            ("['CGA_FILTER_MODE'] = 'veto_soft'", "['CGA_FILTER_MODE'] = 'legacy'"),
            ("['CGA_VETO_PRED_THR'] = '0.7'", "['CGA_VETO_PRED_THR'] = '0.9'"),
            ("['CGA_VETO_LABEL_THR'] = '0.1'", "['CGA_VETO_LABEL_THR'] = '0.2'"),
            ("['CGA_PROTECT_DET_SCORE'] = '0.9'", "['CGA_PROTECT_DET_SCORE'] = '0.8'"),
            ("['CGA_DROP_SCORE'] = '0.0'", "['CGA_DROP_SCORE'] = '0.1'"),
        )
        before_env = dict(os.environ)
        for old, new in mutations:
            with self.subTest(mutation=new):
                self.rsar.write_text(original.replace(old, new))
                with self.assertRaises(ValueError):
                    self.build()
                self.assertEqual(dict(os.environ), before_env)
        self.rsar.write_text(original + "\nos.environ['SARCLIP_LORA'] = 'bad'\n")
        with self.assertRaisesRegex(ValueError, "forbid"):
            self.build()
        self.rsar.write_text(original + "\nraise RuntimeError('broken source config')\n")
        with self.assertRaisesRegex(RuntimeError, "broken source"):
            self.build()
        self.assertEqual(dict(os.environ), before_env)

    def test_rejects_unapproved_controls_and_wrong_topology(self):
        for key, value in {
            "model.cfg.vlst_text_visual_alpha": 1.,
            "model.cfg.use_bbox_reg": True,
            "model.cfg.vlst_loss_weight": .2,
            "model.cfg.vlst_temperature": .2,
            "model.cfg.vlst_prototype_momentum": .8,
            "model.cfg.score_thr": .9,
            "model.cfg.vlst_lora_path": "lora.pth",
            "data.train.pipeline_weak": [],
            "optimizer.momentum": .8,
            "model": {"cfg": {"vlst_enabled": False}},
            "CGA_BLEND_DET_WEIGHT": "1.0",
            "data.samples_per_gpu": 32,
            "optimizer.lr": .01,
            "runner.max_epochs": 2,
        }.items():
            with self.subTest(key=key):
                with patch.dict(self.overrides, {key: value}):
                    with self.assertRaises(ValueError):
                        self.build()

    def test_rejects_wrong_source_sha_missing_inputs_and_unknown_method(self):
        self.git.return_value = "wrong SHA\n"
        with self.assertRaisesRegex(ValueError, "training code"):
            self.build()
        self.git.return_value = deletion.TRAINING_CODE_SHA
        (self.code / "train.py").unlink()
        with self.assertRaisesRegex(ValueError, "train.py"):
            self.build()
        self.weights.unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint does not exist"):
            self.build()
        with self.assertRaisesRegex(ValueError, "dataset"):
            self.build(dataset="other")
        with self.assertRaisesRegex(ValueError, "deletion"):
            self.build(method="F")

    def test_rejects_unknown_pretty_text_constructor(self):
        self.rsar.write_text(
            self.rsar.read_text()
            + "\nfrom tools.tests.test_b_regression import Transform\n"
            + "unsupported = Transform()\n"
            + "del Transform\n")
        before_env = dict(os.environ)
        with self.assertRaisesRegex(ValueError, "Unsupported constructor"):
            self.build()
        self.assertEqual(dict(os.environ), before_env)


if __name__ == "__main__":
    unittest.main()
