"""Gating tests for the approved comparison matrix (DIOR-17 + DRU required)."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

from tools.dataset.generate_dior_corruptions import (
    ALREADY_COMPARED_DIOR_C,
    DIOR_C_CORRUPTIONS,
    remaining_dior_c_corruptions,
)
from experiments.comparison.execution_matrix import (
    B_DETECTOR_TYPE,
    DEFENSIVE_EXCLUDED,
    DRU_DETECTOR_TYPE,
    EXTERNAL_METHODS,
    MAIN_TABLE_METHODS,
    NEW_DIOR_SEED,
    REQUIRED_TEST_ROLE,
    STUDENT_REQUIRED_TEST,
    dru_is_method_b,
    method_is_defensive,
    method_is_required,
)


def _load_dru_dynamics():
    path = Path(__file__).resolve().parents[3] / "sfod" / "dru_dynamics.py"
    spec = importlib.util.spec_from_file_location("iraod_dru_dynamics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_dataset_stubs():
    def _mod(name):
        module = types.ModuleType(name)
        sys.modules[name] = module
        return module

    if "torch" not in sys.modules:
        torch = _mod("torch")
        utils = _mod("torch.utils")
        data = _mod("torch.utils.data")

        class Dataset:
            def __init__(self):
                pass

        data.Dataset = Dataset
        utils.data = data
        torch.utils = utils

    if "numpy" not in sys.modules:
        numpy = _mod("numpy")

        def zeros(shape, dtype=None):
            n = shape[0] if isinstance(shape, tuple) else shape
            return [0] * int(n)

        numpy.zeros = zeros
        numpy.uint8 = 0
        numpy.float32 = 0.0
        numpy.int64 = 0

    if "mmcv" not in sys.modules:
        mmcv = _mod("mmcv")
        mmcv_utils = _mod("mmcv.utils")

        def build_from_cfg(transform, registry=None):
            return transform

        mmcv_utils.build_from_cfg = build_from_cfg
        mmcv.utils = mmcv_utils

    if "mmrotate" not in sys.modules:
        mmrotate = _mod("mmrotate")
        datasets = _mod("mmrotate.datasets")
        builder = _mod("mmrotate.datasets.builder")
        dota = _mod("mmrotate.datasets.dota")

        class _Registry:
            def register_module(self, name=None, force=False):
                def decorator(cls):
                    return cls

                return decorator

        builder.ROTATED_DATASETS = _Registry()
        builder.ROTATED_PIPELINES = _Registry()

        class DOTADataset:
            pass

        dota.DOTADataset = DOTADataset
        datasets.DOTADataset = DOTADataset
        datasets.builder = builder
        datasets.dota = dota
        mmrotate.datasets = datasets


def _load_strict_dataset_module(path: Path):
    _install_dataset_stubs()
    spec = importlib.util.spec_from_file_location("iraod_semi_dota_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ComparisonExecutionMatrixTests(unittest.TestCase):
    def test_remaining_seventeen_imported_from_generator(self):
        remaining = remaining_dior_c_corruptions()
        self.assertEqual(len(DIOR_C_CORRUPTIONS), 19)
        self.assertEqual(len(remaining), 17)
        self.assertEqual(
            remaining,
            tuple(name for name in DIOR_C_CORRUPTIONS if name not in ALREADY_COMPARED_DIOR_C),
        )
        self.assertNotIn("brightness", remaining)
        self.assertNotIn("contrast", remaining)
        self.assertNotIn("cloudy", remaining)
        self.assertNotIn("cloudy", DIOR_C_CORRUPTIONS)

    def test_external_six_including_dru_are_required(self):
        for method_id in ("IRG", "LPLD", "SFUT", "DRU", "AASFOD", "SFYOLO"):
            self.assertIn(method_id, EXTERNAL_METHODS)
            self.assertIn(method_id, MAIN_TABLE_METHODS)
            self.assertTrue(method_is_required(method_id))

    def test_dru_is_not_method_b(self):
        self.assertNotEqual("DRU", "B")
        self.assertNotEqual(DRU_DETECTOR_TYPE, B_DETECTOR_TYPE)
        self.assertEqual(DRU_DETECTOR_TYPE, "DRUUnbiasedTeacher")
        self.assertEqual(B_DETECTOR_TYPE, "UnbiasedTeacher")
        self.assertFalse(dru_is_method_b())

    def test_defensive_queue_is_out(self):
        self.assertFalse(STUDENT_REQUIRED_TEST)
        self.assertEqual(REQUIRED_TEST_ROLE, "ema")
        self.assertEqual(NEW_DIOR_SEED, 42)
        for name in (
            "Student-required-TEST",
            "LoRA-CGA-oracle",
            "B_REG",
            "F_text_only",
            "F_veto_only",
        ):
            self.assertIn(name, DEFENSIVE_EXCLUDED)
            self.assertTrue(method_is_defensive(name))
            self.assertFalse(method_is_required(name))
        self.assertNotIn("DRU", DEFENSIVE_EXCLUDED)

    def test_dru_dynamics_are_not_fixed_ema(self):
        dru = _load_dru_dynamics()
        base = 0.998
        better = dru.adaptive_ema_momentum(base, student_loss=0.4, reference_loss=0.8)
        worse = dru.adaptive_ema_momentum(base, student_loss=1.2, reference_loss=0.8)
        self.assertLess(better, base)
        self.assertGreater(worse, base)
        self.assertTrue(dru.should_retrain_student(2.0, 1.0, ratio=1.25))
        self.assertFalse(dru.should_retrain_student(1.0, 1.0, ratio=1.25))
        mse = dru.historical_student_mse([1.0, 0.0], [0.0, 1.0])
        self.assertGreater(mse, 0.0)
        self.assertEqual(dru.historical_student_mse([1.0], [1.0]), 0.0)

        class _Vec:
            def __init__(self, xs):
                self.xs = list(xs)
                self.shape = (len(self.xs),)

            def detach(self):
                return _Vec(self.xs)

            def __sub__(self, other):
                return _Vec([a - b for a, b in zip(self.xs, other.xs)])

            def __pow__(self, power):
                return _Vec([x ** power for x in self.xs])

            def mean(self):
                return sum(self.xs) / len(self.xs)

        live = _Vec([1.0, 0.0])
        stored = _Vec([0.0, 1.0])
        tensor_mse = dru.historical_student_mse(live, stored)
        self.assertGreater(tensor_mse, 0.0)
        self.assertEqual(dru.historical_student_mse(_Vec([0.5, 0.5]), _Vec([0.5, 0.5])), 0.0)

        repo = Path(__file__).resolve().parents[3]
        cfg = repo / (
            "configs/unbiased_teacher/sfod/"
            "unbiased_teacher_oriented_rcnn_selftraining_dru_rsar_orthonet_strict.py"
        )
        text = cfg.read_text(encoding="utf-8")
        self.assertIn("type='DRUUnbiasedTeacher'", text)
        self.assertNotIn("this is method B", text.lower())
        detector = (repo / "sfod" / "rotated_dru.py").read_text(encoding="utf-8")
        self.assertIn('losses["loss_dru_historical"]', detector)
        self.assertIn("_historical_prediction_loss", detector)

    def test_strict_source_free_dataset_is_importable(self):
        repo = Path(__file__).resolve().parents[3]
        path = repo / "sfod" / "semi_dota_dataset.py"
        module = _load_strict_dataset_module(path)
        self.assertTrue(hasattr(module, "StrictSourceFreeDOTADataset"))
        self.assertTrue(callable(getattr(module, "_discover_image_filenames")))

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "a.jpg").write_bytes(b"x")
            (Path(directory) / "b.png").write_bytes(b"y")
            names = module._discover_image_filenames(directory)
            self.assertEqual(names, ["a.jpg", "b.png"])
            dataset = module.StrictSourceFreeDOTADataset(
                img_prefix=directory,
                pipeline_share=[],
                pipeline_weak=[],
                pipeline_strong=[],
                classes=("ship",),
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.CLASSES, ("ship",))

        init_text = (repo / "sfod" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn("StrictSourceFreeDOTADataset", init_text)
        gen = (
            repo / "tools/dataset/generate_dior_corruptions.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def patch_numpy2_aliases", gen)
        self.assertIn("patch_numpy2_aliases()", gen)
        self.assertIn('"float_": "float64"', gen)

        for cfg_name in (
            "unbiased_teacher_oriented_rcnn_selftraining_st_baseline_rsar_orthonet_strict.py",
            "unbiased_teacher_oriented_rcnn_selftraining_st_baseline_dior_orthonet_strict.py",
        ):
            cfg = (
                repo
                / "configs/unbiased_teacher/sfod"
                / cfg_name
            ).read_text(encoding="utf-8")
            self.assertIn("type='StrictSourceFreeDOTADataset'", cfg)


if __name__ == "__main__":
    unittest.main()
