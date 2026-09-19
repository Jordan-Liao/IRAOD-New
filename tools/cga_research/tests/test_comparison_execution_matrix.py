"""Gating tests for the approved comparison matrix (DIOR-17 + DRU required)."""
from __future__ import annotations

import importlib.util
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
        mse = dru.historical_student_consistency([1.0, 0.0], [0.0, 1.0])
        self.assertGreater(mse, 0.0)
        self.assertEqual(dru.historical_student_consistency([1.0], [1.0]), 0.0)
        cfg = Path(__file__).resolve().parents[3] / (
            "configs/unbiased_teacher/sfod/"
            "unbiased_teacher_oriented_rcnn_selftraining_dru_rsar_orthonet_strict.py"
        )
        text = cfg.read_text(encoding="utf-8")
        self.assertIn("type='DRUUnbiasedTeacher'", text)
        self.assertNotIn("this is method B", text.lower())


if __name__ == "__main__":
    unittest.main()
