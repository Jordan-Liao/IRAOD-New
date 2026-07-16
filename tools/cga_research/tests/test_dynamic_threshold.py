from __future__ import annotations

from collections import deque
import unittest

import numpy as np
import torch

from sfod.rotated_unbiased_teacher import UnbiasedTeacher


def make_teacher(
    *,
    enabled: bool = True,
    quantile: float = 0.10,
    momentum: float = 0.90,
    minimum: float = 0.90,
    maximum: float = 0.95,
    queue_size: int = 100,
    min_samples: int = 20,
    num_classes: int = 2,
) -> UnbiasedTeacher:
    teacher = UnbiasedTeacher.__new__(UnbiasedTeacher)
    teacher.num_classes = num_classes
    teacher.score_thr = 0.90
    teacher.dynamic_threshold_enabled = enabled
    teacher.dynamic_threshold_quantile = quantile
    teacher.dynamic_threshold_momentum = momentum
    teacher.dynamic_threshold_min = minimum
    teacher.dynamic_threshold_max = maximum
    teacher.dynamic_threshold_queue_size = queue_size
    teacher.dynamic_threshold_min_samples = min_samples
    teacher._validate_dynamic_threshold_config()
    teacher.dynamic_score_thresholds = np.full(
        num_classes, teacher.score_thr, dtype=np.float64)
    teacher.dynamic_threshold_targets = np.full(
        num_classes, teacher.score_thr, dtype=np.float64)
    teacher.dynamic_threshold_new_samples = np.zeros(
        num_classes, dtype=np.int64)
    teacher.dynamic_score_queues = [
        deque(maxlen=queue_size) for _ in range(num_classes)
    ]
    teacher.pseudo_num = np.zeros(num_classes)
    teacher.pseudo_num_tp = np.zeros(num_classes)
    teacher.pseudo_num_gt = np.zeros(num_classes)
    return teacher


def detections(*class_scores: list[float]) -> list[np.ndarray]:
    rows = []
    for scores in class_scores:
        values = np.zeros((len(scores), 6), dtype=np.float32)
        if scores:
            values[:, -1] = np.asarray(scores, dtype=np.float32)
        rows.append(values)
    return rows


class DynamicThresholdTests(unittest.TestCase):
    def test_disabled_keeps_fixed_threshold_and_queue_empty(self) -> None:
        teacher = make_teacher(enabled=False, min_samples=1)
        teacher._update_dynamic_score_thresholds([
            detections([0.91, 0.99], [0.93])])
        self.assertEqual([len(q) for q in teacher.dynamic_score_queues], [0, 0])
        self.assertEqual(teacher._pseudo_score_threshold(0), 0.90)

    def test_min_samples_and_class_independence(self) -> None:
        teacher = make_teacher(momentum=0.0, min_samples=3)
        teacher._update_dynamic_score_thresholds([
            detections([0.91, 0.92], [0.99, 0.99, 0.99])])
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[0], 0.90)
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[1], 0.95)

    def test_class_without_new_samples_does_not_drift(self) -> None:
        teacher = make_teacher(
            quantile=0.5, momentum=0.5, min_samples=2)
        teacher._update_dynamic_score_thresholds([
            detections([0.92, 0.94], [0.91, 0.93])])
        before = teacher.dynamic_score_thresholds.copy()
        teacher._update_dynamic_score_thresholds([
            detections([], [0.94, 0.96])])
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[0], before[0])
        self.assertEqual(teacher.dynamic_threshold_new_samples[0], 0)

    def test_final_ema_respects_hard_minimum(self) -> None:
        teacher = make_teacher(
            quantile=0.5,
            momentum=0.9,
            minimum=0.93,
            maximum=0.95,
            min_samples=2,
        )
        teacher._update_dynamic_score_thresholds([
            detections([0.94, 0.95], [0.94, 0.95])])
        np.testing.assert_allclose(
            teacher.dynamic_score_thresholds, [0.93, 0.93])

    def test_ema_quantile_and_clamps(self) -> None:
        teacher = make_teacher(
            quantile=0.5, momentum=0.5, min_samples=2)
        teacher._update_dynamic_score_thresholds([
            detections([0.92, 0.94], [0.99, 1.00])])
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[0], 0.915)
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[1], 0.925)

    def test_queue_is_bounded_and_ignores_low_or_nonfinite_scores(self) -> None:
        teacher = make_teacher(
            momentum=0.0, queue_size=3, min_samples=1)
        teacher._update_dynamic_score_thresholds([
            detections([0.50, np.nan, 0.91, 0.92, 0.93, 0.94], [])])
        np.testing.assert_allclose(
            list(teacher.dynamic_score_queues[0]),
            [0.92, 0.93, 0.94],
        )

    def test_actual_pseudo_filter_uses_updated_class_threshold(self) -> None:
        teacher = make_teacher(
            quantile=0.5, momentum=0.0, min_samples=2)
        batch = [detections([0.91, 0.93], [0.94, 0.96])]
        image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        boxes, labels = teacher.create_pseudo_results(
            image, batch, [], torch.device('cpu'))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(labels[0].tolist(), [0, 1])
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[0], 0.92)
        self.assertAlmostEqual(teacher.dynamic_score_thresholds[1], 0.95)

    def test_disabled_actual_filter_matches_fixed_score_threshold(self) -> None:
        teacher = make_teacher(enabled=False, min_samples=1)
        batch = [detections([0.899, 0.90, 0.95], [])]
        image = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        boxes, labels = teacher.create_pseudo_results(
            image, batch, [], torch.device('cpu'))
        self.assertEqual(len(boxes[0]), 2)
        self.assertEqual(labels[0].tolist(), [0, 0])

    def test_updates_are_deterministic(self) -> None:
        first = make_teacher(quantile=0.25, momentum=0.7, min_samples=2)
        second = make_teacher(quantile=0.25, momentum=0.7, min_samples=2)
        batches = [
            [detections([0.91, 0.96], [0.92, 0.99])],
            [detections([0.94, 0.98], [0.93, 0.97])],
        ]
        for batch in batches:
            first._update_dynamic_score_thresholds(batch)
            second._update_dynamic_score_thresholds(batch)
        np.testing.assert_allclose(
            first.dynamic_score_thresholds,
            second.dynamic_score_thresholds,
        )

    def test_invalid_parameters_are_rejected(self) -> None:
        invalid = (
            {'quantile': -0.1},
            {'momentum': 1.0},
            {'minimum': 0.89},
            {'minimum': 0.96, 'maximum': 0.95},
            {'queue_size': 0},
            {'queue_size': 2, 'min_samples': 3},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    make_teacher(**values)


if __name__ == '__main__':
    unittest.main()
