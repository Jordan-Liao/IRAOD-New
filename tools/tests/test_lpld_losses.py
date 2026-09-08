"""Focused deterministic CPU tests; no detector imports or rotated operators."""

import importlib.util
import math
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "lpld_losses",
    Path(__file__).resolve().parents[2] / "sfod/extensions/lpld_losses.py")
lpld = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lpld)


class CandidateMaskTest(unittest.TestCase):
    def test_inclusive_iou_background_thresholds_and_original_order(self):
        # Each pair isolates a threshold: exact boundary, then just outside it.
        logits = torch.tensor([
            [0.95, 0.05, 0.01], [0.95, 0.05, 0.01],
            [0.0095, 0.0005, 0.99], [0.00095, 0.00005, 0.999],
        ], dtype=torch.float64).log().requires_grad_()
        iou = torch.tensor([
            [0.1, 0.4], [0.1, 0.40001],
            [0.0, 0.2], [0.0, 0.2],
        ], dtype=torch.float64, requires_grad=True)
        self.assertEqual(logits[2].softmax(-1)[-1].item(),
                         0.99)
        mask = lpld.lpld_candidate_mask(logits, iou)
        self.assertEqual(mask.tolist(), [True, False, True, False])
        self.assertFalse(mask.requires_grad)

    def test_inclusive_foreground_threshold(self):
        # Float32 realizes the exact 0.9 comparison boundary after softmax.
        logits = torch.tensor([[0.9, 0.1, 0.01],
                               [0.899, 0.101, 0.01]]).log()
        self.assertEqual(logits[0, :-1].softmax(-1).max().item(),
                         torch.tensor(0.9).item())
        self.assertEqual(
            lpld.lpld_candidate_mask(logits, torch.zeros(2, 1)).tolist(),
            [True, False])

    def test_foreground_confidence_excludes_background(self):
        # Foreground confidence is 0.95, despite low all-class foreground mass.
        logits = torch.tensor([[0.019, 0.001, 0.98],
                               [0.01, 0.01, 0.98]], dtype=torch.float64).log()
        mask = lpld.lpld_candidate_mask(logits, torch.zeros(2, 1))
        self.assertEqual(mask.tolist(), [True, False])

    def test_no_hpl_and_no_proposals(self):
        logits = torch.tensor([[10., 0., -10.], [0., 10., -10.]])
        self.assertEqual(
            lpld.lpld_candidate_mask(logits, torch.empty(2, 0)).tolist(),
            [False, False])
        for hpl_count in (0, 2):
            mask = lpld.lpld_candidate_mask(
                torch.empty(0, 3), torch.empty(0, hpl_count))
            self.assertEqual(mask.shape, (0,))
            self.assertEqual(mask.dtype, torch.bool)


class DistillationLossTest(unittest.TestCase):
    def test_exact_kl_detached_weights_and_kept_count_normalization(self):
        teacher = torch.tensor([[9., 1., 90.], [1., 9., 40.], [3., 1., 1.]],
                               dtype=torch.float64).log().requires_grad_()
        student_probs = [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3], [0.2, 0.7, 0.1]]
        student = torch.tensor(student_probs, dtype=torch.float64).log()
        student.requires_grad_()
        teacher_features = torch.tensor([[1., 0.], [1., 0.], [1., 0.]],
                                        dtype=torch.float64, requires_grad=True)
        student_features = torch.tensor([[0., 1.], [-1., 0.], [0., 1.]],
                                        dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([True, True, False])
        loss = lpld.lpld_loss(student, teacher, student_features,
                              teacher_features, mask)
        targets = [[0.9, 0.1, 1e-10], [0.1, 0.9, 1e-10]]
        weights = [1., 2.]
        expected = sum(
            weight * sum(t * math.log(t / p) for t, p in zip(target, probs))
            for weight, target, probs in zip(weights, targets, student_probs)
        ) / 2 / 10
        self.assertEqual(loss.ndim, 0)
        self.assertAlmostEqual(loss.item(), expected, delta=1e-14)
        loss.backward()
        # A target with total mass 1+1e-10 has gradient mass*p - target.
        expected_grad = torch.zeros_like(student)
        for index, (weight, target) in enumerate(zip(weights, targets)):
            expected_grad[index] = weight / 2 / 10 * (
                (1.0 + 1e-10) * torch.tensor(student_probs[index],
                                            dtype=torch.float64)
                - torch.tensor(target, dtype=torch.float64))
        torch.testing.assert_close(student.grad, expected_grad, rtol=0, atol=1e-14)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(teacher_features.grad)
        self.assertIsNone(student_features.grad)

    def test_background_target_mass_is_not_removed_or_renormalized(self):
        # Uniform student logits and a single foreground class make every term
        # analytic, including the small nonzero background contribution.
        student = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
        teacher = torch.tensor([[0., 20.]], dtype=torch.float64,
                               requires_grad=True)
        loss = lpld.lpld_loss(
            student, teacher, torch.tensor([[0., 1.]]),
            torch.tensor([[1., 0.]]), torch.tensor([True]))
        expected = (math.log(2) + 1e-10 * math.log(2e-10)) / 10
        self.assertAlmostEqual(loss.item(), expected, delta=1e-15)
        loss.backward()
        torch.testing.assert_close(
            student.grad,
            torch.tensor([[(-0.5 + 0.5e-10) / 10,
                           (0.5 - 0.5e-10) / 10]], dtype=torch.float64),
            rtol=0, atol=1e-15)
        self.assertIsNone(teacher.grad)

    def test_zero_weight_proposal_still_counts_in_denominator(self):
        student = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
        teacher = torch.zeros_like(student)
        loss = lpld.lpld_loss(
            student, teacher, torch.tensor([[1., 0.], [0., 1.]]),
            torch.tensor([[1., 0.], [1., 0.]]), torch.tensor([True, True]))
        expected = (math.log(2) + 1e-10 * math.log(2e-10)) / 2 / 10
        self.assertAlmostEqual(loss.item(), expected, delta=1e-15)
        loss.backward()
        torch.testing.assert_close(student.grad[0], torch.zeros(2, dtype=student.dtype))
        self.assertGreater(student.grad[1].abs().sum().item(), 0)

    def test_empty_selections_are_graph_connected_student_zeros(self):
        for reason in ("no_hpl", "zero_kept", "no_proposals"):
            with self.subTest(reason=reason):
                count = 0 if reason == "no_proposals" else 2
                student = torch.zeros(count, 3, dtype=torch.float64,
                                      requires_grad=True)
                teacher = torch.tensor([[10., 0., -10.]], dtype=torch.float64)
                teacher = teacher.repeat(count, 1).requires_grad_()
                student_features = torch.ones(count, 2, requires_grad=True)
                teacher_features = torch.ones(count, 2, requires_grad=True)
                iou = (torch.empty(count, 0) if reason == "no_hpl"
                       else torch.ones(count, 1))
                mask = lpld.lpld_candidate_mask(teacher, iou)
                loss = lpld.lpld_loss(student, teacher, student_features,
                                      teacher_features, mask)
                self.assertEqual(loss.ndim, 0)
                self.assertEqual(loss.item(), 0)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                torch.testing.assert_close(student.grad, torch.zeros_like(student))
                self.assertIsNone(teacher.grad)
                self.assertIsNone(student_features.grad)
                self.assertIsNone(teacher_features.grad)


class DecodedBoxSelectionTest(unittest.TestCase):
    def test_class_specific_selection_and_background_fallback(self):
        proposals = torch.arange(15, dtype=torch.float64).reshape(3, 5)
        proposals.requires_grad_()
        decoded = (100 + torch.arange(30, dtype=torch.float64)).reshape(3, 10)
        decoded.requires_grad_()
        logits = torch.tensor([[0., 4., 1.], [0., 1., 4.], [4., 0., 1.]],
                              requires_grad=True)
        selected = lpld.select_teacher_decoded_boxes(proposals, decoded, logits)
        expected = torch.stack((decoded[0, 5:], proposals[1], decoded[2, :5]))
        torch.testing.assert_close(selected, expected, rtol=0, atol=0)
        self.assertFalse(selected.requires_grad)

    def test_class_agnostic_selection_and_background_fallback(self):
        proposals = torch.arange(15, dtype=torch.float64).reshape(3, 5)
        decoded = proposals + 100
        logits = torch.tensor([[0., 4., 1.], [0., 1., 4.], [4., 0., 1.]])
        selected = lpld.select_teacher_decoded_boxes(proposals, decoded, logits)
        torch.testing.assert_close(
            selected, torch.stack((decoded[0], proposals[1], decoded[2])),
            rtol=0, atol=0)
        torch.testing.assert_close(decoded, proposals + 100, rtol=0, atol=0)

    def test_empty_decoded_inputs(self):
        for width in (5, 10):
            with self.subTest(width=width):
                selected = lpld.select_teacher_decoded_boxes(
                    torch.empty(0, 5), torch.empty(0, width), torch.empty(0, 3))
                self.assertEqual(selected.shape, (0, 5))


if __name__ == "__main__":
    unittest.main()
