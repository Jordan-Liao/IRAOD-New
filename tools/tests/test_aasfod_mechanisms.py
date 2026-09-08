"""Bounded CPU tests: real TSD, GRL, both discriminators, rotated FNS and EMA."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from experiments.comparison.aasfod_protocol import budget
from experiments.comparison.train_aasfod import stage_specs


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "sfod/extensions" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mechanisms = load("aasfod_mechanisms")
mosaic = load("aasfod_mosaic")


class AASFODMechanismTest(unittest.TestCase):
    def setUp(self):
        state, threads = torch.get_rng_state(), torch.get_num_threads()
        torch.manual_seed(13)
        torch.set_num_threads(1)
        self.addCleanup(torch.set_rng_state, state)
        self.addCleanup(torch.set_num_threads, threads)

    def test_variance_exact_twenty_sum_of_products_and_high_split(self):
        p, d = torch.rand(20, 3, 7), torch.rand(20, 3, 5)
        vp = ((p - p.mean(0)) ** 2).mean(0).sum(-1)
        vd = ((d - d.mean(0)) ** 2).mean(0).sum(-1)
        torch.testing.assert_close(mechanisms.tsd_variance(p, d), (vp * vd).sum())
        self.assertNotAlmostEqual(float((vp * vd).sum()), float(vp.mean() * vd.mean()))
        with self.assertRaisesRegex(ValueError, "exactly20"):
            mechanisms.tsd_variance(p[:19], d[:19])
        scores = {f"image{i}": float(i) for i in range(11)}
        a, b = mechanisms.high_variance_split(scores)
        self.assertEqual(a, ["image9", "image10"])
        self.assertEqual(len(b), 9)
        self.assertEqual(mechanisms.high_variance_split({"one": 1})[0], [])

    def test_aligned_tsd_single_backbone_rpn_twenty_roi_and_hook_cleanup(self):
        class ROI(nn.Module):
            def __init__(self):
                super().__init__()
                self.bbox_head = nn.Module()
                self.bbox_head.shared_fcs = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
                self.cls, self.box = nn.Linear(8, 3), nn.Linear(8, 5)
                self.rois = []

            def _bbox_forward(self, features, rois):
                self.rois.append(rois)
                value = features
                for fc in self.bbox_head.shared_fcs:
                    value = fc(value).relu()
                return dict(cls_score=self.cls(value), bbox_pred=self.box(value))

        class Detector(nn.Module):
            def __init__(self):
                super().__init__()
                self.roi_head = ROI()
                self.backbone_calls = self.rpn_calls = 0
                self.rpn_head = SimpleNamespace(simple_test_rpn=self.rpn)

            def extract_feat(self, image):
                self.backbone_calls += 1
                return image

            def rpn(self, features, metas):
                self.rpn_calls += 1
                return [torch.ones(4, 6)]

        detector = Detector().requires_grad_(False)
        image = torch.ones(4, 8)
        result = mechanisms.aligned_tsd(detector, image, [{}], lambda p: p[0])
        self.assertGreater(float(result), 0)
        self.assertEqual((detector.backbone_calls, detector.rpn_calls), (1, 1))
        self.assertEqual(len(detector.roi_head.rois), 20)
        self.assertTrue(all(r is detector.roi_head.rois[0] for r in detector.roi_head.rois))
        self.assertFalse(detector.training)
        self.assertTrue(all(not fc._forward_hooks for fc in detector.roi_head.bbox_head.shared_fcs))
        first = detector.roi_head._bbox_forward(image, torch.ones(4, 5))
        second = detector.roi_head._bbox_forward(image, torch.ones(4, 5))
        torch.testing.assert_close(first["cls_score"], second["cls_score"])

    def test_grl_reverses_features_not_discriminator_and_exact_domain_scalars(self):
        value = torch.tensor([2.0], requires_grad=True)
        parameter = torch.tensor([3.0], requires_grad=True)
        (mechanisms.reverse_gradient(value) * parameter).sum().backward()
        self.assertEqual((value.grad.item(), parameter.grad.item()), (-3, 2))
        heads = mechanisms.TargetDiscriminators(4, 8).eval()
        shallow = torch.randn(2, 4, 8, 8, requires_grad=True)
        deep = torch.randn(2, 8, 16, 16, requires_grad=True)
        losses = [heads(shallow, deep, domain) for domain in (0, 1)]
        local = heads.local(shallow)
        torch.testing.assert_close(losses[0]["local"], 0.5 * local.square().mean())
        torch.testing.assert_close(losses[1]["local"], 0.5 * (1 - local).square().mean())
        logits = heads.global_classifier(heads.global_features(deep).mean((-2, -1)))
        p = logits.softmax(-1)[:, 0]
        torch.testing.assert_close(losses[0]["global"], -0.5 * ((1 - p) ** 3 * p.log()).mean())
        sum(v for row in losses for v in row.values()).backward()
        self.assertGreater(shallow.grad.abs().sum().item(), 0)
        self.assertGreater(deep.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None for p in heads.parameters()))

    def test_complete_teacher_reset_frozen_buffers_and_post_optimizer_ema(self):
        student = nn.BatchNorm1d(2)
        teacher = nn.BatchNorm1d(2).requires_grad_(False)
        student.weight.requires_grad_(False)
        student.running_mean.fill_(7)
        mechanisms.copy_detector(student, teacher)
        torch.testing.assert_close(teacher.running_mean, student.running_mean)
        with torch.no_grad():
            student.bias.fill_(10)
        torch.testing.assert_close(teacher.bias, torch.zeros(2))  # no step0 EMA
        mechanisms.ema_detector(student, teacher)
        torch.testing.assert_close(teacher.bias, torch.full((2,), 0.1))
        self.assertFalse(teacher.training)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))

    def test_rotated_crop_anisotropic_map_keeps_labels_and_canonical_geometry(self):
        boxes = torch.tensor([[20., 20., 12., 6., .6], [99., 99., 5., 4., .3]])
        labels = torch.tensor([3, 9])
        out, ids = mosaic.transform_boxes(
            boxes, labels, (0, 0, 40, 40), (20, 40), (0, 0, 40, 20), (0, 0))
        self.assertEqual(ids.tolist(), [3])
        self.assertEqual(tuple(out.shape), (1, 5))
        self.assertGreaterEqual(out[0, 2], out[0, 3])
        self.assertLess(abs(float(out[0, 4])), np.pi / 2)
        self.assertNotAlmostEqual(float(out[0, 4]), .6, places=2)
        torch.testing.assert_close(out[0, :2], torch.tensor([20., 10.]))

    def test_four_image_fns_composes_original_labels_and_clips_partial_objects(self):
        class FixedRNG:
            def randint(self, low, high):
                return (low + high) // 2

        images = torch.stack([torch.full((3, 32, 32), float(i)) for i in range(4)])
        metas = [dict(img_shape=(32, 32, 3)) for _ in range(4)]
        # Each original box crosses the mosaic cut: all four partial targets survive.
        boxes = [torch.tensor([[16., 16., 16., 8., .4]]) for _ in range(4)]
        labels = [torch.tensor([i]) for i in range(4)]
        image, meta, out, ids = mosaic.four_image_mosaic(
            images, metas, boxes, labels, rng=FixedRNG())
        self.assertEqual(ids.tolist(), [0, 1, 2, 3])
        self.assertEqual(tuple(out.shape), (4, 5))
        for i in range(4):
            self.assertEqual(image[0, 8 if i < 2 else 24, 8 if i % 2 == 0 else 24], i)
        self.assertEqual(meta["img_shape"], (32, 32, 3))
        self.assertTrue(torch.all(out[:, 2] < 16))

    def test_stage_budget_counts_actual_updates_reset_and_original_image_slots(self):
        for n, expected in ((5863, (184, 110, 74)), (8467, (265, 159, 106))):
            plan = budget(n)
            self.assertEqual(tuple(plan[k] for k in
                                   ("total_updates", "alignment_updates", "fns_updates")), expected)
            self.assertEqual(plan["ema_interval"], 1)
            cell = dict(aasfod_budget=plan, source_checkpoint="/source.pth")
            a, b = stage_specs(cell, "/work")
            self.assertEqual(b["load_from"], a["student_checkpoint"])
            self.assertEqual(b["teacher_initialization"], a["student_checkpoint"])
            self.assertNotEqual(b["teacher_initialization"], a["teacher_checkpoint"])
            self.assertEqual(a["samples_per_gpu"] * 2, 32)
            self.assertEqual(b["samples_per_gpu"] // 4, 8)
            self.assertEqual(b["warmup_iters"], 0)
            self.assertEqual(a["optimizer"]["lr"], .02)
            self.assertEqual(a["updates"] + b["updates"], expected[0])


if __name__ == "__main__":
    unittest.main()
