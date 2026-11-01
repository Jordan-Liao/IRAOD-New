"""PETS-OBB: exchange timing/swap, Dynamic Teacher EMA, rotated consensus.

CPU only. Run from the repo root (configs use repo-relative paths):
    python -m unittest tools.cga_research.tests.test_pets_obb -v
"""
import math
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

import sfod  # noqa: F401  registers PETSUnbiasedTeacher
from sfod.pets_dynamics import (exchange_iterations, exchange_weights_,
                                pets_consensus)

RSAR_CFG = ('configs/unbiased_teacher/sfod/'
            'unbiased_teacher_oriented_rcnn_selftraining_pets_rsar_orthonet_strict.py')


class _Wrap(nn.Module):
    """Stands in for the MMDataParallel that train_detector puts on ema_model."""

    def __init__(self, module):
        super().__init__()
        self.module = module


def _empty(n=2):
    return [np.zeros((0, 6), np.float32) for _ in range(n)]


class ExchangeScheduleTest(unittest.TestCase):
    def test_rsar_265_iters_gets_official_8_exchanges(self):
        # Official C2F: 10 epochs, warm-up 2, period 1 -> swaps at epochs 2..9.
        self.assertEqual(exchange_iterations(265),
                         [53, 79, 106, 132, 159, 185, 212, 238])

    def test_proportion_holds_for_other_lengths(self):
        self.assertEqual(exchange_iterations(184), [36, 55, 73, 92, 110, 128, 147, 165])
        self.assertEqual(exchange_iterations(100, epochs=5, warmup_epochs=2), [40, 60, 80])


class ExchangeWeightsTest(unittest.TestCase):
    def test_two_way_in_place_swap_of_params_and_buffers(self):
        student = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        static = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        with torch.no_grad():
            for p in student.state_dict().values():
                p.fill_(1)
            for p in static.state_dict().values():
                p.fill_(3)
        weight = student[0].weight
        exchange_weights_(student, static)
        self.assertIs(student[0].weight, weight)  # optimizer keeps its tensor
        for key, value in student.state_dict().items():
            self.assertTrue(torch.all(value == 3), key)
            self.assertTrue(torch.all(static.state_dict()[key] == 1), key)
        exchange_weights_(student, static)
        self.assertTrue(torch.all(student[0].weight == 1))


class DetectorScheduleAndEMATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mmcv import Config
        from mmrotate.models import build_detector
        cfg = Config.fromfile(RSAR_CFG)
        cfg.model.ema_ckpt = None
        cls.model = build_detector(cfg.model)
        cls.model.ema_model = _Wrap(cls.model.ema_model)
        cls.key = next(k for k, v in cls.model.static_teacher.state_dict().items()
                       if v.is_floating_point())

    def _probe(self, module):
        sd = module.state_dict()
        return sd[self.key] if self.key in sd else sd['module.' + self.key]

    def test_exchange_then_ema_with_official_keep_rates(self):
        m = self.model
        m.set_pets_schedule(20)
        self.assertEqual(m.pets_exchange_iters, [4, 6, 8, 10, 12, 14, 16, 18])
        student, static, ema = (self._probe(m), self._probe(m.static_teacher),
                                self._probe(m.ema_model))
        with torch.no_grad():
            student.fill_(1.0)
            static.fill_(3.0)
            ema.fill_(5.0)

        m.cur_iter = 3
        m._pets_before_iter()
        self.assertEqual(m.pets_num_exchanges, 0)
        self.assertEqual(m.momentum, 0.9996)
        m.update_ema_model(m.momentum)
        expected = 0.9996 * 5.0 + 0.0004 * 1.0
        self.assertTrue(torch.allclose(self._probe(m.ema_model), torch.tensor(expected)))

        m.cur_iter = 4
        m._pets_before_iter()
        self.assertEqual(m.pets_num_exchanges, 1)
        self.assertEqual(m.momentum, 0.999)
        self.assertTrue(torch.all(self._probe(m) == 3.0))
        self.assertTrue(torch.all(self._probe(m.static_teacher) == 1.0))
        m.update_ema_model(m.momentum)
        expected = 0.999 * expected + 0.001 * 3.0
        self.assertTrue(torch.allclose(self._probe(m.ema_model), torch.tensor(expected)))

        m.cur_iter = 5
        m._pets_before_iter()
        self.assertEqual(m.pets_num_exchanges, 1)

    def test_training_without_schedule_fails(self):
        m = self.model
        saved = m.pets_exchange_iters
        m.pets_exchange_iters = None
        try:
            with self.assertRaises(RuntimeError):
                m._pets_before_iter()
        finally:
            m.pets_exchange_iters = saved

    def test_both_teachers_feed_the_fused_pseudo_labels_to_the_student(self):
        m = self.model
        m.set_pets_schedule(20)
        m.cur_iter = 0
        static, dynamic = _empty(6), _empty(6)
        static[0] = np.array([[50, 50, 20, 10, 0.1, 0.9],      # paired with dynamic
                              [300, 300, 20, 10, 0.0, 0.9]],   # static only
                             np.float32)
        dynamic[0] = np.array([[52, 50, 20, 10, 0.1, 0.6],     # paired with static
                               [500, 500, 20, 10, 0.0, 0.9]],  # dynamic only
                              np.float32)
        img = torch.zeros(1, 3, 32, 32)
        meta = [dict(scale_factor=np.ones(4, np.float32))]
        empty_boxes = [torch.zeros((0, 5))]
        empty_labels = [torch.zeros((0,), dtype=torch.long)]
        seen = {}

        def student_forward_train(img_, metas_, gt_bboxes, gt_labels, *a, **k):
            seen['img'], seen['boxes'], seen['labels'] = img_, gt_bboxes, gt_labels
            return {'loss_cls': torch.tensor(1.0)}

        with mock.patch.object(m.static_teacher, 'simple_test',
                               return_value=[static]) as st, \
                mock.patch.object(m.ema_model.module, 'simple_test',
                                  return_value=[dynamic]) as dt, \
                mock.patch.object(m, 'forward_train', side_effect=student_forward_train):
            strong = torch.ones(1, 3, 32, 32)
            m.forward_train_semi(img, meta, empty_boxes, empty_labels,
                                 img, meta, empty_boxes, empty_labels,
                                 strong, meta, empty_boxes, empty_labels)
        self.assertEqual(st.call_count, 1)
        self.assertEqual(dt.call_count, 1)
        self.assertIs(st.call_args.args[0], img)   # both teachers see the weak view
        self.assertIs(dt.call_args.args[0], img)
        self.assertIs(seen['img'], strong)         # the student trains on the strong view
        fused = pets_consensus(static, dynamic)[0]
        self.assertEqual(fused.shape, (1, 6))
        self.assertAlmostEqual(float(fused[0, 0]), (0.9 * 50 + 0.6 * 52) / 1.5, places=4)
        torch.testing.assert_close(seen['boxes'][0], torch.from_numpy(fused[:, :5]))
        self.assertEqual(seen['labels'][0].tolist(), [0])


class RotatedConsensusTest(unittest.TestCase):
    def test_filter_pair_and_fuse_like_official(self):
        static = _empty()
        static[0] = np.array([[50, 50, 20, 10, 0.1, 0.9],     # A, paired
                              [200, 200, 20, 10, 0.0, 0.8],   # B, paired with D
                              [50, 50, 20, 10, 0.1, 0.4]],    # below 0.5
                             np.float32)
        dynamic = _empty()
        dynamic[0] = np.array([[52, 50, 20, 10, 0.1, 0.6],    # A'
                               [400, 400, 20, 10, 0.0, 0.9]],  # no partner
                              np.float32)
        dynamic[1] = np.array([[200, 200, 20, 10, 0.0, 0.7]], np.float32)  # D
        out = pets_consensus(static, dynamic)
        # Pairing is class-agnostic; WBF then fuses per label (official).
        self.assertEqual(out[0].shape, (2, 6))
        self.assertEqual(out[1].shape, (1, 6))
        np.testing.assert_allclose(out[0][0], [(0.9 * 50 + 0.6 * 52) / 1.5, 50, 20, 10, 0.1, 0.9],
                                   rtol=1e-5)  # weighted box, max score
        np.testing.assert_allclose(out[0][1], [200, 200, 20, 10, 0.0, 0.8], rtol=1e-5)
        np.testing.assert_allclose(out[1][0], [200, 200, 20, 10, 0.0, 0.7], rtol=1e-5)

    def test_equivalent_obb_representations_fuse_to_same_box(self):
        static, dynamic = _empty(1), _empty(1)
        static[0] = np.array([[100, 100, 40, 10, 0.0, 0.9]], np.float32)
        dynamic[0] = np.array([[100, 100, 10, 40, -math.pi / 2, 0.9]], np.float32)
        out = pets_consensus(static, dynamic)
        np.testing.assert_allclose(out[0][0, :5], [100, 100, 40, 10, 0.0], atol=1e-4)

    def test_angle_wraps_across_le90_boundary(self):
        static, dynamic = _empty(1), _empty(1)
        static[0] = np.array([[100, 100, 40, 10, math.pi / 2 - 0.01, 0.8]], np.float32)
        dynamic[0] = np.array([[100, 100, 40, 10, -math.pi / 2 + 0.01, 0.8]], np.float32)
        box = pets_consensus(static, dynamic)[0][0]
        np.testing.assert_allclose(box[2:4], [40, 10], atol=1e-4)
        self.assertAlmostEqual(abs(box[4]), math.pi / 2, places=4)

    def test_no_pair_gives_no_pseudo_labels(self):
        static, dynamic = _empty(), _empty()
        static[0] = np.array([[50, 50, 20, 10, 0.0, 0.9]], np.float32)
        out = pets_consensus(static, dynamic)
        self.assertEqual([o.shape for o in out], [(0, 6), (0, 6)])


if __name__ == '__main__':
    unittest.main()
