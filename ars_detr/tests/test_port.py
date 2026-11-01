"""Short CPU checks of the vendored ARS-DETR port.

Run from the repo root: python -m unittest ars_detr.tests.test_port -v
"""
import math
import os.path as osp
import unittest

import numpy as np
import torch
from mmcv import Config
from mmrotate.core.bbox.builder import build_assigner, build_bbox_coder
from mmrotate.models import build_detector

import ars_detr  # noqa: F401  (registers the modules)

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
RSAR_CFG = osp.join(REPO, 'configs/ars_detr/arcsl_detr_r50_rsar.py')


class TestConfigsBuild(unittest.TestCase):

    def test_rsar_and_dior_build(self):
        for name, n_cls in (('arcsl_detr_r50_rsar.py', 6),
                            ('arcsl_detr_r50_dior.py', 20)):
            cfg = Config.fromfile(osp.join(REPO, 'configs/ars_detr', name))
            model = build_detector(cfg.model)
            self.assertEqual(type(model).__name__, 'ARSDETR')
            self.assertEqual(model.bbox_head.num_classes, n_cls)
            self.assertEqual(model.bbox_head.angle_coder.window,
                             'aspect_ratio')


class TestARCSLCoder(unittest.TestCase):

    def test_encode_decode_round_trip(self):
        coder = build_bbox_coder(
            dict(type='ARCSLCoder', angle_version='le90', omega=1,
                 window='aspect_ratio', radius=6, normalize=True))
        self.assertEqual(coder.coding_len, 180)
        angles = torch.tensor([[-1.5], [-0.7], [0.0], [0.3], [1.5]])
        ratios = torch.tensor([[1.2], [2.0], [3.0], [5.0], [8.0]])
        code = coder.encode(angles, ratios)
        self.assertEqual(tuple(code.shape), (5, 180))
        decoded = coder.decode(code.unsqueeze(0)).view(-1, 1)
        # 1-degree bins decode to the bin centre.
        self.assertTrue(torch.all((decoded - angles).abs() <= math.radians(1)))


class TestARSHungarianAssigner(unittest.TestCase):

    def test_one_to_one_on_exact_predictions(self):
        cfg = Config.fromfile(RSAR_CFG)
        assigner = build_assigner(cfg.model.train_cfg.assigner)
        coder = build_bbox_coder(cfg.model.bbox_head.angle_coder)
        h = w = 800
        gt_xyxy = torch.tensor([[100., 100., 200., 160.],
                                [400., 300., 460., 500.],
                                [600., 600., 760., 700.]])
        gt_labels = torch.tensor([0, 3, 5])
        gt_theta = torch.tensor([[0.2], [-0.9], [1.1]])
        ratio = torch.tensor([[100 / 60], [200 / 60], [160 / 100]])
        gt_angles = coder.encode(gt_theta, ratio)
        # 10 queries: queries 7, 2, 4 reproduce gts 0, 1, 2; the rest are far.
        order = torch.tensor([7, 2, 4])
        pred = torch.rand(10, 4) * 0.05
        pred[:, 2:] += 0.05
        pred[order] = gt_xyxy / torch.tensor([w, h, w, h])
        cls = torch.full((10, 6), -8.)
        cls[order, gt_labels] = 8.
        ang = torch.full((10, 180), -8.)
        ang[order] = (gt_angles * 16 - 8)
        result = assigner.assign(pred, cls, ang, gt_xyxy, gt_labels, gt_angles,
                                 dict(img_shape=(h, w, 3)), None, ratio)
        pos = torch.nonzero(result.gt_inds > 0).view(-1)
        self.assertEqual(len(pos), 3)
        self.assertEqual(result.gt_inds[order].tolist(), [1, 2, 3])
        self.assertEqual(result.labels[order].tolist(), gt_labels.tolist())


class TestARSDETRHead(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cfg = Config.fromfile(RSAR_CFG)
        cfg.model.backbone.init_cfg = None  # no download in a unit test
        torch.manual_seed(0)
        cls.model = build_detector(cfg.model)
        cls.model.init_weights()
        cls.num_classes = cfg.model.bbox_head.num_classes

    def _metas(self, s):
        return [dict(img_shape=(s, s, 3), pad_shape=(s, s, 3),
                     ori_shape=(s, s, 3), batch_input_shape=(s, s),
                     scale_factor=np.ones(4, dtype=np.float32), flip=False)]

    def test_train_losses(self):
        s = 128
        self.model.train()
        img = torch.randn(1, 3, s, s)
        gt = [torch.tensor([[40., 40., 30., 12., 0.3],
                            [90., 80., 20., 40., -1.0]])]
        losses = self.model.forward_train(img, self._metas(s), gt,
                                          [torch.tensor([0, 2])])
        for key in ('loss_cls', 'loss_bbox', 'loss_piou', 'loss_angle',
                    'd0.loss_iou', 'd0.loss_angle', 'enc_loss_cls',
                    'enc_loss_piou', 'enc_loss_angle'):
            self.assertIn(key, losses)
        total = sum(v for k, v in losses.items() if 'loss' in k)
        self.assertTrue(torch.isfinite(total))

    def test_simple_test_format(self):
        s = 128
        self.model.eval()
        with torch.no_grad():
            out = self.model.simple_test(torch.randn(1, 3, s, s),
                                         self._metas(s), rescale=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(len(out[0]), self.num_classes)
        dets = np.concatenate(out[0])
        # top-k without NMS: num_query rows of (cx, cy, w, h, a, score)
        self.assertEqual(dets.shape, (300, 6))
        self.assertTrue(np.all(np.abs(dets[:, 4]) <= math.pi / 2 + 1e-6))
        self.assertTrue(np.all((dets[:, 5] >= 0) & (dets[:, 5] <= 1)))


if __name__ == '__main__':
    unittest.main()
