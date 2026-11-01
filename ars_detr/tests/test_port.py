"""Short CPU checks of the vendored ARS-DETR port.

Run from the repo root: python -m unittest ars_detr.tests.test_port -v
"""
import math
import os.path as osp
import unittest

import numpy as np
import torch
from mmcv import Config
from mmdet.core import bbox_cxcywh_to_xyxy
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

    def test_box_costs_decide_the_match(self):
        cfg = Config.fromfile(RSAR_CFG)
        assigner = build_assigner(cfg.model.train_cfg.assigner)
        coder = build_bbox_coder(cfg.model.bbox_head.angle_coder)
        h = w = 800
        factor = torch.tensor([w, h, w, h], dtype=torch.float32)
        # gts as the head holds them: unnormalized (cx, cy, w, h, a), le90
        gt = torch.tensor([[150., 130., 100., 60., 0.2],
                           [430., 400., 60., 200., -0.9],
                           [680., 650., 160., 100., 1.1]])
        gt_labels = torch.tensor([0, 3, 5])
        ratio = torch.max(gt[:, 2] / gt[:, 3], gt[:, 3] / gt[:, 2]).view(-1, 1)
        gt_angles = coder.encode(gt[:, 4:5], ratio)
        # Per gt: an exact copy, a copy shifted by 0.4 w and a 1.5x copy. The
        # shift leaves the upstream L1 term unchanged, so the IoU term must
        # pick the exact copy; equal cls/angle logits make those costs
        # constant per gt, so only the box costs decide.
        exact = gt[:, :4] / factor
        shifted = exact.clone()
        shifted[:, 0] += 0.4 * exact[:, 2]
        scaled = exact.clone()
        scaled[:, 2:] *= 1.5
        perm = torch.tensor([4, 0, 7, 2, 8, 5, 1, 6, 3])
        pred_cxcywh = torch.cat([shifted, scaled, exact])[perm]
        order = torch.argsort(perm)[6:]  # rows holding exact[0], [1], [2]
        cls = torch.zeros(9, 6)
        ang = torch.zeros(9, 180)
        # Same call as ARSDeformableDETRHead._get_target_single.
        pred_xyxy = bbox_cxcywh_to_xyxy(pred_cxcywh)
        gt_xyxy = bbox_cxcywh_to_xyxy(gt[:, :4])
        result = assigner.assign(pred_xyxy, cls, ang, gt_xyxy, gt_labels,
                                 gt_angles, dict(img_shape=(h, w, 3)), None,
                                 ratio)
        self.assertEqual(int((result.gt_inds > 0).sum()), 3)
        self.assertEqual(result.gt_inds[order].tolist(), [1, 2, 3])
        self.assertEqual(result.labels[order].tolist(), gt_labels.tolist())
        # Upstream semantics kept: BBoxL1Cost(box_format='xywh') compares the
        # xyxy predictions with the gt converted to cxcywh.
        reg = assigner.reg_cost(pred_xyxy, gt_xyxy / factor)
        want = 2.0 * torch.cdist(pred_xyxy, gt[:, :4] / factor, p=1)
        self.assertTrue(torch.allclose(reg, want, atol=1e-5))


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
