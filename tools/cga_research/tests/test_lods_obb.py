"""LODS-OBB port checks (CPU): proposal correspondence, rotated IoU, rotated
RoI features, graph-alignment loss. Needs the iraod env (mmcv-full, mmrotate).

    PYTHONPATH=. python -m unittest tools.cga_research.tests.test_lods_obb -v
"""
from __future__ import annotations

import copy
import math
import types
import unittest

import torch
from mmcv.ops import RoIAlignRotated
from mmrotate.core import rbbox2roi, rbbox_overlaps
from mmrotate.models import build_head

from mmcv import ConfigDict

from sfod.rotated_lods import (LODSTeacher, gromov_wasserstein_loss,
                               instance_relation_prior, roi_cls_and_feat,
                               teacher_proposals, valid_rotated_boxes)

ROI_HEAD = dict(
    type='OrientedStandardRoIHead',
    bbox_roi_extractor=dict(
        type='RotatedSingleRoIExtractor',
        roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
        out_channels=8, featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='RotatedShared2FCBBoxHead', in_channels=8, fc_out_channels=16,
        roi_feat_size=7, num_classes=6,
        bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', angle_range='le90',
                        norm_factor=None, edge_swap=True, proj_xy=True,
                        target_means=(0., 0., 0., 0., 0.),
                        target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
        reg_class_agnostic=True))


def _brute_gw(a_s, a_t, T):
    """Definition: sum_ijkl L(As_ik, At_jl) T_ij T_kl, L(a,b)=a log a - a + b - a log b."""
    a = a_s[:, None, :, None]
    b = a_t[None, :, None, :]
    L = a * torch.log(a + 1e-7) - a + b - a * torch.log(b + 1e-7)
    return torch.einsum('ijkl,ij,kl->', L, T, T)


def _cos(f):
    f = f / f.norm(dim=1, keepdim=True)
    return f @ f.t()


class RotatedIoUTest(unittest.TestCase):
    def test_known_overlaps(self):
        sq = torch.tensor([[10., 10., 2., 2., 0.]])
        cases = [
            (sq, 1.0),
            (torch.tensor([[10., 10., 2., 2., math.pi / 2]]), 1.0),
            (torch.tensor([[11., 10., 2., 2., 0.]]), 1 / 3),
            (torch.tensor([[10., 10., 2., 2., math.pi / 4]]), math.sqrt(2) / 2),
        ]
        for box, expect in cases:
            self.assertAlmostEqual(float(rbbox_overlaps(sq, box)[0, 0]), expect, places=4)

    def test_degenerate_rejection(self):
        boxes = torch.tensor([[5., 5., 4., 2., 0.1, .9], [5., 5., 0., 2., 0., .9],
                              [5., 5., 4., -1., 0., .9], [float('nan'), 5., 4., 2., 0., .9]])
        self.assertEqual(valid_rotated_boxes(boxes).tolist(), [True, False, False, False])


class RotatedRoIFeatureTest(unittest.TestCase):
    def test_roi_align_rotated_follows_angle(self):
        H = W = 64
        ramp = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W).expand(1, 1, H, W).contiguous()
        op = RoIAlignRotated(7, spatial_scale=1.0, sampling_ratio=2, clockwise=True)
        cx, cy, w, h = 32.5, 30.5, 14., 7.
        out0 = op(ramp, torch.tensor([[0., cx, cy, w, h, 0.]]))[0, 0]
        cols = cx - 0.5 + (torch.arange(7) + 0.5 - 3.5) * w / 7  # aligned=True pixel model
        self.assertTrue(torch.allclose(out0, cols.expand(7, 7), atol=1e-3))
        out90 = op(ramp, torch.tensor([[0., cx, cy, w, h, math.pi / 2]]))[0, 0]
        # rotated by 90 deg the box width runs along image y: values vary by row only
        self.assertTrue(torch.allclose(out90, out90[:, :1].expand(7, 7), atol=1e-3))
        self.assertGreater(float(out90[:, 0].std()), 0.5)

    def test_extractor_head_on_rotated_rois(self):
        torch.manual_seed(0)
        head = build_head(ROI_HEAD)
        x = [torch.randn(2, 8, 128 // s, 128 // s) for s in (4, 8, 16, 32)]
        props = [torch.tensor([[40., 40., 30., 10., .3, .9], [80., 60., 12., 50., -1.2, .8]]),
                 torch.tensor([[64., 64., 100., 90., .7, .7]])]
        rois = rbbox2roi([p[:, :5] for p in props])
        cls, feat = roi_cls_and_feat(head, x, rois)
        self.assertEqual(tuple(cls.shape), (3, 7))   # 6 classes + background
        self.assertEqual(tuple(feat.shape), (3, 16))
        self.assertTrue((feat >= 0).all())           # post-ReLU fc7, as official
        ref_cls, _ = head.bbox_head(head.bbox_roi_extractor(x[:4], rois))
        self.assertTrue(torch.allclose(cls, ref_cls, atol=1e-6))


RPN_HEAD = dict(
    type='OrientedRPNHead', in_channels=8, feat_channels=8, version='le90',
    anchor_generator=dict(type='AnchorGenerator', scales=[8], ratios=[0.5, 1.0, 2.0],
                          strides=[4, 8, 16, 32, 64]),
    bbox_coder=dict(type='MidpointOffsetCoder', angle_range='le90',
                    target_means=[0.] * 6, target_stds=[1., 1., 1., 1., 0.5, 0.5]))
SHARED_RPN_TEST_CFG = dict(nms_pre=2000, max_per_img=2000,
                           nms=dict(type='nms', iou_threshold=0.8), min_bbox_size=0)


class ProposalCorrespondenceTest(unittest.TestCase):
    def test_teacher_uses_official_300_nms07(self):
        torch.manual_seed(4)
        rpn = build_head(dict(RPN_HEAD, test_cfg=ConfigDict(SHARED_RPN_TEST_CFG))).eval()
        x = [torch.randn(1, 8, 256 // s, 256 // s) for s in (4, 8, 16, 32, 64)]
        metas = [dict(img_shape=(256, 256, 3), scale_factor=1.0, pad_shape=(256, 256, 3))]
        cfg = ConfigDict(nms_pre=2000, max_per_img=300,
                         nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0)
        with torch.no_grad():
            lods = teacher_proposals(rpn, x, metas, cfg)[0]
            shared = rpn.simple_test_rpn(x, metas)[0]
        self.assertEqual(lods.size(0), 300)       # official post-NMS top-N
        self.assertGreater(shared.size(0), 300)   # shared detector test cfg untouched
        self.assertEqual(lods.size(1), 6)         # (cx, cy, w, h, a, score)

    def test_student_scores_the_teacher_rois(self):
        torch.manual_seed(1)
        head_t = build_head(ROI_HEAD).eval()
        head_s = copy.deepcopy(head_t)
        x = [torch.randn(2, 8, 160 // s, 160 // s) for s in (4, 8, 16, 32)]
        props = [torch.tensor([[50., 50., 40., 12., .4, .9], [10., 10., 0., 5., 0., .9],
                               [90., 30., 20., 60., -.9, .6]]),
                 torch.tensor([[70., 80., 33., 18., 1.1, .8]])]
        kept = [p[valid_rotated_boxes(p)] for p in props]
        rois = rbbox2roi([p[:, :5] for p in kept])
        self.assertEqual(rois[:, 0].tolist(), [0., 0., 1.])
        # identity correspondence: roi k of the student is proposal k of the teacher
        boxes = torch.cat([p[:, :5] for p in kept])
        self.assertTrue(torch.allclose(rbbox_overlaps(rois[:, 1:], boxes).diag(),
                                       torch.ones(3), atol=1e-4))
        with torch.no_grad():
            cls_t, feat_t = roi_cls_and_feat(head_t, x, rois)
        cls_s, feat_s = roi_cls_and_feat(head_s, x, rois)
        self.assertTrue(torch.allclose(cls_s, cls_t) and torch.allclose(feat_s, feat_t))

    def test_enhanced_view_keeps_size_and_geometry(self):
        norm = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        img = torch.randn(2, 3, 64, 64)
        img[1, :, 48:, :] = 0  # padding of a 48x64 image
        metas = [dict(img_shape=(64, 64, 3), img_norm_cfg=norm),
                 dict(img_shape=(48, 64, 3), img_norm_cfg=norm)]
        seen = []

        def add_style(content, style, alpha):
            seen.append(tuple(content.shape))
            return content  # identity SEM: enhance must then be the identity

        fake = types.SimpleNamespace(
            sem=types.SimpleNamespace(add_style=add_style), style_add_alpha=1.0,
            _random_style=lambda meta, h, w, device: torch.zeros(1, 3, h, w))
        out = LODSTeacher.enhance(fake, img, metas)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(seen, [(1, 3, 64, 64), (1, 3, 48, 64)])  # unpadded region only
        self.assertTrue(torch.allclose(out, img, atol=1e-4))


class GraphAlignmentLossTest(unittest.TestCase):
    def test_matches_gw_definition(self):
        torch.manual_seed(2)
        fs, ft = torch.rand(5, 12), torch.rand(5, 12)
        t = torch.rand(5, 5)
        for prior in (None, t):
            T = torch.eye(5) + (0.5 * prior if prior is not None else 0)
            ref = _brute_gw(_cos(fs), _cos(ft), T / T.sum())
            got = gromov_wasserstein_loss(fs, ft, prior, beta=0.5)
            self.assertAlmostEqual(float(got), float(ref), places=5)

    def test_zero_when_graphs_agree_and_has_gradient(self):
        torch.manual_seed(3)
        ft = torch.rand(6, 10)
        self.assertAlmostEqual(float(gromov_wasserstein_loss(ft.clone(), ft)), 0.0, places=5)
        fs = torch.rand(6, 10, requires_grad=True)
        loss = gromov_wasserstein_loss(fs, ft)
        loss.backward()
        self.assertGreater(float(loss), 0)
        self.assertTrue(torch.isfinite(fs.grad).all() and fs.grad.abs().sum() > 0)

    def test_instance_relation_prior(self):
        # 4 rois, classes 0..2 + background index 3
        logits = torch.tensor([[5., 0., 0., 0.], [4., 0., 0., 1.], [0., 6., 0., 0.], [0., 0., 0., 9.]])
        conf, labels = logits.softmax(1).max(1)
        t = instance_relation_prior(logits, labels, conf, thresh=0.8, bg_index=3)
        same = torch.zeros(4, 4, dtype=torch.bool)
        same[0, 0] = same[2, 2] = True       # roi 1 is below thresh -> background
        if float(conf[1]) >= 0.8:
            same[0, 1] = same[1, 0] = same[1, 1] = True
        self.assertTrue(((t != 0) == same).all())
        self.assertAlmostEqual(float(t[0, 0]), 1.0, places=5)  # cosine of a logit with itself


if __name__ == '__main__':
    unittest.main()
