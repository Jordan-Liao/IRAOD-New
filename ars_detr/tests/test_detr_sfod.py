"""Short CPU checks of the DETR-group SFOD wrappers (sfod/detr_sfod.py).

Run from the repo root: python -m unittest ars_detr.tests.test_detr_sfod -v
"""
import copy
import os
import types
import unittest

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmrotate.models import build_detector

import ars_detr  # noqa: F401
from sfod import detr_sfod
from sfod.pets_dynamics import pets_consensus
from sfod.rotated_dru import DRUUnbiasedTeacher

CFG = 'configs/ars_detr/sfod/detr_{}_rsar_strict.py'
REG_KEYS = ('loss_bbox', 'loss_piou', 'loss_angle', 'd0.loss_iou',
            'd0.loss_bbox', 'd0.loss_angle', 'd4.loss_iou', 'enc_loss_bbox',
            'enc_loss_piou', 'enc_loss_angle')


def build(method, **cfg_overrides):
    cfg = Config.fromfile(CFG.format(method))
    cfg.model.ema_ckpt = None
    cfg.model.cfg.update(cfg_overrides)
    torch.manual_seed(0)
    model = build_detector(cfg.model)
    model.init_weights()  # as train.py; load_from would overwrite it
    model.ema_model.load_state_dict(model.state_dict(), strict=False)
    if hasattr(model, 'static_teacher'):
        model.static_teacher.load_state_dict(model.state_dict(), strict=False)
    model.ema_model = MMDataParallel(model.ema_model)  # as train_detector
    return model


def metas(n, s, flip=False):
    return [dict(img_shape=(s, s, 3), pad_shape=(s, s, 3), ori_shape=(s, s, 3),
                 batch_input_shape=(s, s), filename=f'img{i}.png',
                 scale_factor=np.ones(4, dtype=np.float32), flip=flip,
                 flip_direction='horizontal' if flip else None)
            for i in range(n)]


PSEUDO_B = [torch.tensor([[40., 40., 30., 12., 0.3], [90., 80., 20., 40., -1.0]]),
            torch.tensor([[60., 50., 24., 24., 0.7]])]
PSEUDO_L = [torch.tensor([0, 2]), torch.tensor([5])]


def strict_batch(n=2, s_weak=128, s_strong=128):
    empty_b = [torch.zeros((0, 5)) for _ in range(n)]
    empty_l = [torch.zeros((0,), dtype=torch.long) for _ in range(n)]
    return dict(img=torch.randn(n, 3, s_weak, s_weak), img_metas=metas(n, s_weak),
                gt_bboxes=empty_b, gt_labels=empty_l,
                img_unlabeled_1=torch.randn(n, 3, s_strong, s_strong),
                img_metas_unlabeled_1=metas(n, s_strong),
                gt_bboxes_unlabeled_1=empty_b, gt_labels_unlabeled_1=empty_l)


class TestMeanTeacher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ['CGA_SCORER'] = 'none'
        cls.model = build('st')
        cls.model.train()

    def test_regression_losses_zeroed_on_pseudo_labels(self):
        raw = self.model.student_losses(torch.randn(2, 3, 128, 128), metas(2, 128),
                                        PSEUDO_B, PSEUDO_L, None)
        for key in REG_KEYS:
            self.assertIn(key, raw)
            self.assertGreater(float(raw[key]), 0)
        out = detr_sfod.weight_pseudo_label_losses(raw, 1.0)
        for key, value in out.items():
            if key.endswith('loss_cls'):
                self.assertEqual(float(value), float(raw[key]))
            else:
                self.assertEqual(float(value), 0.0, key)
        self.model.zero_grad()
        sum(v for k, v in out.items() if not k.endswith('loss_cls')).backward()
        self.assertTrue(all(p.grad is None or not p.grad.any()
                            for p in self.model.parameters()))

    def test_strict_forward_ema_and_gt_rejection(self):
        model = build('st', score_thr=0.0)  # random init: admit every query
        model.train()
        ema = model.ema_model.module
        key = 'bbox_head.cls_branches.5.weight'
        with torch.no_grad():
            model.state_dict()[key].add_(1.0)
        before = ema.state_dict()[key].clone()
        student = model.state_dict()[key].clone()
        losses = model(**strict_batch())
        expected = 0.998 * before + 0.002 * student
        self.assertTrue(torch.allclose(ema.state_dict()[key], expected))
        self.assertEqual(float(losses['pseudo_num']), 300.0)
        total = sum(v for k, v in losses.items() if 'loss' in k)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(float(losses[f'{k}_unlabeled']) == 0 for k in REG_KEYS))
        bad = strict_batch()
        bad['gt_bboxes'] = PSEUDO_B
        with self.assertRaises(ValueError):
            model(**bad)

    def test_empty_pseudo_labels(self):
        losses = self.model(**strict_batch())  # random init: scores << 0.7
        self.assertEqual(float(losses['pseudo_num']), 0.0)
        self.assertTrue(torch.isfinite(
            sum(v for k, v in losses.items() if 'loss' in k)))


def _scored_teacher(model):
    """Spread the last-layer class scores so some exceed 0.7."""
    teacher = model.ema_model.module
    with torch.no_grad():
        branch = teacher.bbox_head.cls_branches[5]
        branch.bias.copy_(torch.linspace(-3.0, 3.0, branch.bias.numel()))
    return teacher


class TestPETSOnDETR(unittest.TestCase):

    def test_exchange_ema_and_consensus(self):
        model = build('pets')
        model.train()
        model.set_pets_schedule(10)
        self.assertEqual(model.pets_exchange_iters, [2, 3, 4, 5, 6, 7, 8, 9])
        key = 'bbox_head.cls_branches.5.bias'
        with torch.no_grad():
            model.state_dict()[key].add_(1.0)
        student = model.state_dict()[key].clone()
        static = model.static_teacher.state_dict()[key].clone()
        ema_before = model.ema_model.module.state_dict()[key].clone()
        self.assertEqual(model.momentum, 0.9996)
        model.cur_iter = 2  # exchange boundary: swap before this step's EMA
        losses = model(**strict_batch())
        self.assertEqual(float(losses['pets_exchanges']), 1.0)
        self.assertTrue(torch.equal(model.state_dict()[key], static))
        self.assertTrue(torch.equal(model.static_teacher.state_dict()[key], student))
        self.assertEqual(model.momentum, 0.999)
        expected = 0.999 * ema_before + 0.001 * static
        self.assertTrue(torch.allclose(model.ema_model.module.state_dict()[key], expected))
        self.assertTrue(torch.isfinite(sum(v for k, v in losses.items() if 'loss' in k)))

        teacher = _scored_teacher(model)
        teacher.eval()
        with torch.no_grad():
            dets = teacher.simple_test(torch.randn(1, 3, 128, 128), metas(1, 128),
                                       rescale=True)[0]
        self.assertEqual(sum(len(r) for r in dets), 300)
        fused = pets_consensus(dets, copy.deepcopy(dets), score_thr=0.5)
        n_in = sum(int((r[:, 5] > 0.5).sum()) for r in dets)
        n_out = sum(len(r) for r in fused)
        self.assertGreater(n_in, 0)
        self.assertTrue(0 < n_out <= n_in)  # WBF merges the top-k duplicates
        self.assertTrue(all(r.shape[1] == 6 and (r[:, 5] > 0.5).all() for r in fused))


class _FakeScorer:
    """Deterministic per-box class probabilities (a stand-in for SARCLIP)."""

    def __init__(self, num_classes):
        self.w = np.random.RandomState(0).randn(4, num_classes)

    def __call__(self, filename, boxes, scores, labels):
        z = np.sin(np.asarray(boxes, np.float64) / 7.0) @ self.w * 3.0
        p = np.exp(z - z.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True), []


class TestCGAOnDETR(unittest.TestCase):

    def test_prefilter_is_exact_for_legacy_and_veto_soft(self):
        model = build('st')
        teacher = _scored_teacher(model)
        img, img_metas = torch.randn(2, 3, 128, 128), metas(2, 128)
        for mode in detr_sfod.SCORE_NON_INCREASING_CGA_MODES:
            os.environ.update(CGA_SCORER='sarclip', CGA_FILTER_MODE=mode)
            teacher.cga, teacher.exclude_ids = _FakeScorer(6), []
            teacher.cga_filter_mode = mode
            with torch.no_grad():
                full = teacher.refine_test(
                    teacher.simple_test(img, img_metas, rescale=True), img_metas)
            ref_b, ref_l = model.pseudo_labels(full, 'cpu')
            got_b, got_l = model.pseudo_labels(
                model.inference_unlabeled(img, img_metas), 'cpu')
            for rb, gb, rl, gl in zip(ref_b, got_b, ref_l, got_l):
                self.assertTrue(torch.equal(rb, gb) and torch.equal(rl, gl))
            raw_b, _ = model.pseudo_labels(
                teacher.simple_test(img, img_metas, rescale=True), 'cpu')
            self.assertLess(sum(map(len, ref_b)), sum(map(len, raw_b)), mode)
        os.environ['CGA_FILTER_MODE'] = 'consensus_boost'
        with self.assertRaises(ValueError):
            model.inference_unlabeled(img, img_metas)
        os.environ['CGA_SCORER'] = 'none'


class TestVLSTOnDETR(unittest.TestCase):

    def test_dims_positives_and_gradient_path(self):
        model = build('vlst')
        model.train()
        text_encoder = torch.nn.Linear(6, 512).requires_grad_(False)  # frozen VLM stand-in
        model.vlst_teacher.set_text_prototypes(text_encoder(torch.eye(6)))
        self.assertEqual(model.vlst_teacher.projection_head[0].in_features, 256)

        head = model.bbox_head
        outs = head(model.extract_feat(torch.randn(2, 3, 128, 128)), metas(2, 128))
        queries = torch.randn(2, head.num_query, 256)
        feats, labels = detr_sfod.last_layer_positive_queries(
            head, outs, queries, PSEUDO_B, PSEUDO_L, metas(2, 128))
        self.assertEqual(feats.shape, (3, 256))  # one query per pseudo GT
        self.assertEqual(sorted(labels.tolist()), [0, 2, 5])

        model.zero_grad()
        raw = model.student_losses(torch.randn(2, 3, 128, 128), metas(2, 128),
                                   PSEUDO_B, PSEUDO_L, None)
        out = detr_sfod.weight_pseudo_label_losses(raw, 1.0)
        self.assertGreater(float(out['loss_vlst_proto']), 0)
        out['loss_vlst_proto'].backward()
        grad = lambda m: any(p.grad is not None and p.grad.abs().sum() > 0
                             for p in m.parameters())
        self.assertTrue(grad(model.vlst_teacher.projection_head))
        self.assertTrue(grad(head.transformer.decoder))
        self.assertTrue(grad(model.backbone))
        self.assertFalse(grad(head.cls_branches[5]))  # not on the VLST path
        self.assertTrue(all(p.grad is None for p in text_encoder.parameters()))
        self.assertFalse(model.vlst_teacher.text_prototypes.requires_grad)
        self.assertNotIn('vlst_teacher.projection_head.0.weight',
                         model.ema_model.module.state_dict())

    def test_flip_restore(self):
        boxes = np.array([[10., 20., 30., 40.]])
        meta = dict(flip=True, flip_direction='diagonal', ori_shape=(100, 200, 3))
        np.testing.assert_allclose(detr_sfod.restore_unflipped_aabb(boxes, meta),
                                   [[169., 59., 189., 79.]])


class TestConfigEnv(unittest.TestCase):

    def test_method_cga_env_survives_base(self):
        for method, scorer, mode in (('st', 'none', None),
                                     ('clip_cga', 'clip', 'legacy'),
                                     ('sarclip_cga', 'sarclip', 'veto_soft'),
                                     ('vlst', 'none', None),
                                     ('vlst_cga', 'sarclip', 'veto_soft')):
            for key in [k for k in os.environ if k.startswith('CGA_')]:
                os.environ.pop(key)
            cfg = Config.fromfile(CFG.format(method))
            self.assertEqual(os.environ['CGA_SCORER'], scorer, method)
            self.assertEqual(os.environ.get('CGA_FILTER_MODE'), mode, method)
            self.assertEqual(cfg.model.type, {'vlst': 'DETRVLSTTeacher',
                                              'vlst_cga': 'DETRVLSTTeacher'}
                             .get(method, 'DETRMeanTeacher'))
        os.environ['CGA_SCORER'] = 'none'


class TestDRUGuard(unittest.TestCase):

    def test_raises_without_roi_bbox_head(self):
        detr_like = types.SimpleNamespace()
        with self.assertRaises(NotImplementedError):
            DRUUnbiasedTeacher._capture_student_cls(detr_like, lambda: {})


if __name__ == '__main__':
    unittest.main()
