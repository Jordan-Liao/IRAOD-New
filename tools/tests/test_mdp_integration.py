"""CPU MDP integration with native rotated RPN, ROI heads, losses and RoIAlign."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config, ConfigDict
from mmcv.ops import RoIAlignRotated
from mmcv.runner import load_checkpoint
from mmrotate.models.builder import build_head

from sfod.extensions.mdp import MDPOBB
from sfod.extensions.mdp_losses import rotated_class_prototypes
from sfod.extensions.proposal_teacher import EpochFinalTeacherHook
from sfod.semi_dota_dataset import StrictSourceFreeDOTADataset

ROOT = Path(__file__).resolve().parents[2]


def detector_settings():
    train = ConfigDict(
        rpn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=.7, neg_iou_thr=.3,
                              min_pos_iou=.3, match_low_quality=True, ignore_iof_thr=-1),
                 sampler=dict(type='RandomSampler', num=32, pos_fraction=.5,
                              neg_pos_ub=-1, add_gt_as_proposals=False),
                 allowed_border=0, pos_weight=-1, debug=False),
        rpn_proposal=dict(nms_pre=40, max_per_img=16, nms=dict(type='nms', iou_threshold=.8),
                          min_bbox_size=0),
        rcnn=dict(assigner=dict(type='MaxIoUAssigner', pos_iou_thr=.5, neg_iou_thr=.5,
                               min_pos_iou=.5, match_low_quality=False, ignore_iof_thr=-1,
                               iou_calculator=dict(type='RBboxOverlaps2D')),
                  sampler=dict(type='RRandomSampler', num=8, pos_fraction=.5,
                               neg_pos_ub=-1, add_gt_as_proposals=True),
                  pos_weight=-1, debug=False))
    test = ConfigDict(
        rpn=deepcopy(train.rpn_proposal),
        rcnn=dict(nms_pre=40, min_bbox_size=0, score_thr=.05,
                  nms=dict(iou_thr=.1), max_per_img=16))
    return train, test


class TinyBackbone(nn.Module):
    res_layers = ('layer1', 'layer2', 'layer3', 'layer4')

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                  nn.ReLU(), nn.AvgPool2d(2))
        for index, name in enumerate(self.res_layers):
            setattr(self, name, nn.Sequential(
                nn.Conv2d(16, 16, 3, stride=1 if index == 0 else 2, padding=1),
                nn.ReLU()))

    def _tap_channels(self, index):
        return 16

    def forward(self, images):
        value = self.stem(images)
        features = []
        for name in self.res_layers:
            value = getattr(self, name)(value)
            features.append(value)
        return tuple(features) + (F.max_pool2d(value, 1, stride=2),)


def native_heads(train, test):
    rpn = build_head(dict(
        type='OrientedRPNHead', in_channels=16, feat_channels=16, version='le90',
        anchor_generator=dict(type='AnchorGenerator', scales=[2],
                              ratios=[.5, 1., 2.], strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(type='MidpointOffsetCoder', angle_range='le90',
                        target_means=[0.] * 6, target_stds=[1., 1., 1., 1., .5, .5]),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.),
        loss_bbox=dict(type='SmoothL1Loss', beta=1 / 9, loss_weight=1.),
        train_cfg=train.rpn, test_cfg=test.rpn))
    roi = build_head(dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
            out_channels=16, featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead', in_channels=16, fc_out_channels=16,
            roi_feat_size=7, num_classes=2, reg_class_agnostic=True,
            bbox_coder=dict(type='DeltaXYWHAOBBoxCoder', angle_range='le90',
                            norm_factor=None, edge_swap=True, proj_xy=True,
                            target_means=(0.,) * 5, target_stds=(.1, .1, .2, .2, .1)),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.),
            loss_bbox=dict(type='SmoothL1Loss', beta=1., loss_weight=1.)),
        train_cfg=train.rcnn, test_cfg=test.rcnn))
    with torch.no_grad():
        roi.bbox_head.fc_cls.weight.zero_()
        roi.bbox_head.fc_cls.bias.copy_(torch.tensor([4., 0., 0.]))
        roi.bbox_head.fc_reg.weight.normal_(0, .02)
        roi.bbox_head.fc_reg.bias.zero_()
    return rpn, roi


class TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.train_cfg, self.test_cfg = detector_settings()
        self.rpn_head, self.roi_head = native_heads(self.train_cfg, self.test_cfg)

    def extract_feat(self, images):
        return self.backbone(images)

    def simple_test(self, images, metas, rescale=False):
        features = self.extract_feat(images)
        proposals = self.rpn_head.simple_test_rpn(features, metas)
        return self.roi_head.simple_test(features, proposals, metas, rescale=rescale)


class TinyMDP(MDPOBB):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = TinyBackbone()
        self.train_cfg, self.test_cfg = detector_settings()
        self.rpn_head, self.roi_head = native_heads(self.train_cfg, self.test_cfg)
        self.ema_model = TinyDetector().requires_grad_(False).eval()
        self.num_classes = 2
        self.score_thr = .7
        self.weight_u = 1.
        self.strict_source_free = self.use_bbox_reg = True
        self.cur_iter = self.image_num = 0
        self.pseudo_num = np.zeros(2)
        self.pseudo_sem_weight = np.zeros(2)
        self._mdp_teacher_loaded = self._mdp_weights_loaded = False
        self._initialize_mdp()
        # The C2=16 fixture has one hidden unit; exercise an active gradient path.
        with torch.no_grad():
            self.mdp.afsp.style[0].weight.fill_(.02)

    def extract_feat(self, images):
        return self.backbone(images)

    def analysis(self):
        pass


def meta(name, size, scale=1.):
    return dict(filename=name, ori_filename=name, img_shape=(size, size, 3),
                ori_shape=(64, 64, 3), pad_shape=(size, size, 3),
                scale_factor=np.array([scale] * 4, dtype=np.float32),
                flip=False, flip_direction=None)


class MDPIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        np.random.seed(83)
        random.seed(83)

    def initialized_model(self):
        model = TinyMDP()
        source = {key: value.clone() for key, value in model.state_dict().items()
                  if not key.startswith('mdp.')}
        with tempfile.TemporaryDirectory(prefix='mdp-source-cpu-') as directory:
            checkpoint = Path(directory) / 'source.pth'
            torch.save({'state_dict': source, 'meta': {'epoch': 100}}, checkpoint)
            load_checkpoint(model, str(checkpoint), map_location='cpu', strict=False)
        return model, source

    def test_real_four_losses_two_branches_prototypes_and_gradients(self):
        model, _ = self.initialized_model()
        strong = torch.randn(2, 3, 64, 64)
        weak = F.interpolate(strong, size=(128, 128), mode='bilinear', align_corners=False)
        weak_meta = [meta(f'{i}.png', 128, 2.) for i in range(2)]
        strong_meta = [meta(f'{i}.png', 64) for i in range(2)]
        boxes = [torch.empty(0, 5) for _ in range(2)]
        labels = [torch.empty(0, dtype=torch.long) for _ in range(2)]
        teacher_before = {key: value.clone() for key, value in model.ema_model.state_dict().items()}
        activations = []
        hook = model.mdp.afsp.style[1].register_forward_hook(
            lambda _module, _args, output: activations.append(output.detach()))
        try:
            with patch.object(model.mdp.afsp, 'forward', wraps=model.mdp.afsp.forward) as afsp:
                losses = model.forward_train_semi(
                    weak, weak_meta, boxes, labels, weak, weak_meta, boxes, labels,
                    strong, strong_meta, boxes, labels)
        finally:
            hook.remove()
        self.assertEqual(afsp.call_count, 1)
        self.assertGreater(activations[0].abs().sum().item(), 0)
        for branch in ('msp', 'afsp'):
            for name in ('loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox'):
                self.assertIn(f'mdp_{branch}_{name}', losses)
                self.assertTrue(torch.isfinite(losses[f'mdp_{branch}_{name}']))
        self.assertGreater(losses['pseudo_num'].item(), 0)
        self.assertGreater(losses['loss_mdp_pfd'].item(), 0)
        self.assertGreater(losses['mdp_pfd_classes'].item(), 0)
        sum(value for key, value in losses.items() if 'loss' in key).backward()
        for name, module in (
                ('C4', model.backbone.layer3), ('RPN', model.rpn_head),
                ('ROI', model.roi_head.bbox_head), ('AFSP', model.mdp.afsp),
                ('transformation', model.mdp.transform)):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0, name)
        self.assertTrue(all(parameter.grad is None for parameter in model.ema_model.parameters()))
        for key, value in teacher_before.items():
            torch.testing.assert_close(model.ema_model.state_dict()[key], value)
        self.assertEqual(len(model.backbone.layer1._forward_hooks), 0)
        self.assertEqual(len(model.roi_head.bbox_head._forward_hooks), 0)
        with self.assertRaisesRegex(ValueError, 'nonempty target ground truth'):
            model.forward_train_semi(
                weak, weak_meta, [torch.ones(1, 5), boxes[1]], labels,
                weak, weak_meta, boxes, labels, strong, strong_meta, boxes, labels)

    def test_checkpoint_loading_does_not_hide_missing_detector_weights(self):
        model, source = self.initialized_model()
        for key, value in source.items():
            torch.testing.assert_close(model.state_dict()[key], value)
            torch.testing.assert_close(model.ema_model.state_dict()[key], value)
        partial = dict(source)
        partial.pop(next(iter(source)))
        with self.assertRaisesRegex(RuntimeError, 'MDP checkpoint mismatch'):
            model.load_state_dict(partial, strict=False)
        different = {key: value.clone() for key, value in source.items()}
        key = next(iter(different))
        different[key] += 1
        with self.assertRaisesRegex(RuntimeError, 'source weights differ'):
            model.load_state_dict(different, strict=False)
        target = TinyMDP()
        target.load_state_dict(model.state_dict())
        self.assertFalse(target._mdp_weights_loaded)

    def test_native_rotated_pooling_retains_angle_and_chunked_mean(self):
        y, x = torch.meshgrid(torch.arange(8.), torch.arange(8.), indexing='ij')
        features = torch.stack([x.square(), y.square()] * 8)[None].requires_grad_()
        pool = RoIAlignRotated(7, 1., 2, True, True)
        rois = torch.tensor([[0., 3.5, 3.5, 4., 2., 0.],
                             [0., 3.5, 3.5, 4., 2., np.pi / 2]])
        logits = torch.tensor([[4., 0.], [0., 4.]], requires_grad=True)
        small, present = rotated_class_prototypes(features, rois, logits, 2, pool, chunk_size=1)
        full, _ = rotated_class_prototypes(features, rois, logits, 2, pool)
        torch.testing.assert_close(small, full)
        self.assertTrue(present.all())
        self.assertFalse(torch.allclose(small[0], small[1]))
        small.square().sum().backward()
        self.assertGreater(features.grad.abs().sum().item(), 0)
        self.assertIsNone(logits.grad)

    def test_eval_bypasses_all_auxiliary_modules_and_epoch_hook_updates_only_detector(self):
        model, source = self.initialized_model()
        baseline = TinyDetector().eval()
        baseline.load_state_dict(source)
        model.eval()
        images = torch.randn(1, 3, 64, 64)
        metas = [meta('eval.png', 64)]
        with torch.no_grad(), patch.object(model.mdp.afsp, 'forward', side_effect=AssertionError), \
                patch.object(model.mdp.transform, 'forward', side_effect=AssertionError):
            actual = model.simple_test(images, metas, rescale=False)
            expected = baseline.simple_test(images, metas, rescale=False)
        for a, b in zip(actual[0], expected[0]):
            np.testing.assert_array_equal(a, b)
        teacher = model.ema_model
        key, parameter = next(iter(model.named_parameters()))
        before = teacher.state_dict()[key].clone()
        with torch.no_grad():
            parameter.add_(1)
        runner = type('Runner', (), {'model': model})()
        hook = EpochFinalTeacherHook()
        hook.before_run(runner)
        hook.after_train_iter(runner)
        torch.testing.assert_close(teacher.state_dict()[key], before)
        hook.after_train_epoch(runner)
        torch.testing.assert_close(teacher.state_dict()[key], before + .1)
        self.assertFalse(any(key.startswith('mdp.') for key in teacher.state_dict()))

    def test_source_free_loader_is_independent_of_target_annotations(self):
        with tempfile.TemporaryDirectory(prefix='mdp-image-only-') as directory:
            root = Path(directory)
            images = root / 'images'
            annotations = root / 'annfiles'
            images.mkdir()
            annotations.mkdir()
            cv2.imwrite(str(images / 'one.png'), np.full((16, 16, 3), 91, dtype=np.uint8))
            annotation = annotations / 'one.txt'
            annotation.write_text('target labels must not be read')
            dataset = StrictSourceFreeDOTADataset(
                str(images), [dict(type='LoadImageFromFile')], [], [])
            first = dataset[0]
            annotation.unlink()
            second = dataset[0]
            np.testing.assert_array_equal(first['img'], second['img'])
            self.assertEqual(first['gt_bboxes'].shape, (0, 5))
            self.assertEqual(first['gt_labels'].shape, (0,))

    def test_configs_keep_source_geometry_budget_and_epoch_ema(self):
        for dataset, classes, size in (('rsar', 6, 8467), ('dior', 20, 5863)):
            cfg = Config.fromfile(
                ROOT / f'configs/unbiased_teacher/sfod/extensions/mdp_{dataset}.py',
                import_custom_modules=False)
            self.assertEqual(cfg.model.type, 'MDPOBB')
            self.assertEqual(cfg.model.roi_head.bbox_head.num_classes, classes)
            self.assertTrue(cfg.model.cfg.strict_source_free)
            self.assertTrue(cfg.model.cfg.use_bbox_reg)
            self.assertEqual(cfg.model.cfg.score_thr, .7)
            self.assertEqual(cfg.data.samples_per_gpu, 32)
            self.assertEqual(cfg.data.train.unlabeled_epoch_size, size)
            self.assertEqual(cfg.runner.max_epochs, 1)
            self.assertEqual(cfg.optimizer.lr, .02)
            self.assertEqual(cfg.model.ema_ckpt, cfg.load_from)
            self.assertIn(dataset, Path(cfg.load_from).name)
            self.assertTrue(any(h['type'] == 'EpochFinalTeacherHook' for h in cfg.custom_hooks))
            self.assertFalse(any(h['type'] == 'AfterOptimizerTeacherHook' for h in cfg.custom_hooks))


if __name__ == '__main__':
    unittest.main()
