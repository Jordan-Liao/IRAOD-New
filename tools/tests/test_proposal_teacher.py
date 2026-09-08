"""CPU seams for proposal alignment, real pre-classifier capture and epoch EMA."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mmcv import Config
from mmcv import ConfigDict
from mmcv.runner import EpochBasedRunner, Hook
import torch
from torch import nn

from sfod.extensions.lpld import LPLDOBB
from sfod.extensions.irg import IRGOBB
from sfod.extensions.irg_losses import IRGLosses
from sfod.extensions.sfut import SFUTOBB
from sfod.extensions.proposal_teacher import (
    AfterOptimizerTeacherHook, EpochFinalTeacherHook, ProposalAlignedTeacher,
    map_shared_geometry, preclassifier_roi_forward)


ROOT = Path(__file__).resolve().parents[2]


class TinyROI(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.bbox_head = nn.Module()
        self.bbox_head.fc_cls = nn.Linear(4, 3)

    def _bbox_forward(self, features, rois):
        vectors = self.shared(features[0][rois[:, 0].long()]).relu()
        logits = self.bbox_head.fc_cls(vectors)
        return {'cls_score': logits, 'bbox_pred': vectors.new_zeros((len(rois), 5))}


class ProposalTeacherTest(unittest.TestCase):
    def test_full_forward_keeps_four_losses_full_pseudos_and_per_image_auxiliary_cap(self):
        class RPN(nn.Module):
            def simple_test_rpn(self, features, metas):
                p = features[0].new_ones((302, 6))
                p[:, -1] = torch.linspace(1, 0, 302)
                return [p.clone() for _ in metas]

            def forward_train(self, features, metas, boxes, **kwargs):
                value = features[0].square().mean()
                return {'loss_rpn_cls': value, 'loss_rpn_bbox': 2 * value}, self.simple_test_rpn(features, metas)

        class ROI(TinyROI):
            def simple_test(self, features, proposals, metas, rescale):
                self.pseudo_proposal_counts = [len(p) for p in proposals]
                return 'full-pseudo-results'

            def forward_train(self, features, metas, proposals, boxes, labels, **kwargs):
                value = features[0].square().mean()
                return {'loss_cls': 3 * value, 'loss_bbox': 4 * value}

        class Teacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(4, 4)
                self.rpn_head, self.roi_head = RPN(), ROI()

            def extract_feat(self, images):
                return (self.projection(images),)

        class Port(ProposalAlignedTeacher):
            def __init__(self):
                nn.Module.__init__(self)
                self.projection = nn.Linear(4, 4)
                self.rpn_head, self.roi_head = RPN(), ROI()
                self.ema_model = Teacher().requires_grad_(False)
                self.cur_iter = self.image_num = 0
                self.pseudo_num = torch.tensor([2.])
                self.weight_u = 1
                self.train_cfg = ConfigDict(rpn_proposal={})
                self.test_cfg = ConfigDict(rpn={})
                self.aux_counts = []

            def extract_feat(self, images):
                return (self.projection(images),)

            def create_pseudo_results(self, image, results, transforms, device):
                assert results == 'full-pseudo-results'
                return [torch.ones(1, 5) for _ in image], [torch.zeros(1, dtype=torch.long) for _ in image]

            def analysis(self):
                pass

            def update_ema_model(self, momentum):
                raise AssertionError('No iteration-start EMA is allowed')

            def proposal_loss(self, proposals, hpl, teacher_roi, student_roi, weak_meta, strong_meta):
                assert not teacher_roi['preclassifier'].requires_grad
                assert student_roi['preclassifier'].requires_grad
                self.aux_counts.append(len(proposals))
                return {'loss_aux': student_roi['cls_score'].square().mean()}

        model = Port()
        weak = torch.randn(2, 4)
        strong = torch.randn(2, 4)
        metas = [dict(ori_filename=f'{i}.png', flip=False) for i in range(2)]
        boxes = [torch.empty(0, 5) for _ in metas]
        labels = [torch.empty(0, dtype=torch.long) for _ in metas]
        losses = model.forward_train_semi(
            weak, metas, boxes, labels, weak, metas, boxes, labels, strong, metas, boxes, labels)
        self.assertEqual(model.ema_model.roi_head.pseudo_proposal_counts, [302, 302])
        self.assertEqual(model.aux_counts, [300, 300])
        for key in ('loss_rpn_cls_unlabeled', 'loss_rpn_bbox_unlabeled',
                    'loss_cls_unlabeled', 'loss_bbox_unlabeled'):
            self.assertGreater(losses[key].item(), 0)
        sum(value for key, value in losses.items() if 'loss' in key).backward()
        self.assertIsNotNone(model.roi_head.bbox_head.fc_cls.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.ema_model.parameters()))

    def test_geometry_mapping_preserves_order_angles_and_shared_flip(self):
        boxes = torch.tensor([[10., 20., 4., 6., .2], [4., 8., 2., 3., -.4]])
        weak = dict(ori_filename='same.png', flip=True, flip_direction='horizontal',
                    scale_factor=[.5, .5, .5, .5])
        strong = dict(ori_filename='same.png', flip=True, flip_direction='horizontal')
        mapped = map_shared_geometry(boxes, weak, strong)
        torch.testing.assert_close(mapped[:, :4], 2 * boxes[:, :4])
        torch.testing.assert_close(mapped[:, 4], boxes[:, 4])
        torch.testing.assert_close(boxes[0], torch.tensor([10., 20., 4., 6., .2]))
        with self.assertRaisesRegex(ValueError, 'shared geometric'):
            map_shared_geometry(boxes, weak, {**strong, 'flip': False})

    def test_capture_is_actual_attached_fc_input_and_splits_original_proposal_order(self):
        head = TinyROI()
        features = torch.randn(2, 4, requires_grad=True)
        proposals = [torch.zeros(3, 5), torch.zeros(2, 5)]
        result = preclassifier_roi_forward(head, (features,), proposals)
        expected = head.shared(features).relu()
        torch.testing.assert_close(result['preclassifier'],
                                   torch.cat((expected[0:1].expand(3, -1),
                                              expected[1:2].expand(2, -1))))
        result['cls_score'].sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertEqual(len(head.bbox_head.fc_cls._forward_pre_hooks), 0)

    def test_epoch_ema_runs_before_checkpoint_without_auxiliary_parameters(self):
        class Model(ProposalAlignedTeacher):
            epoch_teacher_momentum = .75

            def __init__(self):
                nn.Module.__init__(self)
                self.weight = nn.Parameter(torch.tensor([4.]))
                self.auxiliary = nn.Linear(1, 1)
                teacher = nn.Module()
                teacher.weight = nn.Parameter(torch.tensor([0.]), requires_grad=False)
                self.ema_model = teacher

        model = Model()
        runner = object.__new__(EpochBasedRunner)
        runner.model = model
        runner._hooks = []
        saved = []

        class Save(Hook):
            def after_train_epoch(self, runner):
                saved.append(runner.model.ema_model.weight.detach().clone())

        runner.register_hook(Save(), priority='NORMAL')
        runner.register_hook(EpochFinalTeacherHook(), priority='HIGH')
        runner.call_hook('before_run')
        torch.testing.assert_close(model.ema_model.weight, torch.tensor([0.]))
        runner.call_hook('after_train_epoch')
        torch.testing.assert_close(saved[0], torch.tensor([1.]))
        self.assertEqual(list(model.ema_model.state_dict()), ['weight'])
        self.assertFalse(model.ema_model.training)

    def test_lpld_empty_hpl_skips_decode_and_roi_matching(self):
        model = object.__new__(LPLDOBB)
        nn.Module.__init__(model)
        logits = torch.randn(3, 3, requires_grad=True)
        with patch('sfod.extensions.lpld.rbbox_overlaps',
                   side_effect=AssertionError('empty-HPL images must not be mined')):
            result = model.proposal_loss(torch.zeros(3, 5), torch.empty(0, 5), {},
                                         {'cls_score': logits}, {}, {})
        result['loss_lpld'].backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_configs_keep_geometry_four_losses_and_epoch_final_hook(self):
        for method, dataset, classes, size in [
                (m, ds, c, n) for m in ('lpld', 'irg', 'sfut')
                for ds, c, n in [('rsar', 6, 8467), ('dior', 20, 5863)]]:
            cfg = Config.fromfile(
                ROOT / f'configs/unbiased_teacher/sfod/extensions/{method}_{dataset}.py',
                import_custom_modules=False)
            self.assertEqual(cfg.model.type, method.upper() + 'OBB')
            self.assertTrue(cfg.model.cfg.strict_source_free)
            self.assertTrue(cfg.model.cfg.use_bbox_reg)
            self.assertEqual(cfg.model.roi_head.bbox_head.type, 'RotatedShared2FCBBoxHead')
            self.assertEqual(cfg.model.roi_head.bbox_head.num_classes, classes)
            self.assertEqual(cfg.data.train.unlabeled_epoch_size, size)
            self.assertEqual(cfg.data.samples_per_gpu, 32)
            self.assertEqual(cfg.optimizer.lr, 0.02)
            self.assertEqual(cfg.runner.type, 'SemiEpochBasedRunner')
            self.assertEqual(cfg.runner.max_epochs, 1)
            if method == 'sfut':
                hook = next(h for h in cfg.custom_hooks if h['type'] == 'AfterOptimizerTeacherHook')
                self.assertEqual(hook['priority'], 45)
                self.assertFalse(any(h['type'] == 'EpochFinalTeacherHook' for h in cfg.custom_hooks))
            else:
                hook = next(h for h in cfg.custom_hooks if h['type'] == 'EpochFinalTeacherHook')
                self.assertEqual(hook['priority'], 'HIGH')
            teacher = Config.fromfile(ROOT / cfg.model.ema_config, import_custom_modules=False)
            self.assertEqual(teacher.model.backbone.type, 'OrthoNet')
            self.assertEqual(teacher.model.roi_head.bbox_head.num_classes, classes)

    def test_irg_adapter_uses_distinct_classifier_owners_and_connected_empty_losses(self):
        model = object.__new__(IRGOBB)
        nn.Module.__init__(model)
        model.irg = IRGLosses(4)
        model.roi_head = TinyROI()
        teacher = nn.Module()
        teacher.roi_head = TinyROI().requires_grad_(False)
        model.ema_model = teacher
        xs = torch.randn(3, 4, requires_grad=True)
        xt = torch.randn(3, 4)
        student = dict(preclassifier=xs, cls_score=model.roi_head.bbox_head.fc_cls(xs))
        target = dict(preclassifier=xt, cls_score=teacher.roi_head.bbox_head.fc_cls(xt))
        with patch.object(model.irg, 'forward', wraps=model.irg.forward) as forward:
            result = model.proposal_loss(torch.zeros(3, 5), None, target, student, {}, {})
        self.assertIs(forward.call_args.args[-2], model.roi_head.bbox_head.fc_cls)
        self.assertIs(forward.call_args.args[-1], teacher.roi_head.bbox_head.fc_cls)
        self.assertEqual(result['irg_proposals'].item(), 3)
        sum(value for key, value in result.items() if 'loss' in key).backward()
        self.assertIsNotNone(xs.grad)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        model.zero_grad()
        empty = model.proposal_loss(
            torch.empty(0, 5), None, {}, {'cls_score': xs[:0]}, {}, {})
        sum(value for key, value in empty.items() if 'loss' in key).backward()
        self.assertTrue(all(p.grad is not None for p in model.irg.parameters()))

    def test_sfut_updates_after_optimizer_before_save_and_has_no_auxiliary_head(self):
        class Model(SFUTOBB):
            def __init__(self):
                nn.Module.__init__(self)
                self.weight = nn.Parameter(torch.tensor([0.], dtype=torch.double))
                teacher = nn.Module()
                teacher.weight = nn.Parameter(torch.tensor([0.], dtype=torch.double),
                                              requires_grad=False)
                self.ema_model = teacher

        class Optimizer(Hook):
            def after_train_iter(self, runner):
                with torch.no_grad():
                    runner.model.weight.add_(2)

        saved = []

        class Save(Hook):
            def after_train_iter(self, runner):
                saved.append(runner.model.ema_model.weight.item())

        model = Model()
        runner = object.__new__(EpochBasedRunner)
        runner.model, runner._hooks = model, []
        runner.register_hook(Save(), priority=50)
        runner.register_hook(AfterOptimizerTeacherHook(), priority=45)
        runner.register_hook(Optimizer(), priority=40)
        runner.call_hook('before_run')
        for _ in range(2):
            runner.call_hook('after_train_iter')
        self.assertAlmostEqual(saved[0], .0004 * 2, places=12)
        self.assertAlmostEqual(saved[1], .9996 * saved[0] + .0004 * 4, places=12)
        self.assertEqual(model.auxiliary_losses(None, None, None, None, None, None, None), {})
        self.assertEqual(list(model.state_dict()), ['weight'])


if __name__ == '__main__':
    unittest.main()
