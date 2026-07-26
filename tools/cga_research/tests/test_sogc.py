import copy
import gc
import io
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from mmcv import Config
from mmdet.models.builder import BACKBONES
from mmrotate.models import build_detector

import mmdet_extension  # noqa: F401
import sfod  # noqa: F401
from mmdet_extension.models.utils import SOGCCalibrator
from sfod.rotated_semi_base import SemiBaseDetector


SOURCE_CONFIG = 'configs/baseline/oriented_rcnn_orthonet_rsar.py'
SOGC_CONFIG = 'configs/baseline/oriented_rcnn_sogc_orthonet_rsar.py'
SFOD_CONFIG = (
    'configs/unbiased_teacher/sfod/'
    'unbiased_teacher_oriented_rcnn_selftraining_sogc_rsar_orthonet.py')
EMA_CONFIG = (
    'configs/baseline/ema_config/'
    'baseline_oriented_rcnn_ema_rsar_sogc_orthonet.py')
PRODUCTION_CHECKPOINT = Path(
    '/myfile/pretrain/oriented_rcnn_orthonet_rsar_epoch_100.pth')


def _ready_calibrator(channels=8, mode='calibrate', gamma=0.1):
    calibrator = SOGCCalibrator(
        channels, mode='collect', gamma=gamma, reduction=2)
    calibrator(torch.randn(5, channels, 1, 1),
               torch.ones(5, channels, 1, 1))
    calibrator.finalize_source_stats()
    calibrator.set_sogc_mode(mode)
    return calibrator


class TestSOGCCalibrator(unittest.TestCase):

    def test_off_returns_original_attention(self):
        calibrator = SOGCCalibrator(8, mode='off')
        response = torch.randn(3, 8, 1, 1)
        original = torch.sigmoid(torch.randn_like(response))
        output = calibrator(response, original)
        self.assertIs(output, original)
        self.assertEqual(int(calibrator.source_count.item()), 0)

    def test_zero_initialized_calibration_is_exact_identity(self):
        calibrator = _ready_calibrator()
        response = torch.randn(3, 8, 1, 1)
        original = torch.sigmoid(torch.randn_like(response))
        output = calibrator(response, original)
        self.assertTrue(torch.equal(output, original))
        self.assertTrue(torch.count_nonzero(
            calibrator.mlp[-1].weight).item() == 0)
        self.assertTrue(torch.count_nonzero(
            calibrator.mlp[-1].bias).item() == 0)

    def test_batch_welford_matches_direct_population_statistics(self):
        torch.manual_seed(11)
        values = torch.randn(17, 6, 1, 1) * 2.0 + 0.4
        calibrator = SOGCCalibrator(6, mode='collect')
        original = torch.ones_like(values)
        offsets = (0, 3, 8, 10, 17)
        for start, end in zip(offsets[:-1], offsets[1:]):
            calibrator(values[start:end], original[start:end])
        calibrator.finalize_source_stats()
        self.assertEqual(int(calibrator.source_count.item()), len(values))
        self.assertTrue(torch.allclose(
            calibrator.source_mean,
            values.double().mean(dim=0, keepdim=True),
            atol=1e-12,
            rtol=1e-12))
        expected_var = values.double().var(
            dim=0, unbiased=False, keepdim=True).clamp_min(calibrator.eps)
        self.assertTrue(torch.allclose(
            calibrator.source_var, expected_var,
            atol=1e-12, rtol=1e-12))

    def test_checkpoint_roundtrip_preserves_source_statistics(self):
        calibrator = _ready_calibrator(channels=5)
        buffer = io.BytesIO()
        torch.save({'state_dict': calibrator.state_dict()}, buffer)
        buffer.seek(0)
        restored = SOGCCalibrator(5, mode='calibrate', reduction=2)
        restored.load_state_dict(torch.load(buffer)['state_dict'])
        for name in (
                'source_mean', 'source_var', 'source_count', 'source_m2',
                'stats_ready'):
            self.assertTrue(torch.equal(
                getattr(calibrator, name), getattr(restored, name)), name)

    def test_strict_calibrate_without_statistics_raises(self):
        calibrator = SOGCCalibrator(
            4, mode='calibrate', strict_stats=True)
        with self.assertRaisesRegex(
                RuntimeError, 'requires finalized source statistics'):
            calibrator(torch.randn(2, 4, 1, 1), torch.ones(2, 4, 1, 1))

    def test_nonzero_calibration_changes_output_is_bounded_and_has_gradient(self):
        calibrator = _ready_calibrator(channels=8, gamma=0.1)
        with torch.no_grad():
            calibrator.mlp[0].weight.fill_(0.01)
            calibrator.mlp[0].bias.fill_(0.1)
            calibrator.mlp[-1].weight.fill_(0.02)
            calibrator.mlp[-1].bias.fill_(0.2)
        response = torch.randn(4, 8, 1, 1, requires_grad=True)
        original = torch.sigmoid(torch.randn(4, 8, 1, 1))
        output = calibrator(response, original)
        self.assertFalse(torch.equal(output, original))
        diagnostics = calibrator.get_sogc_diagnostics()
        self.assertGreaterEqual(
            float(diagnostics['calibration_factor_min'].min()), 0.9)
        self.assertLessEqual(
            float(diagnostics['calibration_factor_max'].max()), 1.1)
        output.sum().backward()
        gradients = [
            parameter.grad for parameter in calibrator.mlp.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum())
                               for gradient in gradients), 0.0)

    def test_z_clip_fraction_reports_clamp_saturation(self):
        calibrator = _ready_calibrator(channels=4)
        with torch.no_grad():
            calibrator.source_mean.zero_()
            calibrator.source_var.fill_(1.0)
        # Two channels sit far beyond z_clip, two sit inside it.
        response = torch.tensor(
            [[100.0, -100.0, 0.5, -0.5]]).view(1, 4, 1, 1)
        calibrator(response, torch.ones(1, 4, 1, 1))
        diagnostics = calibrator.get_sogc_diagnostics()
        self.assertAlmostEqual(
            float(diagnostics['z_clip_fraction'].item()), 0.5, places=6)
        self.assertAlmostEqual(
            float(diagnostics['max_abs_z'].item()), calibrator.z_clip,
            places=6)

    def test_z_clip_fraction_is_zero_without_saturation(self):
        calibrator = _ready_calibrator(channels=4)
        with torch.no_grad():
            calibrator.source_mean.zero_()
            calibrator.source_var.fill_(1.0)
        calibrator(torch.full((1, 4, 1, 1), 0.25), torch.ones(1, 4, 1, 1))
        diagnostics = calibrator.get_sogc_diagnostics()
        self.assertEqual(float(diagnostics['z_clip_fraction'].item()), 0.0)

    def test_source_statistics_are_non_parameter_buffers(self):
        calibrator = _ready_calibrator()
        parameter_ids = {id(parameter) for parameter in calibrator.parameters()}
        for name in (
                'source_mean', 'source_var', 'source_count', 'source_m2',
                'stats_ready'):
            value = getattr(calibrator, name)
            self.assertNotIn(id(value), parameter_ids)
            self.assertFalse(value.requires_grad)
        calibrator.half()
        self.assertEqual(calibrator.source_mean.dtype, torch.float64)
        self.assertEqual(calibrator.source_var.dtype, torch.float64)


class _Payload(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0]))
        self.sogc = _ready_calibrator(channels=3)


class _Detector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone


class _EMAStudent(SemiBaseDetector):
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = _Payload()
        teacher = _Detector(copy.deepcopy(self.backbone))
        # ``nn.DataParallel`` relocates the wrapped teacher to cuda:0, so the
        # student has to follow the production path and move as well.
        self.ema_model = nn.DataParallel(teacher)
        if torch.cuda.is_available():
            self.cuda()


class TestSOGCIntegration(unittest.TestCase):

    def test_ema_update_preserves_source_statistics(self):
        student = _EMAStudent()
        before_student = {
            name: getattr(student.backbone.sogc, name).clone()
            for name in ('source_mean', 'source_var', 'source_count',
                         'source_m2', 'stats_ready')
        }
        teacher_sogc = student.ema_model.module.backbone.sogc
        before_teacher = {
            name: getattr(teacher_sogc, name).clone()
            for name in before_student
        }
        with torch.no_grad():
            student.backbone.weight.add_(2.0)
        student.update_ema_model(momentum=0.5)
        for name in before_student:
            self.assertTrue(torch.equal(
                getattr(student.backbone.sogc, name), before_student[name]))
            self.assertTrue(torch.equal(
                getattr(teacher_sogc, name), before_teacher[name]))
        teacher_weight = student.ema_model.module.backbone.weight
        self.assertTrue(torch.equal(
            teacher_weight,
            torch.tensor([2.0], device=teacher_weight.device)))

    def test_old_and_new_configs_build_without_changing_detector_heads(self):
        old_cfg = Config.fromfile(SOURCE_CONFIG)
        old_model = build_detector(old_cfg.model)
        self.assertEqual(old_model.backbone.__class__.__name__, 'OrthoNet')
        self.assertEqual(len(list(
            old_model.backbone.iter_sogc_calibrators())), 0)
        old_head_types = (
            old_cfg.model.neck.type,
            old_cfg.model.rpn_head.type,
            old_cfg.model.roi_head.type,
            old_cfg.model.roi_head.bbox_head.type,
        )
        del old_model
        gc.collect()

        new_cfg = Config.fromfile(SOGC_CONFIG)
        new_model = build_detector(new_cfg.model)
        self.assertEqual([
            (stage, block, calibrator.channels)
            for stage, block, calibrator in
            new_model.backbone.iter_sogc_calibrators()
        ], [(2, 5, 1024), (3, 2, 2048)])
        new_head_types = (
            new_cfg.model.neck.type,
            new_cfg.model.rpn_head.type,
            new_cfg.model.roi_head.type,
            new_cfg.model.roi_head.bbox_head.type,
        )
        self.assertEqual(old_head_types, new_head_types)

    def test_legacy_checkpoint_only_misses_new_sogc_keys(self):
        common = dict(
            type='OrthoNet', depth=18, base_channels=8,
            frozen_stages=-1, init_cfg=None)
        legacy = BACKBONES.build(common)
        enabled = BACKBONES.build(dict(
            **common,
            sogc_cfg=dict(
                enabled=True, stages=(2, 3), block_selector='last',
                mode='off', reduction=4)))
        incompatible = enabled.load_state_dict(
            legacy.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(incompatible.missing_keys)
        self.assertTrue(all(
            '.sogc.' in key for key in incompatible.missing_keys))

    @unittest.skipUnless(
        PRODUCTION_CHECKPOINT.is_file(),
        'production OrthoNet checkpoint is unavailable')
    def test_production_checkpoint_only_misses_new_sogc_keys(self):
        cfg = Config.fromfile(SOGC_CONFIG)
        model = build_detector(cfg.model)
        checkpoint = torch.load(PRODUCTION_CHECKPOINT, map_location='cpu')
        incompatible = model.load_state_dict(
            checkpoint['state_dict'], strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertEqual(len(incompatible.missing_keys), 18)
        self.assertTrue(all(
            '.sogc.' in key for key in incompatible.missing_keys))
        del checkpoint, model
        gc.collect()

    def test_whole_backbone_off_mode_is_exact_legacy_equivalent(self):
        common = dict(
            type='OrthoNet', depth=18, base_channels=8,
            frozen_stages=-1, init_cfg=None)
        legacy = BACKBONES.build(common).eval()
        enabled = BACKBONES.build(dict(
            **common,
            sogc_cfg=dict(
                enabled=True, stages=(2, 3), block_selector='last',
                mode='off', reduction=4))).eval()
        enabled.load_state_dict(legacy.state_dict(), strict=False)
        inputs = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            legacy_outputs = legacy(inputs)
            enabled_outputs = enabled(inputs)
        self.assertTrue(all(
            torch.equal(old, new)
            for old, new in zip(legacy_outputs, enabled_outputs)))

    def test_student_teacher_sogc_config_contract_and_no_cga_teacher(self):
        student_cfg = Config.fromfile(SFOD_CONFIG)
        teacher_cfg = Config.fromfile(EMA_CONFIG)
        self.assertEqual(teacher_cfg.model.type, 'OrientedRCNN')
        self.assertEqual(
            student_cfg.model.backbone.sogc_cfg,
            teacher_cfg.model.backbone.sogc_cfg)
        self.assertEqual(student_cfg.model.ema_ckpt, student_cfg.load_from)
        self.assertFalse(student_cfg.model.cfg.semantic_reweight)
        self.assertFalse(student_cfg.model.cfg.dynamic_threshold)
        self.assertEqual(student_cfg.model.cfg.score_thr, 0.7)

    def test_small_orthonet_collect_calibrate_forward_backward(self):
        backbone = BACKBONES.build(dict(
            type='OrthoNet', depth=18, base_channels=8,
            frozen_stages=-1, init_cfg=None,
            sogc_cfg=dict(
                enabled=True, stages=(2, 3), block_selector='last',
                mode='collect', reduction=4, strict_stats=True)))
        backbone.eval()
        with torch.no_grad():
            outputs = backbone(torch.randn(2, 3, 64, 64))
        self.assertEqual(len(outputs), 4)
        backbone.finalize_source_stats()
        backbone.set_sogc_mode('calibrate')
        backbone.train()
        outputs = backbone(torch.randn(1, 3, 64, 64))
        loss = sum(output.mean() for output in outputs)
        loss.backward()
        final_layers = [
            calibrator.mlp[-1]
            for _, _, calibrator in backbone.iter_sogc_calibrators()]
        self.assertTrue(all(
            layer.weight.grad is not None for layer in final_layers))


if __name__ == '__main__':
    unittest.main()
