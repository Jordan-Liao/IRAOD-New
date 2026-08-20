"""Tests for the SLRP calibration pieces: corruption family, transform, hook."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmdet_extension  # noqa: E402,F401
from mmdet_extension.core.hooks import FreezeExceptSLRPHook  # noqa: E402
from mmdet_extension.datasets.pipelines import (  # noqa: E402
    RandomSARInterference)
from mmdet_extension.models.utils.slrp import (  # noqa: E402
    SeparableLowRankProjection)
from tools.dataset.random_interference import (  # noqa: E402
    TRAIN_FAMILIES, apply_random_interference, sample_interference)


class _StubLogger:

    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


class _StubRunner:

    def __init__(self, model):
        self.model = model
        self.logger = _StubLogger()


class _StubModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8))
        self.slrp = SeparableLowRankProjection(channels=8)
        self.head = torch.nn.Linear(8, 4)


class TestRandomInterferenceFamily(unittest.TestCase):

    def test_every_family_member_produces_a_valid_image(self):
        image = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(
            np.uint8)
        seen = set()
        for index in range(200):
            rng = np.random.default_rng(index)
            out, itype = apply_random_interference(image, rng)
            seen.add(itype)
            self.assertEqual(out.shape, image.shape)
            self.assertTrue(np.isfinite(out.astype(np.float64)).all())
        self.assertEqual(seen, set(TRAIN_FAMILIES))

    def test_parameters_are_held_out_from_the_evaluation_specs(self):
        # The evaluation set always places interference at the exact centre and
        # uses fixed magnitudes; the calibration family must not reproduce them.
        for index in range(300):
            rng = np.random.default_rng(index)
            itype, params = sample_interference(rng)
            if 'locations' in params:
                for location in params['locations']:
                    self.assertNotEqual(tuple(location), (0.5, 0.5))
            if itype == 'chaff':
                self.assertLess(max(params['cloudSize']), 0.25)
            if itype == 'smart_suppression':
                self.assertLess(max(params['noiseSize']), 0.25)
            if itype == 'noise_suppression':
                self.assertNotEqual(params['blurKsize'], 5)
            if itype in ('am_noise_horizontal', 'am_noise_vertical'):
                self.assertNotEqual(params['lineWidth'], 10)

    def test_interference_actually_changes_the_image(self):
        # 256px, not 64px: at 64px a sigmaFrac of 0.004 is a 0.26-pixel sigma and
        # the point target rounds away entirely, which is a property of the
        # corruption at small scales rather than a defect.
        image = np.full((256, 256, 3), 100, dtype=np.uint8)
        for index in range(30):
            out, itype = apply_random_interference(
                image, np.random.default_rng(index))
            self.assertFalse(
                np.array_equal(out, image),
                msg=f'{itype} left the image untouched at seed {index}')


class TestRandomSARInterferenceTransform(unittest.TestCase):

    def setUp(self):
        self.image = (np.random.default_rng(1).random((48, 48, 3)) * 255).astype(
            np.uint8)

    def test_probability_zero_is_a_passthrough(self):
        transform = RandomSARInterference(p=0.0)
        results = transform({'img': self.image.copy()})
        np.testing.assert_array_equal(results['img'], self.image)
        self.assertEqual(results['sar_interference'], 'none')

    def test_probability_one_always_corrupts_and_records_the_type(self):
        transform = RandomSARInterference(p=1.0, seed=7)
        results = transform({'img': self.image.copy()})
        self.assertIn(results['sar_interference'], TRAIN_FAMILIES)
        self.assertFalse(np.array_equal(results['img'], self.image))

    def test_dtype_and_range_are_preserved(self):
        transform = RandomSARInterference(p=1.0, seed=3)
        results = transform({'img': self.image.copy()})
        self.assertEqual(results['img'].dtype, np.uint8)
        self.assertGreaterEqual(int(results['img'].min()), 0)
        self.assertLessEqual(int(results['img'].max()), 255)

    def test_half_probability_leaves_clean_samples_in_the_stream(self):
        transform = RandomSARInterference(p=0.5, seed=11)
        types = [transform({'img': self.image.copy()})['sar_interference']
                 for _ in range(200)]
        clean = sum(1 for value in types if value == 'none')
        # Clean samples are what stop the gate learning to fire unconditionally.
        self.assertGreater(clean, 60)
        self.assertLess(clean, 140)

    def test_successive_calls_vary_the_corruption(self):
        transform = RandomSARInterference(p=1.0, seed=5)
        first = transform({'img': self.image.copy()})['img']
        second = transform({'img': self.image.copy()})['img']
        self.assertFalse(np.array_equal(first, second))

    def test_missing_image_is_rejected(self):
        with self.assertRaises(KeyError):
            RandomSARInterference(p=1.0)({})

    def test_invalid_probability_is_rejected(self):
        for value in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                RandomSARInterference(p=value)


class TestFreezeExceptSLRPHook(unittest.TestCase):

    def test_only_slrp_stays_trainable(self):
        model = _StubModel()
        runner = _StubRunner(model)
        FreezeExceptSLRPHook().before_run(runner)
        for name, parameter in model.named_parameters():
            self.assertEqual(
                parameter.requires_grad, 'slrp' in name, msg=name)

    def test_parameter_counts_are_logged(self):
        model = _StubModel()
        runner = _StubRunner(model)
        FreezeExceptSLRPHook().before_run(runner)
        expected = sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if 'slrp' in name)
        self.assertTrue(
            any(f'training {expected} parameters' in message
                for message in runner.logger.messages),
            msg=str(runner.logger.messages))

    def test_norm_layers_outside_slrp_are_held_in_eval_mode(self):
        model = _StubModel()
        runner = _StubRunner(model)
        hook = FreezeExceptSLRPHook()
        hook.before_run(runner)
        model.train()
        hook.before_train_iter(runner)
        self.assertFalse(model.backbone[1].training)

    def test_no_matching_parameter_raises_when_strict(self):
        runner = _StubRunner(_StubModel())
        hook = FreezeExceptSLRPHook(trainable_markers=('nonexistent',))
        with self.assertRaises(RuntimeError):
            hook.before_run(runner)

    def test_empty_markers_are_rejected(self):
        with self.assertRaises(ValueError):
            FreezeExceptSLRPHook(trainable_markers=())


class TestGateDeviationExposure(unittest.TestCase):

    def test_gate_deviation_is_differentiable(self):
        module = SeparableLowRankProjection(channels=8)
        torch.nn.init.normal_(module.gate[-1].weight, std=0.1)
        module(torch.randn(1, 8, 16, 16))
        module.last_gate_deviation().square().mean().backward()
        self.assertIsNotNone(module.gate[0].weight.grad)
        self.assertGreater(module.gate[0].weight.grad.abs().sum().item(), 0.0)

    def test_zero_init_gate_deviation_is_zero(self):
        module = SeparableLowRankProjection(channels=8).eval()
        module(torch.randn(1, 8, 16, 16))
        self.assertLess(module.last_gate_deviation().abs().max().item(), 1e-9)

    def test_deviation_before_forward_raises(self):
        with self.assertRaises(RuntimeError):
            SeparableLowRankProjection(channels=8).last_gate_deviation()


if __name__ == '__main__':
    unittest.main()
