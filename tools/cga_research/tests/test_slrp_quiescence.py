"""Tests for the clean-input gate quiescence penalty."""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmdet_extension  # noqa: E402,F401
from mmdet_extension.models.detectors.slrp_calibration_rcnn import (  # noqa: E402
    SLRPCalibrationRCNN)
from mmdet_extension.models.utils.slrp import (  # noqa: E402
    SeparableLowRankProjection)


class _StubBackbone:

    def __init__(self, modules):
        self._modules_list = modules

    def iter_slrp_modules(self):
        for index, module in enumerate(self._modules_list):
            yield index, module


class _Harness(SLRPCalibrationRCNN):
    """Bypasses OrientedRCNN construction; only the penalty logic is tested."""

    def __init__(self, modules, quiescence_weight=1.0, require_tag=True):
        self.quiescence_weight = float(quiescence_weight)
        self.clean_tag = 'none'
        self.require_tag = bool(require_tag)
        self.backbone = _StubBackbone(modules)


def _ran_module(batch, channels=8, gate_std=0.0):
    module = SeparableLowRankProjection(channels=channels)
    if gate_std > 0.0:
        torch.nn.init.normal_(module.gate[-1].weight, std=gate_std)
    module(torch.randn(batch, channels, 16, 16))
    return module


class TestCleanMask(unittest.TestCase):

    def test_mask_selects_the_clean_samples(self):
        harness = _Harness([_ran_module(3)])
        metas = [{'sar_interference': 'none'},
                 {'sar_interference': 'chaff'},
                 {'sar_interference': 'none'}]
        mask = harness._clean_mask(metas, torch.device('cpu'))
        self.assertEqual(mask.tolist(), [True, False, True])

    def test_missing_tag_raises_when_required(self):
        harness = _Harness([_ran_module(1)], require_tag=True)
        with self.assertRaises(KeyError):
            harness._clean_mask([{}], torch.device('cpu'))

    def test_missing_tag_is_tolerated_when_not_required(self):
        harness = _Harness([_ran_module(1)], require_tag=False)
        self.assertIsNone(harness._clean_mask([{}], torch.device('cpu')))


class TestQuiescenceLoss(unittest.TestCase):

    def test_zero_init_gate_gives_zero_penalty(self):
        harness = _Harness([_ran_module(4)])
        mask = torch.tensor([True, True, True, True])
        self.assertLess(harness._quiescence_loss(mask).item(), 1e-12)

    def test_active_gate_gives_a_positive_penalty(self):
        harness = _Harness([_ran_module(4, gate_std=0.5)])
        mask = torch.tensor([True, False, True, False])
        self.assertGreater(harness._quiescence_loss(mask).item(), 0.0)

    def test_penalty_is_differentiable(self):
        module = _ran_module(2, gate_std=0.3)
        harness = _Harness([module])
        harness._quiescence_loss(torch.tensor([True, True])).backward()
        self.assertGreater(
            module.gate[0].weight.grad.abs().sum().item(), 0.0)

    def test_batch_with_no_clean_sample_yields_zero_with_a_graph(self):
        module = _ran_module(3, gate_std=0.4)
        harness = _Harness([module])
        loss = harness._quiescence_loss(
            torch.tensor([False, False, False]))
        self.assertEqual(loss.item(), 0.0)
        # Must stay connected so DDP gradient reduction sees the parameter.
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(module.gate[0].weight.grad)

    def test_only_clean_samples_contribute(self):
        module = _ran_module(4, gate_std=0.5)
        harness = _Harness([module])
        deviation = module.last_gate_deviation()
        mask = torch.tensor([True, False, False, False])
        expected = deviation[mask].square().mean().item()
        self.assertAlmostEqual(
            harness._quiescence_loss(mask).item(), expected, places=9)

    def test_multiple_modules_are_averaged(self):
        modules = [_ran_module(2, gate_std=0.4), _ran_module(2, gate_std=0.4)]
        harness = _Harness(modules)
        mask = torch.tensor([True, True])
        expected = sum(
            module.last_gate_deviation()[mask].square().mean().item()
            for module in modules) / 2
        self.assertAlmostEqual(
            harness._quiescence_loss(mask).item(), expected, places=7)

    def test_batch_size_mismatch_is_rejected(self):
        harness = _Harness([_ran_module(4)])
        with self.assertRaises(RuntimeError):
            harness._quiescence_loss(torch.tensor([True, False]))

    def test_no_slrp_module_is_rejected(self):
        harness = _Harness([])
        with self.assertRaises(RuntimeError):
            harness._quiescence_loss(torch.tensor([True]))


class TestForwardTrainWiring(unittest.TestCase):
    """The penalty must be attached under a ``loss_`` key to be summed."""

    def setUp(self):
        self._original = SLRPCalibrationRCNN.__mro__[1].forward_train
        SLRPCalibrationRCNN.__mro__[1].forward_train = \
            lambda self, img, img_metas, *a, **k: {
                'loss_cls': torch.zeros((), requires_grad=True)}

    def test_zero_weight_skips_the_penalty(self):
        harness = _Harness(
            [_ran_module(2, gate_std=0.5)], quiescence_weight=0.0)
        losses = SLRPCalibrationRCNN.forward_train(
            harness, torch.zeros(2, 3, 8, 8),
            [{'sar_interference': 'none'}] * 2)
        self.assertNotIn('loss_slrp_quiescence', losses)

    def test_absent_tag_without_requirement_skips_the_penalty(self):
        harness = _Harness(
            [_ran_module(2, gate_std=0.5)], quiescence_weight=5.0,
            require_tag=False)
        losses = SLRPCalibrationRCNN.forward_train(
            harness, torch.zeros(2, 3, 8, 8), [{}, {}])
        self.assertNotIn('loss_slrp_quiescence', losses)

    def test_penalty_is_added_to_the_loss_dict(self):
        module = _ran_module(2, gate_std=0.5)
        harness = _Harness([module], quiescence_weight=7.0)
        losses = SLRPCalibrationRCNN.forward_train(
            harness, torch.zeros(2, 3, 8, 8),
            [{'sar_interference': 'none'},
             {'sar_interference': 'chaff'}])
        self.assertIn('loss_slrp_quiescence', losses)
        self.assertTrue(losses['loss_slrp_quiescence'].requires_grad)
        expected = 7.0 * module.last_gate_deviation()[
            torch.tensor([True, False])].square().mean().item()
        self.assertAlmostEqual(
            losses['loss_slrp_quiescence'].item(), expected, places=6)

    def tearDown(self):
        SLRPCalibrationRCNN.__mro__[1].forward_train = self._original


if __name__ == '__main__':
    unittest.main()
