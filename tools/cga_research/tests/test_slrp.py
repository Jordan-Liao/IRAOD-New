"""Tests for the separable low-rank projection module and its OrthoNet wiring."""

import sys
import unittest
from pathlib import Path

import torch
from mmdet.models.builder import BACKBONES

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mmdet_extension  # noqa: E402,F401
from mmdet_extension.models.utils.slrp import (  # noqa: E402
    SeparableLowRankProjection, interference_descriptor,
    local_separable_decomposition)


def _build_orthonet(**kwargs):
    cfg = dict(
        type='OrthoNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3),
        frozen_stages=-1, init_cfg=None)
    cfg.update(kwargs)
    return BACKBONES.build(cfg)


class TestLocalSeparableDecomposition(unittest.TestCase):

    def test_horizontal_stripe_raises_the_row_band(self):
        energy = torch.rand(1, 1, 32, 32) + 0.5
        energy[..., 16, :] += 20.0
        _, _, row_band, col_band, _ = local_separable_decomposition(energy, 7)
        self.assertGreater(row_band[0, 0, 16].mean().item(),
                           col_band[0, 0, 16].mean().item())

    def test_constant_map_has_zero_residual(self):
        energy = torch.full((1, 1, 16, 16), 3.0)
        _, residual, _, _, _ = local_separable_decomposition(energy, 5)
        self.assertLess(residual.abs().max().item(), 1e-5)

    def test_window_larger_than_map_is_clamped(self):
        energy = torch.rand(1, 1, 4, 4) + 0.5
        separable, residual, _, _, _ = local_separable_decomposition(energy, 15)
        self.assertEqual(separable.shape, energy.shape)
        self.assertTrue(torch.isfinite(residual).all())

    def test_even_window_is_rejected(self):
        with self.assertRaises(ValueError):
            local_separable_decomposition(torch.rand(1, 1, 8, 8), 4)

    def test_wrong_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            local_separable_decomposition(torch.rand(1, 3, 8, 8), 5)


class TestInterferenceDescriptor(unittest.TestCase):

    def test_shape_and_finiteness(self):
        descriptor = interference_descriptor(torch.randn(2, 8, 24, 24), 7)
        self.assertEqual(descriptor.shape, (2, 6, 24, 24))
        self.assertTrue(torch.isfinite(descriptor).all())

    def test_raised_noise_floor_lowers_the_local_cv_channel(self):
        feature = torch.randn(1, 8, 32, 32)
        flooded = feature + 6.0
        # Channel 4 is the local CV; a lifted floor homogenises the energy.
        self.assertLess(
            interference_descriptor(flooded, 7)[:, 4].mean().item(),
            interference_descriptor(feature, 7)[:, 4].mean().item())

    def test_wide_area_patch_perturbs_the_cv_contrast_channel(self):
        torch.manual_seed(0)
        feature = torch.randn(1, 8, 48, 48)
        patched = feature.clone()
        # A patch far wider than the window: a narrow/wide window ratio would
        # collapse to one in its interior, a scene-median reference does not.
        # The perturbation is additive, matching how smart_suppression injects
        # noise; a multiplicative one would leave CV unchanged by design.
        patched[:, :, 12:36, 12:36] += 5.0
        base = interference_descriptor(feature, 7)[:, 5]
        perturbed = interference_descriptor(patched, 7)[:, 5]
        interior = (slice(None), slice(18, 30), slice(18, 30))
        self.assertGreater(
            (perturbed[interior].mean() - base[interior].mean()).abs().item(),
            0.1)

    def test_cv_contrast_is_scale_free_across_scene_brightness(self):
        torch.manual_seed(1)
        feature = torch.randn(1, 8, 32, 32)
        torch.testing.assert_close(
            interference_descriptor(feature, 7)[:, 5],
            interference_descriptor(feature * 17.0, 7)[:, 5],
            atol=2e-4, rtol=2e-4)

    def test_padding_does_not_shift_the_cv_reference(self):
        torch.manual_seed(2)
        feature = torch.randn(1, 8, 32, 32)
        padded = torch.zeros(1, 8, 32, 64)
        padded[:, :, :, :32] = feature
        base = interference_descriptor(feature, 7)[:, 5]
        with_padding = interference_descriptor(padded, 7)[:, 5, :, :32]
        # The content region's contrast must not depend on how much zero padding
        # the batch collation appended.
        self.assertLess(
            (base[:, 4:28, 4:28] - with_padding[:, 4:28, 4:28]).abs().max()
            .item(), 0.05)

    def test_all_zero_feature_stays_finite(self):
        descriptor = interference_descriptor(torch.zeros(2, 4, 16, 16), 7)
        self.assertTrue(torch.isfinite(descriptor).all())

    def test_small_map_keeps_all_six_channels_finite(self):
        descriptor = interference_descriptor(torch.randn(1, 4, 3, 5), 9)
        self.assertEqual(descriptor.shape, (1, 6, 3, 5))
        self.assertTrue(torch.isfinite(descriptor).all())

    def test_gain_invariance(self):
        feature = torch.randn(1, 8, 24, 24)
        base = interference_descriptor(feature, 7)
        scaled = interference_descriptor(feature * 9.0, 7)
        torch.testing.assert_close(base, scaled, atol=2e-4, rtol=2e-4)

    def test_stripe_direction_flips_the_band_ratios(self):
        horizontal = torch.randn(1, 8, 32, 32)
        horizontal[:, :, ::5, :] *= 6.0
        vertical = horizontal.transpose(2, 3).contiguous()
        h_desc = interference_descriptor(horizontal, 7)
        v_desc = interference_descriptor(vertical, 7)
        # Channels 2 and 3 are the log row/column band ratios.
        h_gap = (h_desc[:, 2] - h_desc[:, 3])
        v_gap = (v_desc[:, 2] - v_desc[:, 3])
        # On a bright row the row band stays high while the column band mixes
        # in the dark neighbours, so the gap is positive there.
        self.assertGreater(h_gap[:, ::5, :].mean().item(), 0.0)
        # Off-stripe pixels are the majority and see the opposite contrast, so
        # the whole-map mean is negative for horizontal stripes and positive for
        # vertical ones. Either way the two directions separate by their sign.
        self.assertLess(h_gap.mean().item(), 0.0)
        self.assertGreater(v_gap.mean().item(), 0.0)


class TestSeparableLowRankProjection(unittest.TestCase):

    def test_zero_initialization_is_exact_identity(self):
        module = SeparableLowRankProjection(channels=16).eval()
        feature = torch.randn(2, 16, 20, 20)
        torch.testing.assert_close(module(feature), feature)

    def test_gate_stays_within_gamma_bounds(self):
        module = SeparableLowRankProjection(channels=8, gamma=0.4)
        torch.nn.init.normal_(module.gate[-1].weight, std=5.0)
        torch.nn.init.normal_(module.gate[-1].bias, std=5.0)
        feature = torch.randn(1, 8, 16, 16)
        ratio = module(feature) / feature
        self.assertGreaterEqual(ratio.min().item(), 1.0 - 0.4 - 1e-5)
        self.assertLessEqual(ratio.max().item(), 1.0 + 0.4 + 1e-5)

    def test_diagnostics_expose_a_suppression_map(self):
        module = SeparableLowRankProjection(channels=8)
        module(torch.randn(2, 8, 12, 14))
        diagnostics = module.get_slrp_diagnostics()
        self.assertEqual(
            diagnostics['suppression_map'].shape, (2, 1, 12, 14))
        self.assertEqual(diagnostics['gate_magnitude'].shape, (2,))

    def test_gradients_reach_the_gate(self):
        module = SeparableLowRankProjection(channels=8)
        torch.nn.init.normal_(module.gate[-1].weight, std=0.1)
        module(torch.randn(1, 8, 16, 16)).square().mean().backward()
        gradient = module.gate[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    def test_small_feature_map_is_supported(self):
        module = SeparableLowRankProjection(channels=8, window=9).eval()
        feature = torch.randn(1, 8, 4, 4)
        torch.testing.assert_close(module(feature), feature)

    def test_channel_mismatch_is_rejected(self):
        module = SeparableLowRankProjection(channels=8)
        with self.assertRaises(ValueError):
            module(torch.randn(1, 6, 16, 16))

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (dict(channels=0), dict(channels=8, window=4),
                       dict(channels=8, gamma=1.5),
                       dict(channels=8, hidden_channels=0),
                       dict(channels=8, sparse_multiplier=-1.0)):
            with self.assertRaises(ValueError):
                SeparableLowRankProjection(**kwargs)


class TestOrthoNetIntegration(unittest.TestCase):

    def test_disabled_slrp_creates_no_state_keys(self):
        backbone = _build_orthonet()
        self.assertFalse(hasattr(backbone, 'slrp'))
        self.assertFalse(
            any('slrp' in key for key in backbone.state_dict()))
        self.assertEqual(backbone.get_slrp_metadata(), [])

    def test_enabled_slrp_targets_the_requested_stages(self):
        backbone = _build_orthonet(
            slrp_cfg=dict(enabled=True, stages=(2, 3), window=7))
        metadata = backbone.get_slrp_metadata()
        self.assertEqual(
            [entry['feature_stage'] for entry in metadata], ['C4', 'C5'])
        self.assertEqual(
            [entry['channels'] for entry in metadata], [1024, 2048])

    def test_stem_stage_is_sized_from_the_maxpool_output(self):
        backbone = _build_orthonet(slrp_cfg=dict(enabled=True, stages=(-1,)))
        metadata = backbone.get_slrp_metadata()
        self.assertEqual(
            [entry['feature_stage'] for entry in metadata], ['stem_maxpool'])
        self.assertEqual([entry['channels'] for entry in metadata], [64])

    def test_stem_stage_is_the_default(self):
        backbone = _build_orthonet(slrp_cfg=dict(enabled=True))
        self.assertEqual(
            [entry['stage_index'] for entry in backbone.get_slrp_metadata()],
            [-1])

    def test_stem_and_res_stages_can_combine(self):
        backbone = _build_orthonet(
            slrp_cfg=dict(enabled=True, stages=(-1, 3))).eval()
        self.assertEqual(
            [entry['feature_stage']
             for entry in backbone.get_slrp_metadata()],
            ['stem_maxpool', 'C5'])
        with torch.no_grad():
            outputs = backbone(torch.randn(1, 3, 64, 64))
        self.assertEqual(len(outputs), 4)

    def test_stem_slrp_gate_receives_gradient(self):
        backbone = _build_orthonet(slrp_cfg=dict(enabled=True, stages=(-1,)))
        module = backbone.slrp['-1']
        torch.nn.init.normal_(module.gate[-1].weight, std=0.05)
        backbone(torch.randn(1, 3, 64, 64))[-1].square().mean().backward()
        self.assertGreater(module.gate[0].weight.grad.abs().sum().item(), 0.0)

    def test_stem_slrp_alters_every_output_stage_once_trained(self):
        torch.manual_seed(0)
        backbone = _build_orthonet(
            slrp_cfg=dict(enabled=True, stages=(-1,))).eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            reference = backbone(image)
            torch.nn.init.normal_(
                backbone.slrp['-1'].gate[-1].weight, std=0.5)
            perturbed = backbone(image)
        # A stem-level gate propagates through the whole trunk, unlike a C5
        # gate which can only touch the last output.
        for before, after in zip(reference, perturbed):
            self.assertFalse(torch.allclose(before, after))

    def test_deep_stem_is_sized_correctly(self):
        backbone = _build_orthonet(
            deep_stem=True, slrp_cfg=dict(enabled=True, stages=(-1,)))
        self.assertEqual(
            [entry['channels'] for entry in backbone.get_slrp_metadata()], [64])

    def test_stage_below_stem_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_orthonet(slrp_cfg=dict(enabled=True, stages=(-2,)))

    def test_enabled_slrp_preserves_the_legacy_forward(self):
        torch.manual_seed(0)
        plain = _build_orthonet().eval()
        with_slrp = _build_orthonet(
            slrp_cfg=dict(enabled=True, stages=(2, 3))).eval()
        with_slrp.load_state_dict(plain.state_dict(), strict=False)
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            for reference, candidate in zip(plain(image), with_slrp(image)):
                torch.testing.assert_close(reference, candidate)

    def test_legacy_checkpoint_loads_without_slrp_keys(self):
        plain = _build_orthonet()
        with_slrp = _build_orthonet(slrp_cfg=dict(enabled=True, stages=(3,)))
        incompatible = with_slrp.load_state_dict(
            plain.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(
            all('slrp' in key for key in incompatible.missing_keys))

    def test_slrp_and_sogc_coexist(self):
        backbone = _build_orthonet(
            sogc_cfg=dict(enabled=True, stages=(3,), mode='off'),
            slrp_cfg=dict(enabled=True, stages=(3,))).eval()
        with torch.no_grad():
            outputs = backbone(torch.randn(1, 3, 64, 64))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(len(backbone.get_slrp_metadata()), 1)
        self.assertEqual(len(backbone.get_sogc_metadata()), 1)

    def test_slrp_diagnostics_are_exposed_per_stage(self):
        backbone = _build_orthonet(
            slrp_cfg=dict(enabled=True, stages=(2, 3))).eval()
        with torch.no_grad():
            backbone(torch.randn(1, 3, 64, 64))
        diagnostics = backbone.get_slrp_diagnostics()
        self.assertEqual(list(diagnostics), ['C4', 'C5'])
        for stage in diagnostics.values():
            self.assertIn('suppression_map', stage)

    def test_invalid_slrp_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_orthonet(slrp_cfg=dict(enabled=True, stages=(4,)))

    def test_unknown_slrp_option_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_orthonet(slrp_cfg=dict(enabled=True, bogus=1))

    def test_duplicate_slrp_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            _build_orthonet(slrp_cfg=dict(enabled=True, stages=(2, 2)))


if __name__ == '__main__':
    unittest.main()
