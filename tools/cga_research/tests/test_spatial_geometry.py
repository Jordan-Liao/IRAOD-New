"""Synthetic-signal tests for the spatial geometry observables."""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet_extension.models.utils.spatial_geometry import (  # noqa: E402
    OBSERVABLE_NAMES, compute_observables, energy_map,
    separable_decomposition, valid_feature_extent)


def _feature_from_energy(energy, channels=8):
    """Build a [C, H, W] feature whose channel-mean energy equals ``energy``."""
    return energy.sqrt().unsqueeze(0).repeat(channels, 1, 1)


class TestValidFeatureExtent(unittest.TestCase):

    def test_unpadded_input_keeps_everything(self):
        self.assertEqual(
            valid_feature_extent((25, 25), (800, 800), (800, 800)), (25, 25))

    def test_padded_input_crops_to_content(self):
        # 800x501 content padded to 800x512 at stride 32 -> keep 16 of 16 rows
        # and ceil(501/32)=16 columns of 16.
        self.assertEqual(
            valid_feature_extent((25, 16), (800, 501), (800, 512)), (25, 16))

    def test_stride_four_padding_is_cropped(self):
        # 200x101 content padded to 200x128, stride 4 -> 26 of 32 columns.
        self.assertEqual(
            valid_feature_extent((50, 32), (200, 101), (200, 128)), (50, 26))

    def test_content_larger_than_padded_is_rejected(self):
        with self.assertRaises(ValueError):
            valid_feature_extent((25, 25), (800, 900), (800, 800))

    def test_non_positive_extent_is_rejected(self):
        with self.assertRaises(ValueError):
            valid_feature_extent((25, 0), (800, 800), (800, 800))


class TestSeparableDecomposition(unittest.TestCase):

    def test_rank_one_map_has_zero_residual(self):
        row = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
        col = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float64)
        energy = torch.outer(row, col)
        _, residual, _, _ = separable_decomposition(energy)
        self.assertLess(residual.abs().max().item(), 1e-9)

    def test_marginals_are_preserved(self):
        energy = torch.rand(9, 7, dtype=torch.float64) + 0.5
        separable, _, row_profile, col_profile = \
            separable_decomposition(energy)
        torch.testing.assert_close(separable.mean(dim=1), row_profile)
        torch.testing.assert_close(separable.mean(dim=0), col_profile)

    def test_rejects_non_2d_input(self):
        with self.assertRaises(ValueError):
            separable_decomposition(torch.rand(3, 4, 5, dtype=torch.float64))


class TestObservableBehaviour(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.size = 64
        self.texture = torch.rand(
            self.size, self.size, dtype=torch.float64) + 0.5

    def test_all_observables_are_finite_on_texture(self):
        values = compute_observables(_feature_from_energy(self.texture))
        self.assertEqual(set(values), set(OBSERVABLE_NAMES))
        for name, value in values.items():
            self.assertTrue(
                torch.isfinite(torch.tensor(value)), f'{name} is not finite')

    def test_gain_invariance_except_mean_energy(self):
        base = compute_observables(_feature_from_energy(self.texture))
        scaled = compute_observables(
            _feature_from_energy(self.texture) * 7.0)
        for name in OBSERVABLE_NAMES:
            if name == 'mean_energy':
                self.assertGreater(scaled[name], base[name] * 40)
                continue
            self.assertAlmostEqual(
                base[name], scaled[name], places=6, msg=name)

    def test_horizontal_stripes_raise_row_cv_and_anisotropy(self):
        energy = self.texture.clone()
        energy[::5, :] += 20.0
        striped = compute_observables(_feature_from_energy(energy))
        base = compute_observables(_feature_from_energy(self.texture))
        self.assertGreater(striped['row_profile_cv'], base['row_profile_cv'])
        self.assertGreater(striped['profile_anisotropy'], 1.0)
        self.assertGreater(
            striped['row_spectral_peak'], base['row_spectral_peak'])

    def test_vertical_stripes_flip_the_anisotropy_sign(self):
        energy = self.texture.clone()
        energy[:, ::5] += 20.0
        striped = compute_observables(_feature_from_energy(energy))
        self.assertLess(striped['profile_anisotropy'], -1.0)
        self.assertGreater(striped['col_profile_cv'],
                           striped['row_profile_cv'])

    def test_point_target_raises_concentration_and_residual(self):
        energy = self.texture.clone()
        energy[32, 32] += 500.0
        spike = compute_observables(_feature_from_energy(energy))
        base = compute_observables(_feature_from_energy(self.texture))
        self.assertGreater(spike['energy_gini'], base['energy_gini'])
        self.assertGreater(spike['separable_residual_fraction'],
                           base['separable_residual_fraction'])

    def test_separable_blob_stays_near_the_separable_subspace(self):
        axis = torch.arange(self.size, dtype=torch.float64)
        profile = torch.exp(-((axis - 32.0) ** 2) / (2 * 8.0 ** 2))
        blob = compute_observables(
            _feature_from_energy(torch.outer(profile, profile) + 1e-3))
        spiky = self.texture.clone()
        spiky[32, 32] += 500.0
        self.assertLess(
            blob['separable_residual_fraction'],
            compute_observables(
                _feature_from_energy(spiky))['separable_residual_fraction'])

    def test_low_rank_feature_lowers_participation_ratio(self):
        spatial = torch.rand(1, self.size, self.size, dtype=torch.float64)
        rank_one = spatial.repeat(16, 1, 1) * torch.rand(
            16, 1, 1, dtype=torch.float64)
        full_rank = torch.rand(
            16, self.size, self.size, dtype=torch.float64)
        self.assertLess(
            compute_observables(rank_one)['participation_ratio'],
            compute_observables(full_rank)['participation_ratio'])

    def test_point_target_favours_local_over_global_statistics(self):
        energy = self.texture.clone()
        energy[32, 32] += 500.0
        spike = compute_observables(_feature_from_energy(energy))
        base = compute_observables(_feature_from_energy(self.texture))
        # A single spike is diluted by any whole-image average, so the
        # self-referencing statistics must react far more strongly than gini.
        # A single spike is diluted by any whole-image average, so the
        # peak-to-background ratio must react far more strongly than gini.
        # ``local_cv_contrast`` deliberately does not fire here: a lone pixel
        # perturbs ~0.2% of the windows, well below its 0.9 quantile, which is
        # reserved for area interference.
        peak_gain = (spike['residual_peak_to_median']
                     / base['residual_peak_to_median'])
        gini_gain = spike['energy_gini'] / base['energy_gini']
        self.assertGreater(peak_gain, gini_gain)

    def test_area_interference_raises_local_cv_contrast(self):
        energy = self.texture.clone()
        energy[20:44, 20:44] += 30.0
        patch = compute_observables(_feature_from_energy(energy))
        base = compute_observables(_feature_from_energy(self.texture))
        self.assertGreater(
            patch['local_cv_contrast'], base['local_cv_contrast'])

    def test_raised_noise_floor_lowers_local_cv(self):
        # Barrage-style suppression lifts the floor and homogenises the scene,
        # which is the direction fully developed speckle statistics predict.
        flooded = compute_observables(
            _feature_from_energy(self.texture + 50.0))
        base = compute_observables(_feature_from_energy(self.texture))
        self.assertLess(flooded['local_cv_median'], base['local_cv_median'])
        self.assertLess(
            flooded['local_cv_median_wide'], base['local_cv_median_wide'])

    def test_local_cv_field_is_gain_invariant(self):
        from mmdet_extension.models.utils.spatial_geometry import (
            local_cv_field)
        base = local_cv_field(self.texture, 5)
        scaled = local_cv_field(self.texture * 11.0, 5)
        torch.testing.assert_close(base, scaled)

    def test_energy_map_rejects_non_finite_features(self):
        feature = torch.ones(4, 8, 8, dtype=torch.float64)
        feature[0, 0, 0] = float('nan')
        with self.assertRaises(FloatingPointError):
            energy_map(feature)

    def test_energy_map_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            energy_map(torch.rand(4, 8))


if __name__ == '__main__':
    unittest.main()
