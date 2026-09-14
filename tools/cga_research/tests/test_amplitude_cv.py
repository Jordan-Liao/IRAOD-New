"""Tests for the amplitude-domain CV observables."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.amplitude_cv import (  # noqa: E402
    OBSERVABLE_NAMES, RAYLEIGH_AMPLITUDE_CV, _box_mean, compute_observables,
    local_cv_field)


def _reference_box_mean(image, window):
    """Direct O(n*k^2) box mean with edge replication, for cross-checking."""
    radius = window // 2
    padded = np.pad(image, radius, mode='edge')
    height, width = image.shape
    out = np.empty_like(image, dtype=np.float64)
    for row in range(height):
        for col in range(width):
            out[row, col] = padded[
                row:row + window, col:col + window].mean()
    return out


class TestBoxMean(unittest.TestCase):

    def test_matches_the_direct_computation(self):
        rng = np.random.default_rng(0)
        image = rng.normal(size=(17, 23)) * 10.0 + 50.0
        for window in (1, 3, 5, 9):
            np.testing.assert_allclose(
                _box_mean(image, window),
                _reference_box_mean(image, window), rtol=1e-9, atol=1e-9)

    def test_constant_image_is_preserved(self):
        image = np.full((12, 15), 7.0)
        np.testing.assert_allclose(_box_mean(image, 5), image)

    def test_even_window_is_rejected(self):
        with self.assertRaises(ValueError):
            _box_mean(np.zeros((8, 8)), 4)


class TestLocalCvField(unittest.TestCase):

    def test_constant_image_has_zero_cv(self):
        field = local_cv_field(np.full((16, 16), 33.0), 5)
        self.assertLess(np.abs(field).max(), 1e-9)

    def test_gain_invariance(self):
        rng = np.random.default_rng(1)
        image = np.abs(rng.normal(size=(24, 24))) + 1.0
        np.testing.assert_allclose(
            local_cv_field(image, 5), local_cv_field(image * 13.0, 5),
            rtol=1e-9, atol=1e-9)

    def test_rayleigh_speckle_lands_near_the_theoretical_cv(self):
        rng = np.random.default_rng(2)
        # Rayleigh amplitude = magnitude of a circular complex Gaussian.
        real = rng.normal(size=(400, 400))
        imaginary = rng.normal(size=(400, 400))
        amplitude = np.sqrt(real ** 2 + imaginary ** 2)
        # A 15x15 window holds 225 samples, enough for the sample CV to sit
        # close to the population value.
        cv_median = float(np.median(local_cv_field(amplitude, 15)))
        self.assertAlmostEqual(cv_median, RAYLEIGH_AMPLITUDE_CV, delta=0.03)

    def test_window_larger_than_image_is_clamped(self):
        field = local_cv_field(np.abs(np.random.default_rng(3).normal(
            size=(4, 6))) + 1.0, 21)
        self.assertEqual(field.shape, (4, 6))
        self.assertTrue(np.isfinite(field).all())

    def test_non_2d_input_is_rejected(self):
        with self.assertRaises(ValueError):
            local_cv_field(np.zeros((4, 4, 3)), 5)


class TestObservables(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(4)
        real = rng.normal(size=(128, 128))
        imaginary = rng.normal(size=(128, 128))
        self.speckle = np.sqrt(real ** 2 + imaginary ** 2) * 30.0

    def test_all_observables_are_finite(self):
        values = compute_observables(self.speckle)
        self.assertEqual(set(values), set(OBSERVABLE_NAMES))
        self.assertTrue(all(np.isfinite(value) for value in values.values()))

    def test_gain_invariance_except_amplitude_mean(self):
        base = compute_observables(self.speckle)
        scaled = compute_observables(self.speckle * 6.0)
        for name in OBSERVABLE_NAMES:
            if name == 'amplitude_mean':
                self.assertAlmostEqual(
                    scaled[name] / base[name], 6.0, places=6)
                continue
            self.assertAlmostEqual(base[name], scaled[name], places=6, msg=name)

    def test_raised_noise_floor_lowers_cv(self):
        # Barrage-style suppression adds an offset, which homogenises the scene.
        flooded = compute_observables(self.speckle + 120.0)
        base = compute_observables(self.speckle)
        self.assertLess(flooded['cv_median'], base['cv_median'])
        self.assertLess(flooded['cv_p10'], base['cv_p10'])

    def test_additive_noise_lowers_cv_relative_to_clean_speckle(self):
        rng = np.random.default_rng(5)
        noisy = self.speckle + rng.normal(scale=25.0, size=self.speckle.shape)
        noisy = np.clip(noisy, 0.0, None)
        self.assertNotAlmostEqual(
            compute_observables(noisy)['cv_median'],
            compute_observables(self.speckle)['cv_median'], places=3)

    def test_point_target_raises_the_peak_ratio(self):
        spiked = self.speckle.copy()
        spiked[64, 64] += 900.0
        self.assertGreater(
            compute_observables(spiked)['amplitude_peak_to_median'],
            compute_observables(self.speckle)['amplitude_peak_to_median'])

    def test_local_smoothing_raises_the_below_half_fraction(self):
        smoothed = self.speckle.copy()
        smoothed[32:96, 32:96] = smoothed[32:96, 32:96].mean()
        self.assertGreater(
            compute_observables(smoothed)['cv_below_half_fraction'],
            compute_observables(self.speckle)['cv_below_half_fraction'])


if __name__ == '__main__':
    unittest.main()
