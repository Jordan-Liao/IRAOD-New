"""Tests for the statistics layer of the spatial geometry gate."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis import gate_spatial_geometry as gate  # noqa: E402


def _rows(domain, values, start=0):
    """Build gate rows carrying a single observable field."""
    field = f'{gate.STAGE_NAMES[0]}_{gate.OBSERVABLE_NAMES[0]}'
    return [
        {'domain': domain, 'image_index': start + index,
         'filename': f'img_{start + index}.png', field: value}
        for index, value in enumerate(values)
    ]


class TestAuroc(unittest.TestCase):

    def test_perfect_separation(self):
        self.assertEqual(
            gate._auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 1.0)

    def test_reversed_separation(self):
        self.assertEqual(
            gate._auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])), 0.0)

    def test_ties_score_one_half(self):
        self.assertEqual(
            gate._auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.5)

    def test_empty_input_returns_none(self):
        self.assertIsNone(gate._auroc(np.array([]), np.array([1.0])))


class TestPairedStats(unittest.TestCase):

    def test_win_rate_and_relative_shift(self):
        clean = np.array([1.0, 2.0, 4.0, 8.0])
        corrupt = np.array([2.0, 4.0, 8.0, 4.0])
        stats = gate._paired_stats(clean, corrupt)
        self.assertEqual(stats['count'], 4)
        self.assertAlmostEqual(stats['win_rate'], 0.75)
        # Relative shifts are [1.0, 1.0, 1.0, -0.5]; the median of the middle
        # pair is 1.0.
        self.assertAlmostEqual(stats['median_relative_shift'], 1.0)

    def test_empty_input_is_reported_as_missing(self):
        stats = gate._paired_stats(np.array([]), np.array([]))
        self.assertEqual(stats['count'], 0)
        self.assertIsNone(stats['win_rate'])


class TestPairedArrays(unittest.TestCase):

    def test_alignment_is_by_filename_not_order(self):
        field = f'{gate.STAGE_NAMES[0]}_{gate.OBSERVABLE_NAMES[0]}'
        clean = _rows('clean', [1.0, 2.0, 3.0])
        corrupt = list(reversed(_rows('chaff', [10.0, 20.0, 30.0])))
        clean_values, corrupt_values = gate._paired_arrays(
            clean, corrupt, field)
        np.testing.assert_allclose(corrupt_values, clean_values * 10.0)

    def test_unmatched_and_non_finite_pairs_are_dropped(self):
        field = f'{gate.STAGE_NAMES[0]}_{gate.OBSERVABLE_NAMES[0]}'
        clean = _rows('clean', [1.0, float('nan'), 3.0])
        corrupt = _rows('chaff', [2.0, 4.0, 6.0])
        corrupt.append({'domain': 'chaff', 'image_index': 9,
                        'filename': 'missing.png', field: 1.0})
        clean_values, corrupt_values = gate._paired_arrays(
            clean, corrupt, field)
        self.assertEqual(clean_values.size, 2)
        self.assertEqual(corrupt_values.size, 2)


class TestMahalanobisScorer(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.fit = rng.normal(size=(400, 6))

    def test_shifted_samples_score_higher_than_in_distribution_ones(self):
        score = gate._mahalanobis_scorer(self.fit, shrinkage=0.1)
        rng = np.random.default_rng(1)
        inside = score(rng.normal(size=(200, 6)))
        shifted = score(rng.normal(size=(200, 6)) + 1.5)
        self.assertGreater(shifted.mean(), inside.mean() * 2.0)

    def test_score_is_invariant_to_per_feature_rescaling(self):
        scale = np.array([1.0, 100.0, 0.01, 5.0, 2.0, 50.0])
        base = gate._mahalanobis_scorer(self.fit, shrinkage=0.1)
        scaled = gate._mahalanobis_scorer(self.fit * scale, shrinkage=0.1)
        probe = np.random.default_rng(2).normal(size=(50, 6))
        np.testing.assert_allclose(
            base(probe), scaled(probe * scale), rtol=1e-8, atol=1e-8)

    def test_full_shrinkage_reduces_to_standardized_euclidean(self):
        score = gate._mahalanobis_scorer(self.fit, shrinkage=1.0)
        probe = np.random.default_rng(3).normal(size=(20, 6))
        mean = self.fit.mean(axis=0)
        std = self.fit.std(axis=0, ddof=0)
        expected = (((probe - mean) / std) ** 2).sum(axis=1)
        np.testing.assert_allclose(score(probe), expected, rtol=1e-8)

    def test_collinear_fit_data_does_not_raise(self):
        collinear = np.tile(self.fit[:, :1], (1, 4))
        score = gate._mahalanobis_scorer(collinear, shrinkage=0.2)
        self.assertTrue(np.isfinite(score(collinear[:10])).all())


class TestUsableFields(unittest.TestCase):

    def test_constant_and_non_finite_columns_are_dropped(self):
        matrix = np.array([
            [1.0, 5.0, np.nan, 2.0],
            [2.0, 5.0, 1.0, 3.0],
            [3.0, 5.0, 1.0, 4.0],
        ])
        fields = ['varying', 'constant', 'nan', 'ok']
        kept, mask = gate._usable_fields(matrix, fields)
        self.assertEqual(kept, ['varying', 'ok'])
        self.assertEqual(mask.tolist(), [True, False, False, True])


class TestMultivariateScores(unittest.TestCase):

    def _full_rows(self, domain, rng, shift=0.0, count=120):
        fields = gate._field_names()
        return [
            {'domain': domain, 'image_index': index,
             'filename': f'{domain}_{index}.png',
             **{field: value for field, value in zip(
                 fields, rng.normal(size=len(fields)) + shift)}}
            for index in range(count)
        ]

    def test_shifted_domain_is_separated(self):
        rng = np.random.default_rng(0)
        clean = self._full_rows('clean', rng)
        corrupt = self._full_rows('chaff', rng, shift=1.0)
        args = SimpleNamespace(shrinkage=0.2, gate_auroc=0.8)
        scores = gate._multivariate_scores(clean, corrupt, args)
        self.assertIn('all_stages', scores)
        self.assertGreater(scores['all_stages']['separability'], 0.9)

    def test_identical_domains_stay_near_chance(self):
        rng = np.random.default_rng(1)
        clean = self._full_rows('clean', rng)
        corrupt = self._full_rows('chaff', rng)
        args = SimpleNamespace(shrinkage=0.2, gate_auroc=0.8)
        scores = gate._multivariate_scores(clean, corrupt, args)
        # The fit split must not leak into the clean evaluation split; if it
        # did, clean scores would be biased low and this would exceed chance.
        self.assertLess(scores['all_stages']['separability'], 0.65)

    def test_fit_and_eval_splits_are_disjoint(self):
        rng = np.random.default_rng(2)
        clean = self._full_rows('clean', rng, count=100)
        corrupt = self._full_rows('chaff', rng, shift=1.0, count=40)
        args = SimpleNamespace(shrinkage=0.2, gate_auroc=0.8)
        entry = gate._multivariate_scores(clean, corrupt, args)['all_stages']
        self.assertEqual(entry['fit_samples'], 50)
        self.assertEqual(entry['clean_eval_samples'], 50)
        self.assertEqual(entry['corrupt_eval_samples'], 40)

    def test_tiny_clean_set_is_skipped(self):
        rng = np.random.default_rng(3)
        args = SimpleNamespace(shrinkage=0.2, gate_auroc=0.8)
        self.assertEqual(
            gate._multivariate_scores(
                self._full_rows('clean', rng, count=8),
                self._full_rows('chaff', rng, count=8), args),
            {})


class TestValidFeatureExtentUse(unittest.TestCase):

    def test_field_names_cover_every_stage_and_observable(self):
        fields = gate._field_names()
        self.assertEqual(
            len(fields),
            len(gate.STAGE_NAMES) * len(gate.OBSERVABLE_NAMES))
        self.assertEqual(len(set(fields)), len(fields))


if __name__ == '__main__':
    unittest.main()
