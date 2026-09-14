"""Tests for the depth-profile statistics layer."""

import sys
import unittest
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis import depth_profile as dp  # noqa: E402


def _profile(values_by_observable):
    """Build a profile dict from ``{observable: {tap: separability}}``."""
    profile = OrderedDict()
    for name, per_tap in values_by_observable.items():
        profile[name] = OrderedDict(
            (tap, {
                'auroc': per_tap.get(tap),
                'separability': per_tap.get(tap),
                'paired': {'count': 10, 'win_rate': 0.9,
                           'median_relative_shift': 0.1},
            })
            for tap in dp.TAP_NAMES)
    return profile


class TestRetention(unittest.TestCase):

    def test_reference_tap_is_normalised_to_one(self):
        profile = _profile({'obs': {tap: 0.9 for tap in dp.TAP_NAMES}})
        retention = dp._retention(profile)
        self.assertAlmostEqual(retention['by_tap']['input'], 1.0)
        self.assertAlmostEqual(retention['by_tap']['C5'], 1.0)

    def test_decay_to_chance_is_reported_as_zero(self):
        values = {tap: 0.5 for tap in dp.TAP_NAMES}
        values['input'] = 0.9
        retention = dp._retention(_profile({'obs': values}))
        self.assertAlmostEqual(retention['by_tap']['input'], 1.0)
        self.assertAlmostEqual(retention['by_tap']['C4'], 0.0)

    def test_half_decay_is_reported_as_one_half(self):
        values = {tap: 0.7 for tap in dp.TAP_NAMES}
        values['input'] = 0.9
        retention = dp._retention(_profile({'obs': values}))
        self.assertAlmostEqual(retention['by_tap']['C2'], 0.5)

    def test_amplification_exceeds_one(self):
        values = {tap: 0.9 for tap in dp.TAP_NAMES}
        values['input'] = 0.7
        values['C2'] = 0.9
        retention = dp._retention(_profile({'obs': values}))
        self.assertAlmostEqual(retention['by_tap']['C2'], 2.0)

    def test_observable_is_picked_by_the_reference_tap(self):
        profile = _profile({
            'weak_at_input': {tap: 0.99 for tap in dp.TAP_NAMES} | {
                'input': 0.55},
            'strong_at_input': {tap: 0.6 for tap in dp.TAP_NAMES} | {
                'input': 0.95},
        })
        retention = dp._retention(profile)
        self.assertEqual(retention['observable'], 'strong_at_input')
        self.assertAlmostEqual(retention['reference_separability'], 0.95)

    def test_chance_level_reference_yields_no_curve(self):
        retention = dp._retention(
            _profile({'obs': {tap: 0.5 for tap in dp.TAP_NAMES}}))
        self.assertIsNone(retention['by_tap'])

    def test_empty_profile_returns_none(self):
        self.assertIsNone(dp._retention(OrderedDict()))


class TestPeakAndBestPerTap(unittest.TestCase):

    def test_peak_finds_the_global_maximum(self):
        profile = _profile({
            'a': {tap: 0.6 for tap in dp.TAP_NAMES},
            'b': {tap: 0.7 for tap in dp.TAP_NAMES} | {'C3': 0.95},
        })
        peak = dp._peak_tap(profile)
        self.assertEqual(peak['observable'], 'b')
        self.assertEqual(peak['tap'], 'C3')
        self.assertAlmostEqual(peak['separability'], 0.95)

    def test_best_per_tap_selects_independently_at_each_tap(self):
        profile = _profile({
            'shallow': {tap: 0.6 for tap in dp.TAP_NAMES} | {'input': 0.9},
            'deep': {tap: 0.6 for tap in dp.TAP_NAMES} | {'C5': 0.9},
        })
        best = dp._best_per_tap(profile)
        self.assertEqual(best['input']['observable'], 'shallow')
        self.assertEqual(best['C5']['observable'], 'deep')

    def test_missing_values_are_tolerated(self):
        profile = _profile({'a': {}})
        for entry in dp._best_per_tap(profile).values():
            self.assertIsNone(entry)
        self.assertIsNone(dp._peak_tap(profile))


class TestFieldNames(unittest.TestCase):

    def test_covers_every_tap_and_observable_uniquely(self):
        fields = dp._field_names()
        self.assertEqual(
            len(fields), len(dp.TAP_NAMES) * len(dp.OBSERVABLE_NAMES))
        self.assertEqual(len(set(fields)), len(fields))
        self.assertEqual(dp.TAP_NAMES[0], 'input')
        self.assertEqual(dp.TAP_NAMES[-1], 'C5')


if __name__ == '__main__':
    unittest.main()
