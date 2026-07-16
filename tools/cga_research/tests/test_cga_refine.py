import importlib.util
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "iraod_cga_under_test", REPO_ROOT / "sfod" / "cga.py"
)
CGA_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CGA_MODULE)


def _detections(scores):
    scores = np.asarray(scores, dtype=np.float32)
    rows = []
    for index, score in enumerate(scores):
        rows.append([10.0 + index, 20.0, 4.0, 2.0, 0.0, score])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _empty_detections():
    return np.empty((0, 6), dtype=np.float32)


def _clone_results(results):
    return [[class_results.copy() for class_results in image] for image in results]


class _DummyCGA:
    class_names = ["class0", "class1"]

    def __init__(self, logits_by_filename):
        self.logits_by_filename = {
            filename: np.asarray(logits, dtype=np.float64)
            for filename, logits in logits_by_filename.items()
        }
        self.calls = []

    def __call__(self, filename, boxes, scores, labels):
        self.calls.append(
            (
                filename,
                np.asarray(boxes).copy(),
                np.asarray(scores).copy(),
                np.asarray(labels).copy(),
            )
        )
        logits = self.logits_by_filename[filename]
        if len(logits) != len(labels):
            raise AssertionError(
                f"dummy logits mismatch for {filename}: {len(logits)} != {len(labels)}"
            )
        return logits.copy(), []


class _Harness(CGA_MODULE.TestMixins):
    CLASSES = ("class0", "class1")

    def __init__(self, cga, mode="legacy"):
        super().__init__()
        self.cga = cga
        self.exclude_ids = []
        self.cga_filter_mode = mode
        self.cga_blend_detector_weight = 0.7
        self.cga_drop_score = 0.0
        self.cga_disagree_delta = 0.1
        self.cga_disagree_score_thr = 0.9
        self.cga_shuffle_seed = 0
        self.cga_filter_log_every = 0
        # Avoid the intentional first-call diagnostic log in unit tests.
        self._cga_filter_calls = 1


class TestCGARefine(unittest.TestCase):
    def test_two_image_batch_modifies_second_image_and_accumulates_diag(self):
        scorer = _DummyCGA(
            {
                "/images/first.png": [[0.1, 0.9]],
                "/images/second.png": [[0.9, 0.1]],
            }
        )
        harness = _Harness(scorer, mode="legacy")
        results = [
            [_detections([0.9]), _empty_detections()],
            [_empty_detections(), _detections([0.8])],
        ]

        refined = harness.refine_test(
            results,
            [
                {"filename": "/images/first.png"},
                {"filename": "/images/second.png"},
            ],
        )

        self.assertIs(refined, results)
        self.assertAlmostEqual(float(results[0][0][0, -1]), 0.66, places=6)
        self.assertAlmostEqual(float(results[1][1][0, -1]), 0.59, places=6)
        self.assertEqual([call[0] for call in scorer.calls], [
            "/images/first.png", "/images/second.png"
        ])
        self.assertEqual(harness._cga_diag_window["calls"], 2)
        self.assertEqual(harness._cga_diag_window["total"], 2)
        self.assertEqual(harness._cga_diag_window["blended"], 2)

    def test_empty_image_is_safe_and_later_image_is_processed(self):
        scorer = _DummyCGA({"/images/nonempty.png": [[0.1, 0.9]]})
        harness = _Harness(scorer, mode="legacy")
        results = [
            [_empty_detections(), _empty_detections()],
            [_detections([0.9]), _empty_detections()],
        ]

        harness.refine_test(
            results,
            [
                {"filename": "/images/empty.png"},
                {"filename": "/images/nonempty.png"},
            ],
        )

        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(scorer.calls[0][0], "/images/nonempty.png")
        self.assertAlmostEqual(float(results[1][0][0, -1]), 0.66, places=6)

    def test_batch_length_mismatch_is_rejected(self):
        harness = _Harness(_DummyCGA({}), mode="legacy")
        with self.assertRaisesRegex(ValueError, "CGA batch mismatch"):
            harness.refine_test(
                [[_empty_detections(), _empty_detections()]],
                [],
            )

    def test_fixed_disagreement_penalty_only_changes_disagreements(self):
        scorer = _DummyCGA(
            {
                "/images/fixed.png": [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.2, 0.8],
                ]
            }
        )
        harness = _Harness(scorer, mode="fixed_disagreement_penalty")
        harness.cga_disagree_delta = 0.25
        results = [[_detections([0.8, 0.2]), _detections([0.7])]]

        harness.refine_test(results, [{"filename": "/images/fixed.png"}])

        np.testing.assert_allclose(results[0][0][:, -1], [0.55, 0.0], atol=1e-6)
        self.assertAlmostEqual(float(results[0][1][0, -1]), 0.7, places=6)
        self.assertEqual(harness._cga_diag_window["penalized"], 2)

    def test_disagreement_threshold_drops_only_low_score_disagreements(self):
        scorer = _DummyCGA(
            {
                "/images/threshold.png": [
                    [0.1, 0.9],
                    [0.2, 0.8],
                    [0.2, 0.8],
                ]
            }
        )
        harness = _Harness(scorer, mode="disagreement_threshold")
        harness.cga_disagree_score_thr = 0.9
        harness.cga_drop_score = 0.03
        results = [[_detections([0.8, 0.95]), _detections([0.7])]]

        harness.refine_test(results, [{"filename": "/images/threshold.png"}])

        np.testing.assert_allclose(results[0][0][:, -1], [0.03, 0.95], atol=1e-6)
        self.assertAlmostEqual(float(results[0][1][0, -1]), 0.7, places=6)
        self.assertEqual(harness._cga_diag_window["threshold_dropped"], 1)
        self.assertEqual(harness._cga_diag_window["dropped"], 1)

    def test_stratified_shuffle_is_stable_derangement_per_stratum(self):
        probabilities = np.asarray(
            [
                [0.90, 0.05, 0.05],
                [0.10, 0.80, 0.10],
                [0.20, 0.10, 0.70],
                [0.60, 0.20, 0.20],
                [0.15, 0.75, 0.10],
                [0.10, 0.85, 0.05],
                [0.60, 0.30, 0.10],
                [0.70, 0.20, 0.10],
            ],
            dtype=np.float64,
        )
        scores = np.asarray([0.91, 0.92, 0.94, 0.96, 0.97, 0.91, 0.92, 0.85])
        labels = np.asarray([0, 0, 0, 0, 0, 1, 1, 0])
        identities = [f"image-{index}" for index in range(len(scores))]

        shuffled_a, sources_a = (
            CGA_MODULE.stratified_shuffle_probability_vectors(
                probabilities,
                scores,
                labels,
                identities,
                seed=12345,
            )
        )
        shuffled_b, sources_b = (
            CGA_MODULE.stratified_shuffle_probability_vectors(
                probabilities,
                scores,
                labels,
                identities,
                seed=12345,
            )
        )
        np.testing.assert_array_equal(shuffled_a, shuffled_b)
        np.testing.assert_array_equal(sources_a, sources_b)

        reordered = np.asarray([6, 2, 7, 0, 5, 3, 1, 4])
        shuffled_reordered, _ = (
            CGA_MODULE.stratified_shuffle_probability_vectors(
                probabilities[reordered],
                scores[reordered],
                labels[reordered],
                [identities[index] for index in reordered],
                seed=12345,
            )
        )
        restored = np.empty_like(shuffled_reordered)
        restored[reordered] = shuffled_reordered
        np.testing.assert_array_equal(shuffled_a, restored)

        strata = ([0, 1, 2], [3, 4], [5, 6])
        for indices in strata:
            indices = np.asarray(indices)
            self.assertTrue(np.all(sources_a[indices] != indices))
            self.assertEqual(
                sorted(map(tuple, shuffled_a[indices])),
                sorted(map(tuple, probabilities[indices])),
            )
            real_agree = np.argmax(probabilities[indices], axis=1) == labels[indices]
            shuffled_agree = np.argmax(shuffled_a[indices], axis=1) == labels[indices]
            self.assertEqual(int(real_agree.sum()), int(shuffled_agree.sum()))

        # A singleton score stratum is explicitly left in place.
        self.assertEqual(int(sources_a[7]), 7)
        np.testing.assert_array_equal(shuffled_a[7], probabilities[7])

    def test_shuffled_legacy_uses_cross_image_operative_trigger(self):
        first = "/dataset/chaff/first.png"
        second = "/dataset/chaff/second.png"
        logits = {
            first: [[0.90, 0.10]],   # real agreement for detector label 0
            second: [[0.20, 0.80]],  # real disagreement for detector label 0
        }
        original = [
            [_detections([0.91]), _empty_detections()],
            [_detections([0.92]), _empty_detections()],
        ]
        metas = [{"filename": first}, {"filename": second}]

        shuffled_harness = _Harness(
            _DummyCGA(logits), mode="shuffled_legacy")
        shuffled_harness.cga_shuffle_seed = 7
        shuffled_results = _clone_results(original)
        shuffled_harness.refine_test(shuffled_results, metas)

        # Each image is a singleton by itself, so these changes prove that the
        # donor pool spans both images in the current inference batch.
        self.assertAlmostEqual(
            float(shuffled_results[0][0][0, -1]), 0.7 * 0.91 + 0.3 * 0.20,
            places=6,
        )
        self.assertAlmostEqual(
            float(shuffled_results[1][0][0, -1]), 0.92, places=6)
        self.assertEqual(shuffled_harness._cga_diag_window["moved"], 2)
        self.assertEqual(shuffled_harness._cga_diag_window["unmoved"], 0)
        self.assertEqual(shuffled_harness._cga_diag_window["real_agree"], 1)
        self.assertEqual(shuffled_harness._cga_diag_window["operative_agree"], 1)
        self.assertEqual(shuffled_harness._cga_diag_window["shuffled"], 1)

        legacy_harness = _Harness(_DummyCGA(logits), mode="legacy")
        legacy_results = _clone_results(original)
        legacy_harness.refine_test(legacy_results, metas)
        legacy_membership = [
            float(image[0][0, -1]) >= 0.9 for image in legacy_results]
        shuffled_membership = [
            float(image[0][0, -1]) >= 0.9 for image in shuffled_results]
        self.assertEqual(legacy_membership, [True, False])
        self.assertEqual(shuffled_membership, [False, True])

        repeat_harness = _Harness(
            _DummyCGA(logits), mode="shuffled_legacy")
        repeat_harness.cga_shuffle_seed = 7
        repeated = _clone_results(original)
        repeat_harness.refine_test(repeated, metas)
        for first_result, second_result in zip(shuffled_results, repeated):
            for first_class, second_class in zip(first_result, second_result):
                np.testing.assert_array_equal(first_class, second_class)


if __name__ == "__main__":
    unittest.main()
