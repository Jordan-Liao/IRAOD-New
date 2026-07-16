from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.cga_research.evaluate_predictions import (
    DISABLED_CGA_ENV,
    RSAR_CLASSES,
    OfflineEvaluationError,
    evaluate_validated_predictions,
    force_disable_cga,
    load_trusted_pickle,
    reject_test_path,
    validate_dataset_paths,
    validate_predictions,
)


def empty_detections() -> np.ndarray:
    return np.empty((0, 6), dtype=np.float32)


def valid_predictions(num_images: int = 2) -> list[list[np.ndarray]]:
    predictions: list[list[np.ndarray]] = []
    for image_index in range(num_images):
        image_result = [empty_detections() for _ in RSAR_CLASSES]
        image_result[image_index % len(RSAR_CLASSES)] = np.array(
            [[10.0, 20.0, 4.0, 2.0, 0.1, 0.9]], dtype=np.float32
        )
        predictions.append(image_result)
    return predictions


class SyntheticDataset:
    CLASSES = RSAR_CLASSES
    test_mode = True

    def __init__(self) -> None:
        self.data_infos = [
            {"filename": "validation_0001.png"},
            {"filename": "validation_0002.png"},
        ]
        self.annotations = [
            {
                "bboxes": np.array([[10, 20, 4, 2, 0.1]], dtype=np.float32),
                "labels": np.array([0], dtype=np.int64),
                "bboxes_ignore": np.empty((0, 5), dtype=np.float32),
                "labels_ignore": np.empty((0,), dtype=np.int64),
            },
            {
                "bboxes": np.array([[30, 40, 8, 3, -0.2]], dtype=np.float32),
                "labels": np.array([1], dtype=np.int64),
                "bboxes_ignore": np.empty((0, 5), dtype=np.float32),
                "labels_ignore": np.empty((0,), dtype=np.int64),
            },
        ]

    def __len__(self) -> int:
        return len(self.annotations)

    def get_ann_info(self, index: int):
        return self.annotations[index]


class EvaluatePredictionsTests(unittest.TestCase):
    def test_force_disable_cga_overrides_existing_values(self) -> None:
        environment = {
            "CGA_SCORER": "sarclip",
            "CGA_BACKEND": "sarclip",
            "CGA_FILTER_MODE": "legacy",
        }
        effective = force_disable_cga(environment)
        self.assertEqual(effective, DISABLED_CGA_ENV)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_test_path_is_rejected_but_similar_name_is_not(self) -> None:
        with self.assertRaisesRegex(OfflineEvaluationError, "'test' component"):
            reject_test_path(Path("/synthetic/test/annfiles"), "annotations")
        accepted = reject_test_path(Path("/synthetic/contest/annfiles"), "annotations")
        self.assertEqual(accepted, Path("/synthetic/contest/annfiles"))

    def test_dataset_paths_require_existing_non_test_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ann_dir = root / "validation" / "annfiles"
            img_dir = root / "validation" / "images"
            ann_dir.mkdir(parents=True)
            img_dir.mkdir(parents=True)
            ann, img = validate_dataset_paths(
                {"ann_file": str(ann_dir), "img_prefix": str(img_dir)}
            )
            self.assertEqual(ann, ann_dir.resolve())
            self.assertEqual(img, img_dir.resolve())

    def test_pickle_loading_requires_explicit_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.pkl"
            with path.open("wb") as handle:
                pickle.dump({"sentinel": 7}, handle)
            with self.assertRaisesRegex(OfflineEvaluationError, "--trust-pickle"):
                load_trusted_pickle(path, trust_pickle=False)
            self.assertEqual(
                load_trusted_pickle(path, trust_pickle=True), {"sentinel": 7}
            )

    def test_valid_predictions_are_normalized_and_hashed(self) -> None:
        normalized, counts, digest = validate_predictions(
            valid_predictions(), num_images=2, classes=RSAR_CLASSES
        )
        self.assertEqual(len(normalized), 2)
        self.assertEqual(counts, [1, 1, 0, 0, 0, 0])
        self.assertTrue(
            all(item.dtype == np.float32 for row in normalized for item in row)
        )
        self.assertEqual(len(digest), 64)

    def test_prediction_count_class_order_and_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(OfflineEvaluationError, "count mismatch"):
            validate_predictions(
                valid_predictions(1), num_images=2, classes=RSAR_CLASSES
            )
        with self.assertRaisesRegex(OfflineEvaluationError, "class order mismatch"):
            validate_predictions(
                valid_predictions(1),
                num_images=1,
                classes=tuple(reversed(RSAR_CLASSES)),
            )
        malformed = valid_predictions(1)
        malformed[0][0] = np.zeros((1, 5), dtype=np.float32)
        with self.assertRaisesRegex(OfflineEvaluationError, r"expected \(N, 6\)"):
            validate_predictions(malformed, num_images=1, classes=RSAR_CLASSES)

    def test_non_finite_geometry_and_invalid_scores_are_rejected(self) -> None:
        cases = (
            (np.array([[1, 2, np.nan, 4, 0, 0.5]], dtype=np.float32), "NaN"),
            (np.array([[1, 2, 0, 4, 0, 0.5]], dtype=np.float32), "non-positive"),
            (np.array([[1, 2, 3, 4, 0, 1.1]], dtype=np.float32), "outside"),
        )
        for array, message in cases:
            with self.subTest(message=message):
                predictions = valid_predictions(1)
                predictions[0][0] = array
                with self.assertRaisesRegex(OfflineEvaluationError, message):
                    validate_predictions(
                        predictions, num_images=1, classes=RSAR_CLASSES
                    )

    def test_synthetic_dataset_calls_official_contract_and_reports_per_class(
        self,
    ) -> None:
        calls = []

        def fake_eval(det_results, annotations, **kwargs):
            calls.append((det_results, annotations, kwargs))
            class_results = []
            for index in range(len(RSAR_CLASSES)):
                class_results.append(
                    {
                        "num_gts": 1 if index < 2 else 0,
                        "num_dets": 1 if index < 2 else 0,
                        "recall": np.array([1.0]) if index < 2 else np.array([]),
                        "precision": np.array([1.0]) if index < 2 else np.array([]),
                        "ap": 0.9 - index * 0.1,
                    }
                )
            return 0.85, class_results

        result = evaluate_validated_predictions(
            dataset=SyntheticDataset(),
            predictions=valid_predictions(),
            eval_rbbox_map_fn=fake_eval,
            iou_thr=0.5,
            nproc=1,
        )

        self.assertEqual(len(calls), 1)
        _, annotations, kwargs = calls[0]
        self.assertEqual(len(annotations), 2)
        self.assertEqual(kwargs["dataset"], RSAR_CLASSES)
        self.assertEqual(kwargs["iou_thr"], 0.5)
        self.assertTrue(kwargs["use_07_metric"])
        self.assertEqual(kwargs["logger"], "silent")
        self.assertEqual(result["mean_ap"], 0.85)
        self.assertEqual(result["per_class"][0]["class"], "ship")
        self.assertAlmostEqual(result["per_class"][5]["ap"], 0.4)
        self.assertEqual(result["num_detections"], 2)
        self.assertEqual(result["first_image_id"], "validation_0001.png")
        self.assertEqual(len(result["annotations_semantic_sha256"]), 64)
        self.assertEqual(len(result["image_order_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
