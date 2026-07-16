from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.cga_research.box_audit import (
    PROBABILITY_FIELDS,
    SHUFFLED_PROBABILITY_FIELDS,
    ScoreSettings,
    apply_global_shuffled_control,
    classify_match,
    evidence_veto_trigger,
    fixed_train_research_pool,
    inference_without_cga,
    load_completed_images,
    merge_part_csvs,
    score_box_variants,
    select_split_stems,
    shuffled_legacy_scores,
    summarize_method,
)


class MatchClassificationTests(unittest.TestCase):
    def test_iou_category_boundaries(self) -> None:
        self.assertEqual(classify_match(0.5, 2, 2), "TP")
        self.assertEqual(classify_match(0.5, 2, 3), "wrong-class")
        self.assertEqual(classify_match(0.1, 2, 2), "pure-FP")
        self.assertEqual(
            classify_match(float(np.nextafter(0.1, 1.0)), 2, 2),
            "localization-error",
        )
        self.assertEqual(
            classify_match(float(np.nextafter(0.5, 0.0)), 2, 2),
            "localization-error",
        )
        self.assertEqual(classify_match(0.0, 2, -1), "pure-FP")


class ScoringTests(unittest.TestCase):
    def test_legacy_formula_and_threshold_crossing(self) -> None:
        settings = ScoreSettings(score_thr=0.9)
        probabilities = np.array([0.10, 0.80, 0.025, 0.025, 0.025, 0.025])
        result = score_box_variants(0.9, 0, probabilities, settings)

        self.assertFalse(result["agreement"])
        self.assertAlmostEqual(result["legacy_score"], 0.7 * 0.9 + 0.3 * 0.1)
        adaptive_weight = 0.3 + (0.95 - 0.3) * 0.9
        self.assertAlmostEqual(
            result["adaptive_score"],
            0.9 * adaptive_weight + 0.1 * (1.0 - adaptive_weight),
        )
        self.assertGreaterEqual(0.9, settings.score_thr)
        self.assertLess(result["legacy_score"], settings.score_thr)
        self.assertAlmostEqual(result["fixed_penalty_score"], 0.8)

    def test_agreement_preserves_legacy_score(self) -> None:
        probabilities = np.array([0.85, 0.03, 0.03, 0.03, 0.03, 0.03])
        result = score_box_variants(0.92, 0, probabilities, ScoreSettings())
        self.assertTrue(result["agreement"])
        self.assertAlmostEqual(result["legacy_score"], 0.92)
        self.assertAlmostEqual(result["adaptive_score"], 0.92)
        self.assertAlmostEqual(result["fixed_penalty_score"], 0.92)

    def test_disagreement_threshold_uses_strict_less_than(self) -> None:
        probabilities = np.array([0.10, 0.80, 0.025, 0.025, 0.025, 0.025])
        settings = ScoreSettings(
            disagreement_score_thr=0.9, disagreement_drop_score=0.0)
        at_boundary = score_box_variants(0.9, 0, probabilities, settings)
        below = score_box_variants(0.899, 0, probabilities, settings)
        self.assertAlmostEqual(at_boundary["disagreement_threshold_score"], 0.9)
        self.assertAlmostEqual(below["disagreement_threshold_score"], 0.0)

    def test_default_disagreement_threshold_matches_production(self) -> None:
        self.assertAlmostEqual(ScoreSettings().disagreement_score_thr, 0.95)

    def test_evidence_veto_trigger_and_protections(self) -> None:
        probabilities = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
        settings = ScoreSettings()
        self.assertTrue(evidence_veto_trigger(probabilities, 0, settings))

        same_group = np.array([0.01, 0.95, 0.01, 0.01, 0.01, 0.01])
        self.assertFalse(evidence_veto_trigger(same_group, 0, settings))

        skip_context = dataclass_replace(settings, evidence_skip_context=True)
        self.assertFalse(evidence_veto_trigger(probabilities, 0, skip_context))

    def test_shuffled_legacy_is_stable_and_uses_complete_vectors(self) -> None:
        scores = np.array([0.91, 0.92, 0.93])
        labels = np.array([0, 0, 0])
        probabilities = np.array(
            [
                [0.80, 0.04, 0.04, 0.04, 0.04, 0.04],
                [0.10, 0.70, 0.05, 0.05, 0.05, 0.05],
                [0.20, 0.10, 0.60, 0.04, 0.03, 0.03],
            ]
        )
        identities = ["a", "b", "c"]
        first = shuffled_legacy_scores(
            scores,
            labels,
            probabilities,
            identities,
            seed=17,
            detector_weight=0.7,
        )
        second = shuffled_legacy_scores(
            scores,
            labels,
            probabilities,
            identities,
            seed=17,
            detector_weight=0.7,
        )
        for first_value, second_value in zip(first, second):
            np.testing.assert_array_equal(first_value, second_value)
        shuffled, label_probabilities, rescored, sources = first
        self.assertTrue(np.all(sources != np.arange(len(sources))))
        self.assertEqual(
            sorted(map(tuple, shuffled)), sorted(map(tuple, probabilities)))
        np.testing.assert_array_equal(
            label_probabilities, shuffled[:, 0])
        operative_agreement = np.argmax(shuffled, axis=1) == labels
        expected = scores.copy()
        expected[~operative_agreement] = (
            0.7 * scores[~operative_agreement]
            + 0.3 * label_probabilities[~operative_agreement]
        )
        np.testing.assert_allclose(rescored, expected)

    def test_audit_shuffle_is_global_per_split_and_changes_membership(self) -> None:
        def record(
            split: str,
            image: str,
            score: float,
            probabilities: list[float],
        ) -> dict[str, object]:
            row: dict[str, object] = {
                "split": split,
                "image_stem": image,
                "detection_index": 0,
                "detector_label_id": 0,
                "detector_score": score,
                "agreement": int(np.argmax(probabilities) == 0),
            }
            row.update(dict(zip(PROBABILITY_FIELDS, probabilities)))
            return row

        records = [
            record("train", "first", 0.91, [0.90, 0.02, 0.02, 0.02, 0.02, 0.02]),
            record("train", "second", 0.92, [0.20, 0.70, 0.03, 0.03, 0.02, 0.02]),
            # Same label/bin but another split: it must remain a singleton.
            record("val", "third", 0.93, [0.85, 0.03, 0.03, 0.03, 0.03, 0.03]),
        ]
        repeated = copy.deepcopy(records)
        diagnostics = apply_global_shuffled_control(
            records, ScoreSettings(score_thr=0.9), seed=29)
        repeated_diagnostics = apply_global_shuffled_control(
            repeated, ScoreSettings(score_thr=0.9), seed=29)

        self.assertEqual(diagnostics, repeated_diagnostics)
        self.assertEqual(diagnostics["splits"]["train"]["moved"], 2)
        self.assertEqual(diagnostics["splits"]["train"]["real_agree"], 1)
        self.assertEqual(diagnostics["splits"]["train"]["operative_agree"], 1)
        self.assertEqual(diagnostics["splits"]["train"]["trigger_changed"], 2)
        self.assertEqual(diagnostics["splits"]["val"]["moved"], 0)
        self.assertEqual(int(records[2]["shuffled_moved"]), 0)

        real_vectors = [
            tuple(float(row[field]) for field in PROBABILITY_FIELDS)
            for row in records[:2]
        ]
        shuffled_vectors = [
            tuple(float(row[field]) for field in SHUFFLED_PROBABILITY_FIELDS)
            for row in records[:2]
        ]
        self.assertEqual(sorted(real_vectors), sorted(shuffled_vectors))
        self.assertEqual(
            [int(row["agreement"]) for row in records[:2]], [1, 0])
        self.assertEqual(
            [int(row["shuffled_agreement"]) for row in records[:2]], [0, 1])
        self.assertEqual(
            [int(row["enters_student_shuffled_legacy"]) for row in records[:2]],
            [0, 1],
        )
        self.assertEqual(records, repeated)


def dataclass_replace(settings: ScoreSettings, **changes: object) -> ScoreSettings:
    values = {
        field: getattr(settings, field)
        for field in settings.__dataclass_fields__
    }
    values.update(changes)
    return ScoreSettings(**values)


class SummaryMetricTests(unittest.TestCase):
    def test_crossing_metrics(self) -> None:
        records = [
            {
                "detector_score": 0.95,
                "legacy_score": 0.95,
                "match_category": "TP",
            },
            {
                "detector_score": 0.95,
                "legacy_score": 0.80,
                "match_category": "pure-FP",
            },
            {
                "detector_score": 0.95,
                "legacy_score": 0.94,
                "match_category": "wrong-class",
            },
            {
                "detector_score": 0.80,
                "legacy_score": 0.70,
                "match_category": "localization-error",
            },
        ]
        metrics = summarize_method(records, "legacy_score", 0.9)
        self.assertEqual(metrics["baseline_kept"], 3)
        self.assertEqual(metrics["baseline_fp"], 2)
        self.assertEqual(metrics["removed_fp"], 1)
        self.assertEqual(metrics["removed_tp"], 0)
        self.assertAlmostEqual(metrics["fp_removal_precision"], 1.0)
        self.assertAlmostEqual(metrics["fp_removal_recall"], 0.5)
        self.assertAlmostEqual(metrics["tp_damage_rate"], 0.0)
        self.assertEqual(metrics["modified_not_crossed"], 2)

    def test_zero_denominators_are_zero(self) -> None:
        metrics = summarize_method([], "legacy_score", 0.9)
        rate_keys = (
            "fp_removal_precision",
            "fp_removal_recall",
            "tp_damage_rate",
            "wrong_class_removal_rate",
            "localization_error_removal_rate",
            "pure_fp_removal_rate",
        )
        for key in rate_keys:
            self.assertEqual(metrics[key], 0.0)

        evidence = summarize_method(
            [],
            "evidence_veto_score",
            0.9,
            trigger_column="evidence_veto_triggered",
        )
        self.assertEqual(evidence["trigger_fp_precision"], 0.0)
        self.assertEqual(evidence["trigger_fp_recall"], 0.0)
        self.assertEqual(evidence["trigger_tp_damage_rate"], 0.0)


class SamplingTests(unittest.TestCase):
    def test_fixed_train_pool_is_order_and_cli_seed_independent(self) -> None:
        stems = [f"{index:05d}" for index in range(100)]
        pool_a = fixed_train_research_pool(stems, pool_size=20)
        pool_b = fixed_train_research_pool(list(reversed(stems)), pool_size=20)
        self.assertEqual(pool_a, pool_b)
        self.assertEqual(len(pool_a), 20)
        self.assertEqual(len(set(pool_a)), 20)

        selected_a = select_split_stems(
            stems, split="train", limit=10, seed=1, train_pool_size=20)
        selected_b = select_split_stems(
            stems, split="train", limit=10, seed=999, train_pool_size=20)
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(selected_a, pool_a[:10])

    def test_val_sample_is_seeded(self) -> None:
        stems = [f"{index:05d}" for index in range(100)]
        first = select_split_stems(stems, split="val", limit=20, seed=3)
        second = select_split_stems(stems, split="val", limit=20, seed=3)
        different = select_split_stems(stems, split="val", limit=20, seed=4)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


class ResumeTests(unittest.TestCase):
    def test_status_resume_and_csv_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "status.jsonl"
            events = [
                {
                    "run_signature": "sig",
                    "image_stem": "a",
                    "status": "completed",
                },
                {
                    "run_signature": "sig",
                    "image_stem": "b",
                    "status": "failed",
                },
                {
                    "run_signature": "other",
                    "image_stem": "c",
                    "status": "completed",
                },
            ]
            status_path.write_text(
                "".join(json.dumps(event) + "\n" for event in events)
                + "{partial",
                encoding="utf-8",
            )
            self.assertEqual(load_completed_images(status_path, "sig"), {"a"})

            fieldnames = (
                "split",
                "image_stem",
                "detection_index",
                "detector_score",
            )
            part_a = root / "part_a.csv"
            part_b = root / "part_b.csv"
            write_rows(
                part_a,
                fieldnames,
                [
                    {
                        "split": "train",
                        "image_stem": "a",
                        "detection_index": "0",
                        "detector_score": "0.8",
                    },
                    {
                        "split": "train",
                        "image_stem": "a",
                        "detection_index": "0",
                        "detector_score": "0.9",
                    },
                ],
            )
            write_rows(
                part_b,
                fieldnames,
                [
                    {
                        "split": "train",
                        "image_stem": "a",
                        "detection_index": "0",
                        "detector_score": "0.95",
                    },
                    {
                        "split": "val",
                        "image_stem": "b",
                        "detection_index": "1",
                        "detector_score": "0.7",
                    },
                ],
            )
            merged = root / "prediction_boxes.csv"
            self.assertEqual(
                merge_part_csvs([part_a, part_b], merged, fieldnames), 2)
            with merged.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            train_row = next(row for row in rows if row["split"] == "train")
            self.assertEqual(train_row["detector_score"], "0.95")


def write_rows(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class InferenceGuardTests(unittest.TestCase):
    def test_explicit_with_cga_false_when_supported(self) -> None:
        seen: dict[str, object] = {}

        def inference(model: object, image: str, *, with_cga: bool) -> str:
            seen["with_cga"] = with_cga
            return image

        result = inference_without_cga(object(), "image.png", inference)
        self.assertEqual(result, "image.png")
        self.assertIs(seen["with_cga"], False)

    def test_plain_api_requires_false_model_default(self) -> None:
        class Model:
            def simple_test(self, image: object, meta: object, with_cga: bool = False) -> None:
                del image, meta, with_cga

        def inference(model: object, image: str) -> str:
            del model
            return image

        self.assertEqual(
            inference_without_cga(Model(), "image.png", inference), "image.png")


if __name__ == "__main__":
    unittest.main()
