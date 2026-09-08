"""Qualitative completion stays independent; per-class additions preserve old values."""

import csv
from pathlib import Path
import tempfile
import unittest

from experiments.comparison.complete_qualitative import collect_completion
from experiments.comparison.report_inputs import historical_paths
from experiments.comparison.result_completion import write_json
from tools.complete_rsar_per_class import (
    CLASSES, CORRUPTIONS, append_ap, canonical_maps)
from tools.tests import test_aligned_roi_completion as roi_tests


class FinalQualitativeIntegrationTest(unittest.TestCase):
    def test_partial_qualitative_summary_cannot_claim_quantitative_completion(self):
        fixture = roi_tests.CompletionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan_path = fixture.root / "plan.json"
        write_json(plan_path, fixture.plan)
        summary = collect_completion(plan_path, fixture.root / "embeddings",
                                     fixture.root / "complete")
        self.assertEqual(summary["qualitative_status"], "partial")
        self.assertFalse(summary["all_eight_items_complete"])
        self.assertEqual(summary["quantitative_status"], "pending_separate_evidence")
        self.assertEqual(summary["roi_complete"], 0)
        self.assertEqual(summary["roi_expected_image_roles"], 3872)
        self.assertTrue(Path(summary["roi_image_manifest"]).exists())

    def test_append_exactly_180_rows_without_changing_existing_bytes_or_map(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            table = root / "per_class_summary.csv"
            frozen = Path(historical_paths({})[0]).parent
            # The original 78 rows stay at the start after the production append.
            table.write_bytes(b"".join(
                (frozen / "per_class_summary.csv").read_bytes().splitlines(keepends=True)[:79]))
            original = table.read_bytes()
            maps = canonical_maps(frozen / "raw_results.csv")
            records = [{
                "dataset": "RSAR", "corruption": domain, "method": method,
                "seed": 42, "ckpt_role": "ema", "observed_mAP50": float(maps[domain, method]),
                "classes": [{"class_name": c, "AP50": "0.123"} for c in CLASSES["RSAR"]],
            } for domain in CORRUPTIONS for method in "BCDEF"]
            evidence = root / "evidence.json"
            write_json(evidence, {"schema": "iraod-rsar-nonchaff-perclass-evidence-v1",
                                  "records": records})
            self.assertEqual(append_ap(evidence, table, frozen / "raw_results.csv"), 180)
            self.assertTrue(table.read_bytes().startswith(original))
            with table.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 258)
            for row in rows[78:]:
                self.assertEqual(row["mAP50"], maps[row["corruption"], row["method"]])
                self.assertEqual(row["AP50"], "0.123")
            self.assertEqual(append_ap(evidence, table, frozen / "raw_results.csv"), 0)
            records.pop()
            write_json(evidence, {"schema": "iraod-rsar-nonchaff-perclass-evidence-v1",
                                  "records": records})
            before = table.read_bytes()
            with self.assertRaisesRegex(ValueError, "All 30"):
                append_ap(evidence, table, frozen / "raw_results.csv")
            self.assertEqual(before, table.read_bytes())


if __name__ == "__main__":
    unittest.main()
