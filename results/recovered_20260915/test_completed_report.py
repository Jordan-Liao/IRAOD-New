"""Regressions against the first actual corrected-checkpoint native result."""

from copy import deepcopy
import json
import unittest

import build_completed_report as report


class CompletedReportTests(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads((report.ROOT / "recovered_results.json").read_text())
        self.identity = "RSAR/point_target/44/IRG/student"
        self.record = next(
            record for record in self.payload["records"]
            if record["native"]["canonical_key"] == self.identity
        )
        self.model = next(
            model for model in self.payload["models"]
            if model["canonical_key"] == self.identity
        )
        raw = report.previous.load_csv(report.PREVIOUS, "raw_metrics.csv")
        self.original = next(row for row in raw if report.previous.canonical(row) == self.identity)
        self.source = next(
            row for row in raw
            if (row["dataset"], row["domain"], row["method"]) == ("RSAR", "point_target", "A")
        )
        self.names = {"ship", "aircraft", "car", "tank", "bridge", "harbor"}

    def apply(self, record=None, original=None, model=None):
        record = self.record if record is None else record
        return report.recovered_rows(
            self.original if original is None else original,
            record["native"], record["terminal"], record["post_binder"],
            self.model if model is None else model,
            record["class_ap_text"], self.source, self.names,
            self.payload["metric_cutoff_utc"],
        )

    def test_actual_native_precision_and_class_values(self):
        before = deepcopy(self.original)
        row, classes = self.apply()
        self.assertEqual(row["mAP50"], 0.2135608196258545)
        self.assertNotEqual(row["mAP50"], 0.214)
        self.assertEqual(row["n_images"], 8538)
        self.assertEqual(
            {entry["class_name"]: entry["AP50"] for entry in classes},
            dict(ship="0.484", aircraft="0.241", car="0.274",
                 tank="0.091", bridge="0.096", harbor="0.095"),
        )
        self.assertEqual(self.original, before)
        self.assertEqual(self.original["mAP50"], "")

    def test_does_not_replace_a_valid_historical_metric(self):
        original = {**self.original, "mAP50": "0.123"}
        with self.assertRaisesRegex(ValueError, "Cannot replace a valid metric"):
            self.apply(original=original)

    def test_incorrect_or_incomplete_evidence_is_not_promoted(self):
        mutations = [
            ("role", lambda r: r["native"]["execution"].update(role="ema")),
            ("checkpoint", lambda r: r["native"]["execution"].update(checkpoint="iter_266_ema.pth")),
            ("source", lambda r: r["native"]["execution"].update(source_id="wrong-source")),
            ("revision", lambda r: r["native"]["execution"].update(training_code_sha="old-code")),
            ("test_count", lambda r: r["native"].update(pred_count=8537)),
            ("roi_count", lambda r: r["post_binder"].update(roi_n_images=8537)),
            ("roi_status", lambda r: r["terminal"].update(status="TERMINAL_FAILED_VALIDATION")),
            ("nan", lambda r: r["native"]["metric"]["metric"].update(mAP=float("nan"))),
            ("class_coverage", lambda r: r.update(class_ap_text="")),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                record = deepcopy(self.record)
                mutate(record)
                with self.assertRaises(ValueError):
                    self.apply(record)

    def test_partial_bundle_cannot_generate_a_complete_report(self):
        payload = {**self.payload, "records": [self.record], "models": [self.model]}
        with self.assertRaisesRegex(ValueError, "Exactly the twelve"):
            report.assemble(payload)

    def test_complete_bundle_preserves_every_frozen_valid_row(self):
        raw, classes = report.assemble(self.payload)
        old = {
            report.previous.canonical(row): row
            for row in report.previous.load_csv(report.PREVIOUS, "raw_metrics.csv")
        }
        by_key = {report.previous.canonical(row): row for row in raw}
        self.assertEqual(len(raw), 912)
        self.assertEqual(len(classes), 9728)
        for identity, row in old.items():
            if row["mAP50"] != "":
                self.assertEqual(by_key[identity], row)
        for record in self.payload["records"]:
            native = record["native"]
            self.assertEqual(
                by_key[native["canonical_key"]]["mAP50"],
                native["metric"]["metric"]["mAP"],
            )

    def test_delivery_requires_exact_membership_and_provider_proof(self):
        delivery = json.loads((report.ROOT / "delivery_manifest.json").read_text())
        self.assertEqual(report.validate_delivery(delivery)["provider_files"], 699)
        changed = deepcopy(delivery)
        changed["corrected_test_roi12"]["provider_receipts"][0]["provider_size"] += 1
        with self.assertRaisesRegex(ValueError, "Provider size/whole-MD5"):
            report.validate_delivery(changed)
        changed = deepcopy(delivery)
        changed["corrected_test_roi12"]["keys"][0] = changed["canonical_roi720"]["keys"][0]
        with self.assertRaisesRegex(ValueError, "exact original 732"):
            report.validate_delivery(changed)

    def test_same_count_cannot_hide_a_missing_experiment(self):
        raw, classes = report.assemble(self.payload)
        self.assertEqual(report.validate_coverage(raw, classes)["missing_or_extra_keys"], 0)
        changed = deepcopy(raw)
        changed[0]["domain"] = "not_in_the_approved_matrix"
        with self.assertRaisesRegex(ValueError, "Canonical TEST matrix"):
            report.validate_coverage(changed, classes)


if __name__ == "__main__":
    unittest.main()
