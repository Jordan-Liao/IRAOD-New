"""Producer-field correction is explicit and cannot change valid prediction evidence."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.comparison.source_provenance import annotate_producer, read_source_producer
from experiments.comparison.report_inputs import HISTORICAL_ROOT, collect_quantitative
from tools.tests import test_comparison_report as report_tests


ACTUAL = "b474aaa5f7718746215ce13beceb071617bb823e"
RECORDED = "0f98a48b539f9260055fc305784ffbc924f50451"


class SourceProvenanceTest(unittest.TestCase):
    def test_real_consumer_keeps_source_predictions_and_recorded_field_unchanged(self):
        fixture = report_tests.ArtifactConsumerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.cell.update(method="A", role="source")
        checkpoint = "/fixture/rsar_source/train/epoch_100.pth"
        fixture.checkpoint.update(method="A", role="source", path=checkpoint, selection="source")
        sidecar = Path(fixture.cell["prediction_image_ids"])
        original = json.loads(sidecar.read_text())
        original.update(checkpoint=checkpoint, training_code_sha=RECORDED)
        sidecar.write_text(json.dumps(original))
        before = sidecar.read_bytes()
        with (HISTORICAL_ROOT / "per_class_summary.csv").open(newline="") as stream:
            source_rows = [r for r in csv.DictReader(stream)
                           if r["corruption"] == "clean" and r["method"] == "A"]
        map_value = float(source_rows[0]["mAP50"])
        Path(fixture.cell["eval_json"]).write_text(json.dumps({
            "config": "/owner/source_inference.py", "metric": {"mAP": map_value}}))
        (fixture.eval / "eval_status").write_text(
            f"eval_exit=0 name=A domain=clean seed=42 ema={checkpoint}\n")
        (fixture.eval / "class_ap.txt").write_text(
            "| class | gts | dets | recall | ap |\n" +
            "".join(f"| {r['class_name']} | 1 | 1 | 0.5 | {r['AP50']} |\n" for r in source_rows))
        record = fixture.root / "source/reproducibility/git_commit.txt"
        record.parent.mkdir(parents=True)
        record.write_text(ACTUAL + "\n")
        fixture.manifest["source_provenance"] = {
            "RSAR": read_source_producer(checkpoint, "fixture-source", record,
                                        fixture.root / "snapshot/source_git.txt")}
        raw, _, predictions, _ = collect_quantitative(fixture.manifest)
        row = next(r for r in raw if r["status"] == "complete")
        self.assertEqual(row["method"], "A")
        self.assertEqual(row["mAP50"], map_value)
        self.assertEqual(row["training_code_sha"], RECORDED)
        self.assertEqual(row["recorded_training_code_sha"], RECORDED)
        self.assertEqual(row["actual_source_producer_sha"], ACTUAL)
        self.assertEqual(row["effective_training_code_sha"], ACTUAL)
        self.assertFalse(row["producer_metadata_correction"]["sidecar_modified"])
        self.assertEqual(len(predictions), 8538)
        self.assertEqual(sidecar.read_bytes(), before)

    def test_adaptation_producer_is_not_replaced_by_source_producer(self):
        row = {"method": "B", "training_code_sha": RECORDED, "status": "complete", "mAP50": .4}
        result = annotate_producer(row, {"actual_source_producer_sha": ACTUAL})
        self.assertEqual(result["effective_training_code_sha"], RECORDED)
        self.assertEqual(result["status"], "complete")
        self.assertNotIn("producer_metadata_correction", result)

    def test_unknown_source_producer_stays_null_not_adaptation_git(self):
        with tempfile.TemporaryDirectory() as directory:
            source = read_source_producer("/source.pth", "source-weight-id",
                                          Path(directory) / "missing-record")
            row = {"method": "A", "checkpoint": "/source.pth", "source_id": "source-weight-id",
                   "training_code_sha": RECORDED, "status": "complete"}
            result = annotate_producer(row, source)
            self.assertIsNone(result["actual_source_producer_sha"])
            self.assertIsNone(result["effective_training_code_sha"])
            self.assertEqual(result["training_code_sha"], RECORDED)
            self.assertEqual(result["producer_metadata_status"], "source_producer_unknown")


if __name__ == "__main__":
    unittest.main()
