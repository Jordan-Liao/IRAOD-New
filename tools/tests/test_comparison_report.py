"""CPU regression of the real artifact consumers and seed-block statistics."""

import csv
import json
import os
from pathlib import Path
import pickle
import tempfile
import sys
import unittest
from zipfile import ZipFile

import numpy as np

from experiments.comparison.final_report import build_report, figures
from experiments.comparison.report_inputs import (
    CLASSES, EXPECTED_IMAGES, HISTORICAL_ROOT, collect_quantitative, keyed)
from experiments.comparison.report_qualitative import qualitative_evidence
from experiments.comparison.report_statistics import summarize
from experiments.comparison.result_completion import DOMAINS, write_json, visualize
from experiments.comparison.joint_tsne import joint_tsne
from tools.tests import test_aligned_roi_completion as roi_tests


DOCX_PYTHON = Path(os.environ.get("IRAOD_DOCX_PYTHON", sys.executable))


def numeric_rows():
    rows = []
    for ds, domains in DOMAINS.items():
        for j, domain in enumerate(domains):
            base = .5 if domain == "clean" else .2 + .02 * j
            rows.append({"dataset": ds, "domain": domain, "method": "A", "role": "source",
                         "seed": 42, "status": "complete", "mAP50": base})
            for method in "BCDEF":
                for seed, delta in zip((42, 43, 44), (.01, .02, .04)):
                    rows.append({
                        "dataset": ds, "domain": domain, "method": method, "role": "ema",
                        "seed": seed, "status": "complete",
                        "mAP50": base + delta * "ABCDEF".index(method),
                        "gpu_provenance": "single" if method in "BCD" else "DDP2",
                    })
    return rows


class SeedStatisticsTest(unittest.TestCase):
    def test_seed_first_aggregation_fixed_source_and_original_rpc(self):
        report = summarize(numeric_rows())
        blocks = [r for r in report["per_seed"]
                  if r["dataset"] == "RSAR" and r["method"] == "B"]
        np.testing.assert_allclose([r["mPC"] for r in blocks], [.29, .30, .32])
        np.testing.assert_allclose([r["delta_A"] for r in blocks], [.01, .02, .04])
        row = next(r for r in report["summary"]
                   if r["dataset"] == "RSAR" and r["method"] == "B" and r["metric"] == "mPC")
        self.assertEqual(row["n"], 3)
        self.assertAlmostEqual(row["mean"], np.mean([.29, .30, .32]))
        self.assertAlmostEqual(row["sample_std"], np.std([.29, .30, .32], ddof=1))
        self.assertAlmostEqual(blocks[0]["rPC_percent"], 58)
        self.assertAlmostEqual(blocks[0]["method_clean_normalized_percent"], 100 * .29 / .51)
        source = next(r for r in report["summary"]
                      if r["dataset"] == "RSAR" and r["method"] == "A" and r["metric"] == "mPC")
        self.assertEqual(source["n"], 1)
        self.assertIsNone(source["sample_std"])
        self.assertIsNone(source["bootstrap_ci95_low"])
        self.assertFalse(any(r["method"] == "A" for r in report["per_seed"]))
        test = next(r for r in report["paired_statistics"]
                    if r["dataset"] == "RSAR" and r["method"] == "B" and r["metric"] == "delta_A")
        self.assertEqual(test["sign_assignments"], 8)
        self.assertEqual(test["sign_flip_p"], .25)
        self.assertEqual(test["sign_flip_p_holm"], 1)
        self.assertEqual(test["df"], 2)

    def test_partial_blocks_do_not_shrink_domains_seeds_or_holm_family(self):
        rows = [r for r in numeric_rows() if not (
            r["dataset"] == "RSAR" and r["domain"] == "point_target"
            and r["method"] == "B" and r["seed"] == 44)]
        result = summarize(rows)
        block = next(r for r in result["per_seed"]
                     if r["dataset"] == "RSAR" and r["method"] == "B" and r["seed"] == 44)
        self.assertEqual(block["corruptions_observed"], 6)
        self.assertIsNone(block["mPC"])
        row = next(r for r in result["summary"]
                   if r["dataset"] == "RSAR" and r["method"] == "B" and r["metric"] == "mPC")
        self.assertEqual(row["n"], 2)
        self.assertEqual(row["status"], "incomplete")
        self.assertIsNone(row["mean"])
        self.assertIsNone(row["sample_std"])
        family = [r for r in result["paired_statistics"]
                  if r["family"] == "RSAR/ema/delta_A/B-F"]
        self.assertTrue(all(r["sign_flip_p_holm"] is None for r in family))
        self.assertTrue(all(r["holm_status"] == "family_incomplete" for r in family))

    def test_zero_variance_and_no_source_seed_duplication(self):
        rows = numeric_rows()
        for r in rows:
            if r["method"] != "A":
                r["mAP50"] = .3
        result = summarize(rows)
        test = result["paired_statistics"][0]
        self.assertIsNone(test["t_p"])
        self.assertIn("zero variance", test["t_na_reason"])
        self.assertEqual(test["sign_flip_p"], .25)
        source = next(r for r in rows if r["method"] == "A")
        with self.assertRaisesRegex(ValueError, "single seed42 source"):
            keyed([{**source, "seed": 43}])
        with self.assertRaisesRegex(ValueError, "Duplicate cell"):
            keyed([source, source])

    def test_figures_use_complete_statistics(self):
        with tempfile.TemporaryDirectory() as root:
            result = summarize(numeric_rows())
            result["roles"] = ["ema"]
            names = figures(result, Path(root))
            self.assertEqual(len(names), 4)
            self.assertTrue(all((Path(root) / name).stat().st_size > 0 for name in names))


class ArtifactConsumerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ds = "RSAR"
        self.eval = self.root / "eval_full_clean"
        self.eval.mkdir()
        self.checkpoint = {
            "dataset": self.ds, "domain": "clean", "method": "B", "seed": 42, "role": "ema",
            "path": "/owner/rsar/clean/seed_42/methods/B/work/iter_266_ema.pth",
            "selection": "final", "iteration": 266, "verified": True,
            "source_id": "fixture-source", "gpu_provenance": {"topology": "single"},
        }
        self.cell = {
            "dataset": self.ds, "domain": "clean", "method": "B", "seed": 42, "role": "ema",
            "eval_dir": str(self.eval), "eval_json": str(self.eval / "eval_fixture.json"),
            "prediction_image_ids": str(self.root / "prediction_image_ids.json"),
        }
        write_json(self.cell["eval_json"], {
            "config": "/owner/source_inference.py", "metric": {"mAP": .2446555644273758, "AP50": .245}})
        (self.eval / "eval_status").write_text(
            "eval_exit=0 2026-09-06T00:00:00+08:00 name=B domain=clean seed=42 "
            f"ema={self.checkpoint['path']}\n")
        (self.eval / "class_ap.txt").write_text(
            "| class | gts | dets | recall | ap |\n+---+---+---+---+---+\n"
            + "".join(f"| {c} | 1 | 1 | 0.500 | 0.123 |\n" for c in CLASSES[self.ds])
            + "| mAP | | | | 0.245 |\n{'mAP': 0.2446555644273758, 'AP50': 0.245}\n")
        count = EXPECTED_IMAGES[self.ds]
        write_json(self.cell["prediction_image_ids"], {"image_ids": [f"test_{i}" for i in range(count)]})
        per_class = [np.zeros((0, 6), dtype=np.float32) for _ in CLASSES[self.ds]]
        per_class[0] = np.array([[2, 3, 4, 5, .1, .15]], dtype=np.float32)
        with (self.eval / "predictions.pkl").open("wb") as stream:
            pickle.dump([per_class] * count, stream)
        (self.eval / "pred_count.txt").write_text(f"{count} expect={count}\n")
        self.manifest = {
            "schema": "iraod-comparison-report-v1", "roles": ["ema"],
            "cells": [self.cell], "checkpoints": [self.checkpoint],
            "source_ids": {"RSAR": "fixture-source", "DIOR": "fixture-dior-source"},
        }

    def test_real_eval_format_full_prediction_rows_and_csv_manifest(self):
        cells = self.root / "cells.csv"
        with cells.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.cell))
            writer.writeheader()
            writer.writerow(self.cell)
        self.manifest["cells"] = str(cells)
        raw, classes, predictions, roles = collect_quantitative(self.manifest)
        self.assertEqual(len(raw), 192)
        result = next(r for r in raw if r["status"] == "complete")
        self.assertEqual(result["mAP50"], .2446555644273758)
        self.assertNotEqual(result["mAP50"], .245)
        self.assertEqual(result["n_predictions"], 8538)
        self.assertEqual(result["n_post_nms_detections"], 8538)
        self.assertEqual(len(predictions), 8538)
        self.assertEqual(len(classes), 6)
        self.assertEqual(classes[0]["AP50"], .123)
        self.assertEqual(roles, ("ema",))
        self.assertEqual(predictions[-1]["image_id"], "test_8537")

    def test_last_failed_retry_missing_image_order_and_class_placeholder(self):
        with (self.eval / "eval_status").open("a") as stream:
            stream.write("eval_exit=1 retry\n")
        self.cell.pop("prediction_image_ids")
        (self.eval / "class_ap.txt").write_text("MISSING_CLASS_TABLE\n")
        raw, _, images, _ = collect_quantitative(self.manifest)
        result = next(r for r in raw if r.get("eval_dir"))
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("last_eval_status_failed_or_identity_mismatch", result["problems"])
        self.assertIn("missing_prediction_image_order", result["problems"])
        self.assertIn("missing_class_table", result["problems"])
        self.assertEqual(images, [])

    def test_locked_seed42_values_conflict_without_overwrite(self):
        old = self.root / "old_raw.csv"
        old.write_text("dataset,corruption,method,seed,ckpt_role,mAP50\nRSAR,clean,B,42,ema,0.25\n")
        self.manifest["historical_raw"] = [str(old)]
        raw, classes, predictions, _ = collect_quantitative(self.manifest)
        result = next(r for r in raw if r.get("eval_dir"))
        self.assertEqual(result["mAP50"], .25)
        self.assertEqual(result["eval_mAP50"], .2446555644273758)
        self.assertEqual(result["status"], "metric_conflict")
        self.assertTrue(all(r["status"] == "metric_conflict" for r in classes))
        self.assertTrue(all(r["cell_status"] == "metric_conflict" for r in predictions))
        self.assertIn(",0.25\n", old.read_text())

    def test_count_evidence_cannot_turn_partial_predictions_into_success(self):
        (self.eval / "pred_count.txt").write_text("8537 expect=8538\n")
        raw, _, _, _ = collect_quantitative(self.manifest)
        result = next(r for r in raw if r.get("eval_dir"))
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("prediction_count_not_full_test", result["problems"])
        with (self.eval / "predictions.pkl").open("wb") as stream:
            pickle.dump([], stream)
        with self.assertRaisesRegex(ValueError, "Prediction count differs"):
            collect_quantitative(self.manifest)

    def test_stale_class_table_does_not_match_new_eval_json(self):
        path = self.eval / "class_ap.txt"
        path.write_text(path.read_text().replace("'mAP': 0.2446555644273758", "'mAP': 0.2"))
        raw, _, _, _ = collect_quantitative(self.manifest)
        result = next(r for r in raw if r.get("eval_dir"))
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("class_table_eval_metric_mismatch", result["problems"])

    def test_report_docx_partial_not_final_success(self):
        self.assertTrue(DOCX_PYTHON.is_file(), "Use the supplied existing python-docx environment")
        manifest = self.root / "report_input.json"
        write_json(manifest, self.manifest)
        result = build_report(manifest, self.root / "report", str(DOCX_PYTHON))
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["quantitative_complete_cells"], 1)
        self.assertEqual(result["qualitative_coverage"]["roi_identified_image_roles"], 0)
        self.assertEqual(result["qualitative_coverage"]["embeddings_complete"], 0)
        self.assertEqual(len(result["embedding_index"]), 24)
        with ZipFile(self.root / "report/comparison_report_cn.docx") as archive:
            text = archive.read("word/document.xml").decode()
        self.assertIn("partial", text)
        self.assertIn("固定 RSAR32/DIOR16", text)
        self.assertIn("低功效", text)
        status = json.loads((self.root / "report/build_status.json").read_text())
        self.assertEqual(status["result_status"], "partial")
        self.assertEqual(status["report_build"], "complete")
        for name in ("raw_results.csv", "dior_raw_results.csv",
                     "per_class_summary.csv", "dior_per_class.csv"):
            self.assertEqual((self.root / "report/historical" / name).read_bytes(),
                             (HISTORICAL_ROOT / name).read_bytes())


class QualitativeBoundaryTest(unittest.TestCase):
    def test_capture_retains_post_nms_scores_below_visual_threshold(self):
        fixture = roi_tests.AlignedMappingTest()
        fixture.setUp()
        fixture.cfg.max_per_img = 5
        arrays, _, _ = fixture.capture(fixture.features, fixture.rois, fixture.delta)
        self.assertEqual(len(arrays["features"]), 5)
        self.assertLess(float(arrays["scores"][-1]), .3)
        self.assertAlmostEqual(float(arrays["scores"][-1]), .25)
        self.assertEqual(int(arrays["proposal_indices"][-1]), 2)

    def test_actual_subset_and_embedding_evidence_are_not_full_test(self):
        fixture = roi_tests.CompletionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        mapping = roi_tests.AlignedMappingTest()
        mapping.setUp()
        arrays, _, _ = mapping.capture(mapping.features, mapping.rois, mapping.delta)
        runs = [r for r in fixture.plan["runs"]
                if r["dataset"] == "DIOR" and r["domain"] == "clean"
                and r["role"] in ("source", "ema")]
        for run in runs:
            fixture.export_fixture(run, arrays)
        visualize(runs[0])
        target = fixture.root / "embedding"
        joint_tsne(fixture.plan, "DIOR", "clean", "ema", target, cap=5, perplexity=2)
        plan = fixture.root / "plan.json"
        write_json(plan, fixture.plan)
        rows, index, coverage = qualitative_evidence({
            "qualitative_plan": str(plan), "embeddings": [
                {"dataset": "DIOR", "domain": "clean", "comparison": "ema", "directory": str(target)}]})
        self.assertEqual(len(rows), 3520)
        self.assertEqual(coverage["roi_complete"], 96)
        self.assertEqual(coverage["vis_complete"], 16)
        self.assertEqual(coverage["embeddings_complete"], 1)
        self.assertEqual(len(index), 24)
        self.assertIn("not supported", coverage["full_test_roi"])
        (target / "DIOR_clean_ema_F_ema.pdf").unlink()
        _, index, coverage = qualitative_evidence({
            "qualitative_plan": str(plan), "embeddings": [
                {"dataset": "DIOR", "domain": "clean", "comparison": "ema", "directory": str(target)}]})
        self.assertEqual(coverage["embeddings_complete"], 0)


if __name__ == "__main__":
    unittest.main()
