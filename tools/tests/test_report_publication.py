"""Final consumer boundaries: accepted evidence reuse and exact gated publication."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from experiments.comparison.final_report import build_report, render_report, write_csv
from experiments.comparison.publish_report import (
    EXPERIMENT_FILES, REQUIRED_RESULTS, admit_final, stage_publication)
from experiments.comparison.report_inputs import CLASSES, HISTORICAL_ROOT, historical_paths, read_rows
from experiments.comparison.report_qualitative import qualitative_evidence
from experiments.comparison.report_statistics import summarize
from experiments.comparison.result_completion import SCHEMA, read_json, write_json
from tools.tests.test_aligned_roi_completion import CompletionTest
from tools.tests.test_comparison_report import DOCX_PYTHON, numeric_rows


class AcceptedEvidenceReuseTest(unittest.TestCase):
    def setUp(self):
        fixture = CompletionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.plan = fixture.root, fixture.plan
        self.plan_path = self.root / "plan.json"
        write_json(self.plan_path, self.plan)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        groups = []
        for run in self.plan["runs"]:
            root = Path(run["out_dir"])
            root.mkdir(parents=True)
            (root / "visualizations").mkdir()
            write_json(root / "index.json", {
                "schema": SCHEMA, "status": "complete", "run": run,
                "records": [{"image_id": i, "n_detections": 2, "feature_file": i + ".npz"}
                            for i in run["image_ids"]]})
            write_json(root / "visualizations/index.json", {
                "schema": SCHEMA, "status": "complete", "run": run,
                "images": [{"image_id": i, "file": i + ".png"} for i in run["visualization_image_ids"]]})
            groups.append({
                "run_id": run["run_id"], "scope": "full_test", "status": "complete",
                "inspected": True, "out_dir": run["out_dir"],
                "expected_images": len(run["image_ids"]), "validated_images": len(run["image_ids"]),
                "detection_rows": 2 * len(run["image_ids"]),
                "visualizations_complete": len(run["visualization_image_ids"]),
            })
        embeddings = []
        for ds, domain in sorted({(r["dataset"], r["domain"]) for r in self.plan["runs"]}):
            for role in ("ema", "student"):
                root = self.root / "embeddings" / ds / domain / role
                root.mkdir(parents=True)
                runs = sorted((r for r in self.plan["runs"]
                               if r["dataset"] == ds and r["domain"] == domain
                               and (r["method"] == "A" or r["role"] == role)),
                              key=lambda r: r["method"])
                write_json(root / "protocol.json", {
                    "schema": SCHEMA, "runs": runs, "sampling_seed": 42,
                    "tsne_parameters": {"random_state": 42}, "points_per_method": 4})
                embeddings.append({
                    "dataset": ds, "domain": domain, "comparison": role, "directory": str(root),
                    "status": "complete", "n_points": 24, "problems": []})
        write_csv(self.evidence / "roi_group_summary.csv", groups, ())
        write_csv(self.evidence / "embedding_index.csv", embeddings, ())
        (self.evidence / "roi_vis_coverage.csv").write_text("accepted audit fixture\n")
        write_json(self.evidence / "report_input_fragment.json", {
            "qualitative_plan": str(self.plan_path),
            "embeddings": [{k: e[k] for k in ("dataset", "domain", "comparison", "directory")}
                           for e in embeddings]})
        self.summary = {
            "schema": "iraod-qualitative-completion-summary-v1",
            "qualitative_status": "complete", "full_test_roi": "complete",
            "plan": str(self.plan_path),
            "roi_image_manifest": str(self.evidence / "roi_vis_coverage.csv"),
            "roi_group_summary": str(self.evidence / "roi_group_summary.csv"),
            "embedding_index": str(self.evidence / "embedding_index.csv"),
            "report_input_fragment": str(self.evidence / "report_input_fragment.json"),
            "roi_expected_image_roles": 3872, "roi_identified_image_roles": 3872,
            "roi_complete": 3872, "roi_detection_rows": 7744,
            "roi_inspected_groups": 132, "roi_complete_groups": 132,
            "vis_expected_image_roles": 3520, "vis_complete": 3520,
            "embeddings_expected": 24, "embeddings_complete": 24, "embedding_points": 576,
        }
        write_json(self.evidence / "qualitative_summary.json", self.summary)
        self.manifest = {"qualitative_plan": str(self.plan_path),
                         "qualitative_evidence": str(self.evidence)}

    def test_reuse_does_not_load_npz_or_rescan_roi_csv(self):
        with patch("numpy.load", side_effect=AssertionError("NPZ must not be opened")), \
                patch("experiments.comparison.report_qualitative.collect",
                      side_effect=AssertionError("collector must not be rerun")):
            rows, embeddings, coverage = qualitative_evidence(self.manifest)
        self.assertEqual(rows, [])
        self.assertEqual(len(embeddings), 24)
        self.assertEqual(coverage["roi_detection_rows"], 7744)
        self.assertEqual(len(coverage["visualization_index"]), 3520)
        self.assertNotIn("quantitative_status", coverage)

    def test_count_partial_and_checkpoint_binding_mismatches_are_rejected(self):
        self.summary["roi_detection_rows"] += 1
        write_json(self.evidence / "qualitative_summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "summary counts"):
            qualitative_evidence(self.manifest)
        self.summary["roi_detection_rows"] -= 1
        self.summary["qualitative_status"] = "partial"
        write_json(self.evidence / "qualitative_summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "does not bind"):
            qualitative_evidence(self.manifest)
        self.summary["qualitative_status"] = "complete"
        write_json(self.evidence / "qualitative_summary.json", self.summary)
        self.plan["runs"][0]["checkpoint"] = "/different/source.pth"
        write_json(self.plan_path, self.plan)
        with self.assertRaisesRegex(ValueError, "identity/completion mismatch"):
            qualitative_evidence(self.manifest)

    def test_scoped_reuse_is_not_allowed(self):
        with self.assertRaisesRegex(ValueError, "scoped"):
            qualitative_evidence({**self.manifest, "inspect_roi_run_ids": []})


class PublicationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = numeric_rows()
        self.classes, vis, checkpoints = [], [], []
        for row in self.raw:
            row.update(checkpoint=f"/fixture/{row['dataset']}/{row['domain']}/{row['method']}/{row['seed']}.pth",
                       effective_training_code_sha="fixture-producer",
                       evaluation_code_sha="fixture-evaluator", source_id="fixture-source",
                       prediction_image_ids="/fixture/native-sidecar.json")
            base = {k: row[k] for k in ("dataset", "domain", "method", "role", "seed")}
            checkpoints.append({**base, "path": row["checkpoint"], "verified": True})
            self.classes.extend({**base, "class_name": c, "AP50": row["mAP50"],
                                 "status": "complete", "evidence": "/fixture/class_ap.txt"}
                                for c in CLASSES[row["dataset"]])
            if row["seed"] == 42:
                for role in (("source",) if row["method"] == "A" else ("ema", "student")):
                    vis.extend({**base, "role": role, "checkpoint": row["checkpoint"],
                                "corruption": row["domain"], "image": str(i),
                                "config": "/fixture/config.py", "show_dir": "/fixture/vis"}
                               for i in range(32 if row["dataset"] == "RSAR" else 16))
        self.coverage = {
            "roi_complete_groups": 132, "roi_complete": 1267816,
            "roi_expected_image_roles": 1267816, "roi_detection_rows": 18468211,
            "vis_complete": 3520, "embeddings_complete": 24, "full_test_roi": "complete",
            "roi_group_index": [], "visualization_index": vis,
            "roi_image_manifest": "/remote/accepted/roi_vis_coverage.csv",
            "evidence_reuse": "/remote/accepted/qualitative_summary.json",
        }
        self.terminal = {
            "status": "complete", "active": [], "cells": [
                {"cell": {k: r[k] for k in ("dataset", "domain", "seed", "method")},
                 "train_requested": r["method"] != "A" and (r["seed"] != 42 or r["domain"] == "clean"),
                 "train": "complete", "eval": "complete", "reasons": []} for r in self.raw]}
        self.report = {
            "status": "declared_scopes_complete", "roles": ["ema"],
            "quantitative_expected_cells": 192, "quantitative_complete_cells": 192,
            "raw_results": self.raw, "per_class": self.classes,
            "qualitative_coverage": self.coverage,
        }
        self.manifest = {"schema": "iraod-comparison-report-v1", "qualitative_evidence": "/fixture",
                         "source_ids": {"RSAR": "fixture-source", "DIOR": "fixture-source"},
                         "checkpoints": checkpoints}

    def test_publication_requires_terminal_cells_classes_and_provenance(self):
        admit_final(self.report, self.terminal)
        for field, replacement in (("status", "partial"), ("quantitative_complete_cells", 191),
                                   ("roles", ["student"]), ("per_class", self.classes[:-1])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                admit_final({**self.report, field: replacement}, self.terminal)
        terminal = copy.deepcopy(self.terminal)
        terminal["cells"][1]["train_requested"] = False
        with self.assertRaisesRegex(ValueError, "130/130"):
            admit_final(self.report, terminal)
        self.raw[0]["effective_training_code_sha"] = None
        with self.assertRaisesRegex(ValueError, "provenance"):
            admit_final(self.report, self.terminal)

    def test_stage_exact_outputs_from_portable_metadata_and_preserve_original_bytes(self):
        manifest = self.root / "manifest.json"
        write_json(manifest, self.manifest)
        source = self.root / "report"
        with patch("experiments.comparison.final_report.collect_quantitative",
                   return_value=(self.raw, self.classes, [], ("ema",))), \
                patch("experiments.comparison.final_report.qualitative_evidence",
                      return_value=([], [], self.coverage)):
            report = build_report(manifest, source, metadata_only=True)
        self.assertEqual(report["status"], "declared_scopes_complete")
        state = self.root / "terminal.json"
        write_json(state, self.terminal)
        with self.assertRaisesRegex(ValueError, "metadata-only"):
            stage_publication(source, state, self.root / "rejected", str(DOCX_PYTHON))
        self.assertFalse((self.root / "rejected").exists())
        # Actual local rendering uses only the portable numeric evidence, not remote paths.
        render_report(report, source, str(DOCX_PYTHON))
        out = stage_publication(source, state, self.root / "delivery", str(DOCX_PYTHON))
        results = out / "results/paper_comparison"
        self.assertTrue(all((results / name).stat().st_size for name in REQUIRED_RESULTS))
        self.assertTrue(all((out / "experiments/comparison" / name).stat().st_size
                            for name in EXPERIMENT_FILES))
        for name in ("raw_results.csv", "dior_raw_results.csv", "per_class_summary.csv",
                     "dior_per_class.csv", "IRAOD_full_comparison_report_cn.docx"):
            self.assertEqual((results / "historical/seed42" / name).read_bytes(),
                             (HISTORICAL_ROOT / name).read_bytes())
        with patch("experiments.comparison.report_inputs.HISTORICAL_ROOT", results):
            self.assertTrue(all("/historical/seed42/" in p for p in historical_paths({})))
        self.assertEqual(len(read_rows(results / "raw_results.csv")), 192)
        self.assertEqual(len(read_rows(results / "per_class_summary.csv")), 2048)
        self.assertEqual(len(read_rows(results / "visualization_manifest.csv")), 3520)
        self.assertEqual((results / "per_class.csv").read_bytes(),
                         (results / "per_class_summary.csv").read_bytes())
        self.assertFalse((results / "roi_vis_coverage.csv").exists())
        self.assertFalse((results / "prediction_image_coverage.csv").exists())
        table = (results / "main_table.tex").read_text()
        self.assertIn(r"\mathbf{61.67}", table)
        self.assertIn("Source-Free", table)
        self.assertIn("sample std", table)
        data_lines = [line for line in table.splitlines()
                      if any(line.startswith(method + " ") for method in "ABCDEF")]
        self.assertEqual(len(data_lines), 6)
        self.assertTrue(all(line.count(" & ") == 10 for line in data_lines))
        self.assertIn("Not run", (results / "oracle_table.tex").read_text())
        self.assertIn("no extra ablation", (results / "ablation_table.tex").read_text())
        with ZipFile(results / "IRAOD_full_comparison_report_cn.docx") as docx:
            text = docx.read("word/document.xml").decode()
        self.assertIn("Source-Free", text)
        self.assertIn("Recovery", text)
        self.assertIn("Target-supervised", text)
        self.assertEqual(read_json(out / "publication.json")["status"], "complete_staged_not_installed")

    def test_recovery_preserves_negative_and_undefined_values(self):
        rows = numeric_rows()
        row = next(r for r in rows if r["dataset"] == "RSAR" and r["domain"] == "chaff"
                   and r["method"] == "B" and r["seed"] == 42)
        row["mAP50"] = .1
        recovery = summarize(rows)["recovery"]
        result = next(r for r in recovery if all(r[k] == row[k] for k in (
            "dataset", "domain", "method", "role", "seed")))
        self.assertAlmostEqual(result["recovery_ratio"], (.1 - .22) / (.5 - .22))
        source = next(r for r in rows if r["dataset"] == "RSAR" and r["domain"] == "chaff"
                      and r["method"] == "A")
        source["mAP50"] = .5
        recovery = summarize(rows)["recovery"]
        self.assertTrue(all(r["recovery_ratio"] is None for r in recovery
                            if r["dataset"] == "RSAR" and r["domain"] == "chaff"))


if __name__ == "__main__":
    unittest.main()
