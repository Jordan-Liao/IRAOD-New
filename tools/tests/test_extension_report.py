"""Synthetic native-format CPU fixtures; no detector imports or model execution."""

import copy
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from experiments.comparison.collect_extension_report import collect
from experiments.comparison.final_report import build_report, extension_tables, figures
from experiments.comparison.report_inputs import (
    CLASSES, FINAL_ITERATION, SFYOLO_FINAL_ITERATION, collect_quantitative, keyed)
from experiments.comparison.report_statistics import (
    EXTENSION_METHODS, comparison_groups, declared_methods, summarize)
from experiments.comparison.result_completion import DOMAINS, read_json, write_json
from tools.prediction_export import IMAGE_ORDER_ORIGIN, IMAGE_ORDER_SCHEMA


def numeric_rows(methods):
    rows = []
    for ds, domains in DOMAINS.items():
        for domain in domains:
            baseline = .5 if domain == "clean" else .25
            rows.append(dict(dataset=ds, domain=domain, method="A", role="source",
                             seed=42, status="complete", mAP50=baseline))
            for method in methods[1:]:
                for seed, delta in zip((42, 43, 44), (.01, .02, .04)):
                    rows.append(dict(dataset=ds, domain=domain, method=method, role="ema",
                                     seed=seed, status="complete", mAP50=baseline + delta))
    return rows


class ExtensionStatisticsTest(unittest.TestCase):
    def test_latex_keeps_budgets_separate_and_missing_values_explicit(self):
        methods = ["A", "IRG", "LPLD", "SFYOLO", "LoRA-CGA"]
        rows = [r for r in numeric_rows(methods) if r["method"] != "LPLD"]
        report = {"methods": methods, "roles": ["ema"], **summarize(rows, methods=methods)}
        with tempfile.TemporaryDirectory() as directory:
            names = extension_tables(report, directory)
            self.assertEqual(len(names), 3)
            common = (Path(directory) / "common_one_epoch_ema_table.tex").read_text()
            extended = (Path(directory) / "extended_two_epoch_TAM_ema_table.tex").read_text()
            oracle = (Path(directory) / "target_supervised_appendix_ema_table.tex").read_text()
            self.assertIn(r"\toprule", common)
            self.assertIn("LPLD & -- & --", common)
            self.assertNotIn("SFYOLO", common)
            self.assertNotIn("LoRA-CGA", common)
            self.assertIn("SFYOLO", extended)
            self.assertNotIn("IRG", extended)
            self.assertIn("LoRA-CGA", oracle)

    def test_explicit_methods_default_and_no_inferred_hypotheses(self):
        self.assertEqual(declared_methods(), tuple("ABCDEF"))
        for invalid in (["IRG"], ["A", "DRU"], ["A", "G_oracle"], ["A", "IRG", "IRG"]):
            with self.assertRaises(ValueError):
                declared_methods(invalid)
        row = dict(dataset="RSAR", domain="clean", method="IRG", seed=42, role="ema")
        with self.assertRaises(ValueError):
            keyed([row])
        self.assertEqual(len(keyed([row], ["A", "IRG"])), 1)
        with self.assertRaises(ValueError):
            keyed([{**row, "method": "A", "role": "source", "seed": 43}], ["A", "IRG"])
        with self.assertRaises(ValueError):
            keyed([row, row], ["A", "IRG"])

    def test_planned_family_waits_and_sfyolo_oracle_stay_separate(self):
        methods = ["A", "IRG", "LPLD", "SFYOLO", "LoRA-CGA", "LoRA-CGA+VLST"]
        rows = numeric_rows(methods)
        rows = [r for r in rows if r["method"] not in ("LPLD", "LoRA-CGA+VLST")]
        stats = summarize(rows, methods=methods)
        tests = stats["paired_statistics"]
        irg = [r for r in tests if r["method"] == "IRG"]
        self.assertTrue(all(r["holm_status"] == "family_incomplete" for r in irg))
        self.assertTrue(all(r["sign_flip_p_holm"] is None for r in irg))
        sfyolo = [r for r in tests if r["method"] == "SFYOLO"]
        self.assertTrue(all(r["holm_status"] == "complete_family" for r in sfyolo))
        self.assertTrue(all(r["family"].endswith("/extended_two_epoch_TAM") for r in sfyolo))
        oracle = [r for r in tests if r["method"] == "LoRA-CGA"]
        self.assertTrue(all(r["holm_status"] == "family_incomplete" for r in oracle))
        self.assertTrue(all(r["family"].endswith("/target_supervised_appendix") for r in oracle))
        self.assertEqual(len([r for r in stats["per_domain"] if r["method"] == "A"]), 12)
        self.assertFalse(any(r["method"] == "A" for r in stats["per_seed"]))

    def test_legacy_five_method_statistics_unchanged(self):
        stats = summarize(numeric_rows("ABCDEF"))
        self.assertEqual(len(stats["paired_statistics"]), 20)
        self.assertTrue(all(t["family"].endswith("/B-F") for t in stats["paired_statistics"]))
        self.assertTrue(all(t["holm_status"] == "complete_family" for t in stats["paired_statistics"]))
        self.assertTrue(all(t["sign_flip_p"] == .25 for t in stats["paired_statistics"]))
        self.assertFalse(any("comparison_group" in r for r in stats["summary"]))
        partial = [r for r in numeric_rows("ABCDEF") if r["method"] != "F"]
        self.assertTrue(all(t["holm_status"] == "family_incomplete"
                            for t in summarize(partial)["paired_statistics"]))


class ExtensionCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Small synthetic native inference fixtures. Production counts remain8538/11738.
        self.counts = patch.dict("experiments.comparison.report_inputs.EXPECTED_IMAGES",
                                 {"RSAR": 2, "DIOR": 2})
        self.counts.start()
        self.addCleanup(self.counts.stop)
        self.history = patch("experiments.comparison.report_inputs.historical_paths", return_value=[])
        self.history.start()
        self.addCleanup(self.history.stop)
        render_history = patch("experiments.comparison.final_report.historical_paths", return_value=[])
        render_history.start()
        self.addCleanup(render_history.stop)
        self.sources = {}
        core_rows, checkpoints = [], []
        for ds, domains in DOMAINS.items():
            source = self.root / ds / "source.pth"
            source.parent.mkdir()
            source.write_bytes(b"source fixture")
            self.sources[ds] = str(source)
            for domain in domains:
                cell, checkpoint = self.artifacts(ds, domain, "A", 42, "source", source)
                core_rows.append({**cell, "status": "complete"})
                checkpoints.append(checkpoint)
        self.core = self.root / "core.json"
        write_json(self.core, {
            "status": "declared_scopes_complete", "quantitative_complete_cells": 192,
            "source_ids": {"RSAR": "RSAR-source", "DIOR": "DIOR-source"},
            "source_provenance": {}, "raw_results": core_rows, "checkpoints": checkpoints,
        })
        self.scope = self.root / "scope.json"
        self.runtime = self.root / "runtime.json"
        self.runtime_data = {"evaluation_code_sha": "native-eval-sha", "cells": {}}
        self.bindings = []

    def artifacts(self, ds, domain, method, seed, role, checkpoint):
        directory = self.root / "eval" / ds / domain / method / str(seed) / role
        directory.mkdir(parents=True)
        base = dict(dataset=ds, domain=domain, method=method, seed=seed, role=role)
        cell = {**base, "eval_dir": str(directory), "config": "/source-inference.py",
                "eval_json": str(directory / "eval_fixture.json"),
                "prediction_image_ids": str(directory / "predictions.pkl.image_ids.json")}
        write_json(cell["eval_json"], {"config": cell["config"], "metric": {"mAP": .244655564427, "AP50": .245}})
        (directory / "eval_status").write_text(
            f"eval_exit=0 name={method} domain={domain} seed={seed} role={role} checkpoint={checkpoint}\n")
        (directory / "class_ap.txt").write_text(
            "| class | ap |\n" + "".join(f"| {c} | 0.245 |\n" for c in CLASSES[ds])
            + "| mAP | 0.245 |\n")
        (directory / "pred_count.txt").write_text("2 expect=2\n")
        predictions = [[np.zeros((0, 6), dtype=np.float32) for _ in CLASSES[ds]]] * 2
        with (directory / "predictions.pkl").open("wb") as stream:
            pickle.dump(predictions, stream)
        write_json(cell["prediction_image_ids"], {
            "schema": IMAGE_ORDER_SCHEMA, "origin": IMAGE_ORDER_ORIGIN, "status": "complete",
            "predictions_file": "predictions.pkl", "n_images": 2, "dataset_size": 2,
            "image_ids": ["test0", "test1"],
            "records": [{"prediction_index": i, "image_id": f"test{i}",
                         "ori_filename": f"test{i}.png"} for i in range(2)],
            "checkpoint": str(checkpoint), "config": cell["config"],
            "training_code_sha": "actual-train-sha", "evaluation_code_sha": "native-eval-sha",
        })
        iteration = (SFYOLO_FINAL_ITERATION if method == "SFYOLO" else FINAL_ITERATION)[ds]
        return cell, {**base, "path": str(checkpoint), "source_id": ds + "-source",
                      "selection": "source" if method == "A" else "final",
                      "iteration": iteration, "verified": True}

    def bind(self, method="IRG", ds="RSAR", domain="clean", seed=42, role="ema"):
        iteration = (SFYOLO_FINAL_ITERATION if method == "SFYOLO" else FINAL_ITERATION)[ds]
        suffix = "_ema" if role == "ema" else ""
        checkpoint = self.root / "work" / ds / domain / method / str(seed) / f"iter_{iteration}{suffix}.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"final fixture")
        cell, _ = self.artifacts(ds, domain, method, seed, role, checkpoint)
        binding = {**cell, "checkpoint": str(checkpoint), "source_checkpoint": self.sources[ds],
                   "source_id": ds + "-source", "training_code_sha": "prepared-train-sha",
                   "student_checkpoint": str(checkpoint.with_name(f"iter_{iteration}.pth")),
                   "final_checkpoint_iteration": iteration}
        # Real EMA runtime has training config separately from native inference config.
        if role == "ema":
            binding.update(config="/training.py", eval_config=cell["config"])
        write_json(Path(cell["eval_dir"]) / "execution.json", {
            **binding, "config": cell["config"], "training_code_sha": "actual-train-sha",
            "evaluation_code_sha": "native-eval-sha",
        })
        self.runtime_data.setdefault("student_cells" if role == "student" else "cells", {})[
            f"{ds}/{domain}/{seed}/{method}"] = binding
        self.bindings.append(binding)
        return binding

    def collect(self, methods, roles=("ema",), name="collection"):
        write_json(self.scope, {"methods": methods, "roles": roles})
        write_json(self.runtime, self.runtime_data)
        return collect(self.scope, [self.runtime], self.core, self.root / name)

    def test_all_approved_methods_roles_exact_counts_and_final_labels(self):
        methods = ["A", *EXTENSION_METHODS]
        for method in EXTENSION_METHODS:
            for ds in DOMAINS:
                for role in ("ema", "student"):
                    self.bind(method, ds, role=role)
        core_before = self.core.read_bytes()
        manifest_path = self.collect(methods, ("ema", "student"))
        manifest = read_json(manifest_path)
        raw, _, predictions, _ = collect_quantitative(manifest)
        self.assertEqual(len(raw), 12 + 36 * len(EXTENSION_METHODS) * 2)
        self.assertEqual(sum(r["method"] == "A" for r in raw), 12)
        self.assertEqual(sum(r["status"] == "complete" for r in raw), 12 + 40)
        irg = next(r for r in raw if r["method"] == "IRG" and r["status"] == "complete")
        self.assertEqual(irg["mAP50"], .244655564427)
        self.assertEqual(irg["config"], "/source-inference.py")
        self.assertEqual(irg["training_code_sha"], "actual-train-sha")
        self.assertEqual(irg["n_predictions"], 2)
        self.assertEqual(len(predictions), (12 + 40) * 2)
        self.assertEqual(self.core.read_bytes(), core_before)
        for row in raw:
            if row["method"] == "SFYOLO" and row["status"] == "complete":
                self.assertIn(str(SFYOLO_FINAL_ITERATION[row["dataset"]]), row["checkpoint"])
                self.assertEqual(row["comparison_group"], "extended_two_epoch_TAM")
        with self.assertRaises(FileExistsError):
            self.collect(methods, ("ema", "student"))

    def test_no_invented_student_or_core_fallback_and_ambiguous_eval(self):
        binding = self.bind()
        directory = Path(binding["eval_dir"])
        write_json(directory / "eval_retry.json", {"config": "/source-inference.py", "metric": {"mAP": .9}})
        manifest = read_json(self.collect(["A", "IRG"], ("ema", "student")))
        raw, _, _, _ = collect_quantitative(manifest)
        self.assertEqual(len(raw), 84)
        self.assertFalse(any(r["role"] == "student" and r["status"] == "complete" for r in raw))
        irg = next(r for r in raw if r["method"] == "IRG" and r.get("eval_dir"))
        self.assertIsNone(irg["mAP50"])
        self.assertIn("ambiguous_eval_json", irg["problems"])
        self.assertIsNone(irg["eval_json"])
        binding["eval_dir"] = read_json(self.core)["raw_results"][0]["eval_dir"]
        with self.assertRaisesRegex(ValueError, "core evaluation"):
            self.collect(["A", "IRG"], name="bad-core")

    def test_missing_native_ids_checkpoint_and_execution_are_incomplete(self):
        binding = self.bind()
        directory = Path(binding["eval_dir"])
        (directory / "predictions.pkl.image_ids.json").unlink()
        (directory / "execution.json").unlink()
        Path(binding["checkpoint"]).unlink()
        raw, _, _, _ = collect_quantitative(read_json(self.collect(["A", "IRG"])))
        irg = next(r for r in raw if r["method"] == "IRG" and r.get("eval_dir"))
        self.assertEqual(irg["status"], "incomplete")
        for problem in ("missing_prediction_image_order", "missing_native_execution_binding",
                        "checkpoint_missing_or_size_changed"):
            self.assertIn(problem, irg["problems"])

    def test_sidecar_wrong_native_provenance_cannot_complete(self):
        binding = self.bind()
        sidecar = Path(binding["prediction_image_ids"])
        payload = read_json(sidecar)
        payload["training_code_sha"] = "wrong"
        payload["checkpoint"] = "/other/iter_266_ema.pth"
        write_json(sidecar, payload)
        raw, _, _, _ = collect_quantitative(read_json(self.collect(["A", "IRG"])))
        row = next(r for r in raw if r["method"] == "IRG" and r.get("eval_dir"))
        self.assertEqual(row["status"], "incomplete")
        self.assertIn("prediction_sidecar_training_code_sha_mismatch", row["problems"])
        self.assertIn("prediction_sidecar_checkpoint_or_config_mismatch", row["problems"])

    def test_invalid_source_duplicate_and_sfyolo_common_label_rejected(self):
        binding = self.bind("SFYOLO")
        binding["final_checkpoint_iteration"] = 266
        with self.assertRaisesRegex(ValueError, "two-epoch"):
            self.collect(["A", "SFYOLO"])
        binding["final_checkpoint_iteration"] = 531
        binding["source_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "frozen source"):
            self.collect(["A", "SFYOLO"])
        binding["source_id"] = "RSAR-source"
        self.runtime_data["cells"]["duplicate"] = copy.deepcopy(binding)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.collect(["A", "SFYOLO"])

    def test_end_to_end_quant_only_render_separates_families_and_keeps_parent_partial(self):
        self.bind()
        manifest = self.collect(["A", "IRG", "LPLD", "SFYOLO", "LoRA-CGA"])
        output = self.root / "report"
        with patch("experiments.comparison.final_report.qualitative_evidence",
                   side_effect=AssertionError("must not inspect the core ROI scope")):
            report = build_report(manifest, output, docx_python=sys.executable)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["quantitative_status"], "incomplete")
        self.assertEqual(report["quantitative_expected_cells"], 156)
        self.assertEqual(report["quantitative_complete_cells"], 13)
        self.assertFalse(report["qualitative_coverage"]["full_test_roi"])
        self.assertIsNone(report["qualitative_coverage"]["roi_expected_image_roles"])
        self.assertEqual(report["comparison_groups"], comparison_groups(report["methods"]))
        self.assertEqual(len(report["figures"]), 12)
        for group in report["comparison_groups"]:
            self.assertTrue(any(group in f for f in report["figures"]))
        with ZipFile(output / "comparison_report_cn.docx") as archive:
            xml = archive.read("word/document.xml").decode()
        for text in ("IRG", "LPLD", "extended_two_epoch_TAM", "target_supervised_appendix",
                     "531", "369", "pending", "partial"):
            self.assertIn(text, xml)
        self.assertTrue((output / "paired_statistics.csv").is_file())
        with self.assertRaises(FileExistsError):
            build_report(manifest, output, metadata_only=True)

    def test_complete_quantitative_matrix_does_not_complete_parent(self):
        for ds, domains in DOMAINS.items():
            for domain in domains:
                for seed in (42, 43, 44):
                    self.bind(ds=ds, domain=domain, seed=seed)
        manifest = self.collect(["A", "IRG"])
        sidecar = Path(self.bindings[0]["prediction_image_ids"])
        original = sidecar.read_bytes()
        report = build_report(manifest, self.root / "complete-quant", metadata_only=True)
        self.assertEqual(report["quantitative_complete_cells"], 48)
        self.assertEqual(report["quantitative_status"], "complete")
        self.assertEqual(report["status"], "partial")
        self.assertEqual(sidecar.read_bytes(), original)
        self.assertTrue(all(t["holm_status"] == "complete_family"
                            for t in report["paired_statistics"]))

    def test_locked_history_and_source_producer_remain_report_only(self):
        manifest = read_json(self.collect(["A", "IRG"]))
        historical = self.root / "historical.csv"
        historical.write_text(
            "dataset,corruption,method,seed,ckpt_role,mAP50\n"
            "RSAR,chaff,A,42,source,0.123456789\n")
        manifest["source_provenance"] = {
            "RSAR": {"checkpoint": self.sources["RSAR"], "weights_sha256": "RSAR-source",
                     "actual_source_producer_sha": "original-source-producer",
                     "producer_record": "/fixture/source-producer.txt",
                     "producer_record_snapshot": None}}
        source = next(c for c in read_json(manifest["cells"])
                      if c["dataset"] == "RSAR" and c["domain"] == "chaff")
        sidecar = Path(source["prediction_image_ids"])
        original = sidecar.read_bytes()
        with patch("experiments.comparison.report_inputs.historical_paths",
                   side_effect=lambda manifest, per_class=False: [] if per_class else [historical]):
            raw, _, _, _ = collect_quantitative(manifest)
        row = next(r for r in raw if r["dataset"] == "RSAR" and r["domain"] == "chaff"
                   and r["method"] == "A")
        self.assertEqual(row["status"], "metric_conflict")
        self.assertEqual(row["mAP50"], .123456789)
        self.assertEqual(row["eval_mAP50"], .244655564427)
        self.assertEqual(row["effective_training_code_sha"], "original-source-producer")
        self.assertEqual(row["recorded_training_code_sha"], "actual-train-sha")
        self.assertEqual(sidecar.read_bytes(), original)

    def test_missing_eval_json_and_wrong_execution_never_complete(self):
        binding = self.bind()
        Path(binding["eval_json"]).unlink()
        execution_path = Path(binding["eval_dir"]) / "execution.json"
        execution = read_json(execution_path)
        execution["method"] = "LPLD"
        write_json(execution_path, execution)
        raw, _, _, _ = collect_quantitative(read_json(self.collect(["A", "IRG"])))
        row = next(r for r in raw if r["method"] == "IRG" and r.get("eval_dir"))
        self.assertEqual(row["status"], "incomplete")
        self.assertIsNone(row["mAP50"])
        self.assertIn("missing_eval_json", row["problems"])
        self.assertIn("native_execution_binding_mismatch", row["problems"])

    def test_legacy_default_still_has192_cells_and_original_figures(self):
        manifest = {"cells": [], "checkpoints": [],
                    "source_ids": {"RSAR": "RSAR-source", "DIOR": "DIOR-source"}}
        raw, _, _, _ = collect_quantitative(manifest)
        self.assertEqual(len(raw), 192)
        report = {**summarize(numeric_rows("ABCDEF")), "roles": ["ema"]}
        saved = figures(report, self.root)
        self.assertEqual(saved, ["RSAR_ema_mpc.png", "RSAR_ema_mpc.pdf",
                                 "DIOR_ema_mpc.png", "DIOR_ema_mpc.pdf"])


if __name__ == "__main__":
    unittest.main()
