"""Synthetic CPU bindings: no detector, native ops, GPU, or claimed experiment results."""

import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from experiments.comparison.aligned_roi import validate_native_predictions
from experiments.comparison.complete_qualitative import collect_completion
from experiments.comparison.extension_manifest import prepare_qualitative
from experiments.comparison.joint_tsne import joint_tsne
from experiments.comparison.report_qualitative import (
    inspect_embedding, qualitative_evidence, validate_plan, reuse_completed_evidence)
from experiments.comparison.result_completion import (
    DOMAINS, EXPECTED_TEST_IMAGES, FEATURE_POINT, FEATURE_VERSION, PORT_METHODS, ROLES, SCHEMA,
    comparison_runs, load_export, native_binding_evidence, read_json, write_json)


class GeneralizedQualitativeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        sizes = patch.dict(EXPECTED_TEST_IMAGES, {"RSAR": 35, "DIOR": 18})
        sizes.start()
        self.addCleanup(sizes.stop)
        self.plans = {}
        for seed in (42, 43, 44):
            runs = []
            for ds, domains in DOMAINS.items():
                ids = ([f"r{i}" for i in range(35)] if ds == "RSAR"
                       else [str(i) for i in range(11726, 11744)])
                for domain in domains:
                    for method, role in ROLES:
                        run_seed = 42 if method == "A" else seed
                        run_id = f"{ds}/{domain}/{method}/{role}"
                        runs.append({
                            "run_id": run_id, "dataset": ds, "domain": domain,
                            "method": method, "role": role, "seed": run_seed, "scope": "full_test",
                            "checkpoint_domain": "source" if method == "A" else domain,
                            "checkpoint": f"/weights/{ds}/{domain}/{run_seed}/{method}/{role}.pth",
                            "config": f"/configs/{ds}.py", "ann_file": f"/data/{ds}/TEST",
                            "img_prefix": f"/data/{ds}/{domain}", "image_ids": ids,
                            "visualization_image_ids": ids[:32 if ds == "RSAR" else 16],
                            "show_score_thr": .3,
                            "out_dir": str(self.root / "old" / str(run_seed) / run_id),
                        })
            self.plans[seed] = {"schema": SCHEMA, "adaptation_seed": seed, "runs": runs}
            validate_plan(self.plans[seed])
            write_json(self.root / f"old{seed}.json", self.plans[seed])
        self.paths = self.root / "paths.py"
        self.paths.write_text(
            "def ema_path(ds, domain, seed, method):\n"
            "    return f'/weights/{ds}/{domain}/{seed}/{method}/ema.pth'\n"
            "def student_path(ds, domain, seed, method):\n"
            "    return f'/weights/{ds}/{domain}/{seed}/{method}/student.pth'\n")
        cells = {}
        for seed, plan in self.plans.items():
            for source in (r for r in plan["runs"] if r["method"] == "A"):
                ds, domain = source["dataset"], source["domain"]
                for method in PORT_METHODS:
                    iteration = (531 if ds == "RSAR" else 369) if method == "SFYOLO" else (
                        266 if ds == "RSAR" else 185)
                    root = self.root / "native" / ds / domain / str(seed) / method
                    key = f"{ds}/{domain}/{seed}/{method}"
                    cells[key] = {
                        "dataset": ds, "domain": domain, "seed": seed, "method": method,
                        "source_checkpoint": source["checkpoint"], "source_seed": 42,
                        "source_id": f"source-{ds}", "training_code_sha": f"training-{method}",
                        "eval_config": source["config"], "ann_file": source["ann_file"],
                        "img_prefix": source["img_prefix"], "eval_dir": str(root / "eval_ema"),
                        "checkpoint": str(root / f"iter_{iteration}_ema.pth"),
                        "student_checkpoint": str(root / f"iter_{iteration}.pth"),
                    }
        self.runtime = self.root / "runtime.json"
        write_json(self.runtime, {"schema": "iraod-extension-training-v1",
                                 "evaluation_code_sha": "native-eval", "cells": cells})

    def prepare(self, methods=PORT_METHODS, runtimes=None, cpu_python=None):
        return prepare_qualitative(
            self.root / "old42.json", [self.root / "old43.json", self.root / "old44.json"],
            self.paths, runtimes or [self.runtime], list(methods), self.root / "new",
            Path(__file__).resolve().parents[2], sys.executable, cpu_python)

    def test_exact_new_budgets_unique_bindings_and_old_objects_unchanged(self):
        out = self.prepare(cpu_python="/existing/cpu/python")
        scope = read_json(out / "qualitative_scope.json")
        self.assertEqual((scope["new_models"], scope["native_eval_dependencies"],
                          scope["new_roi_groups"], scope["new_visualizations"],
                          scope["new_grouped_embeddings"], scope["panels_per_embedding"]),
                         (180, 360, 360, 9600, 72, 11))
        # Production TEST denominators, independent of the reduced fixture.
        full_images = 8 * 8538 + 4 * 11738
        self.assertEqual(full_images * 5 * 2 * 2, 2305120)  # accepted BF43/44
        self.assertEqual(full_images * 5 * 2 * 3, 3457680)  # five ports, all seeds/roles
        jobs = read_json(out / "roi_jobs.json")
        self.assertEqual(len({j["run_id"] for j in jobs}), 360)
        self.assertEqual(len({j["out_dir"] for j in jobs}), 360)
        self.assertTrue(all(j["status"] == "pending_inputs" for j in jobs))
        self.assertTrue(all(j["method"] in PORT_METHODS for j in jobs))
        self.assertTrue(all(j["export_argv"][0] == sys.executable for j in jobs))
        self.assertTrue(all(j["visualize_argv"][:2] == ["env", "CUDA_VISIBLE_DEVICES="] for j in jobs))
        self.assertTrue(all("/existing/cpu/python" in j["visualize_argv"] for j in jobs))
        for seed in (42, 43, 44):
            plan = read_json(out / f"qualitative_seed{seed}.json")
            self.assertEqual(plan["runs"][:132], self.plans[seed]["runs"])
            validate_plan(plan)
            for run in plan["runs"][132:]:
                self.assertIn(f"/seed_{seed}/", run["run_id"])
                suffix = "_ema.pth" if run["role"] == "ema" else ".pth"
                final = (531 if run["dataset"] == "RSAR" else 369) if run["method"] == "SFYOLO" else (
                    266 if run["dataset"] == "RSAR" else 185)
                self.assertTrue(run["checkpoint"].endswith(f"iter_{final}{suffix}"))
                if run["method"] == "SFYOLO":
                    self.assertFalse(run["rank_with_common_one_epoch"])
            group = comparison_runs(plan, "DIOR", "cloudy", "student")
            self.assertEqual(len(group), 11)
            self.assertEqual(group[0]["seed"], 42)
            self.assertTrue(all(r["seed"] == seed for r in group[1:]))
        embeds = read_json(out / "embedding_jobs.json")
        self.assertTrue(all(len(e["dependencies"]) == 11 for e in embeds))
        self.assertTrue(all("/existing/cpu/python" in e["argv"] for e in embeds))
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_declared_subset_and_split_runtimes_do_not_invent_ablations(self):
        runtime = read_json(self.runtime)
        second = self.root / "runtime2.json"
        write_json(second, {**runtime, "cells": {
            k: v for k, v in runtime["cells"].items() if v["method"] == "SFYOLO"}})
        write_json(self.runtime, {**runtime, "cells": {
            k: v for k, v in runtime["cells"].items() if v["method"] != "SFYOLO"}})
        out = self.prepare(("IRG", "SFYOLO"), [self.runtime, second])
        scope = read_json(out / "qualitative_scope.json")
        self.assertEqual(scope["new_roi_groups"], 144)
        self.assertEqual(scope["panels_per_embedding"], 8)
        self.assertEqual(scope["new_grouped_embeddings"], 72)
        plan = read_json(out / "qualitative_seed44.json")
        plan["runs"][-1]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "fixed matrix"):
            validate_plan(plan)

    def test_reject_missing_runtime_wrong_final_and_rewritten_source(self):
        data = read_json(self.runtime)
        key = next(k for k, c in data["cells"].items() if c["method"] == "SFYOLO")
        original = copy.deepcopy(data)
        data["cells"][key]["checkpoint"] = "/wrong/iter_266_ema.pth"
        write_json(self.runtime, data)
        with self.assertRaisesRegex(ValueError, "method-final"):
            self.prepare()
        data = copy.deepcopy(original)
        del data["cells"][key]
        write_json(self.runtime, data)
        with self.assertRaisesRegex(ValueError, "exactly 12 domains"):
            self.prepare()
        write_json(self.runtime, original)
        plan = read_json(self.root / "old43.json")
        plan["runs"][0]["out_dir"] = "/new/source"
        write_json(self.root / "old43.json", plan)
        with self.assertRaisesRegex(ValueError, "exact existing A/source"):
            self.prepare()

    def native_fixture(self, run):
        root = Path(run["native_prediction"]["eval_dir"])
        root.mkdir(parents=True)
        Path(run["checkpoint"]).write_bytes(b"synthetic checkpoint identity, not a model")
        (root / "eval_status").write_text(
            f"eval_exit=0 name={run['method']} domain={run['domain']} seed={run['seed']} "
            f"role={run['role']} checkpoint={run['checkpoint']}\n")
        write_json(root / "execution.json", {
            **{key: run[key] for key in ("dataset", "domain", "seed", "method", "role",
                                         "checkpoint", "config")},
            "source_id": run["native_prediction"]["source_id"],
            "evaluation_code_sha": run["native_prediction"]["evaluation_code_sha"],
            "training_code_sha": "actual-execution-revision",
        })
        (root / "predictions.pkl").write_bytes(b"synthetic presence fixture; not loaded as predictions")
        ids = list(reversed(run["image_ids"]))
        write_json(root / "predictions.pkl.image_ids.json", {
            "schema": "iraod-prediction-image-order-v1", "origin": "inference_batch_img_metas",
            "status": "complete", "predictions_file": "predictions.pkl",
            "checkpoint": run["checkpoint"], "config": run["config"],
            "training_code_sha": "actual-execution-revision",
            "evaluation_code_sha": run["native_prediction"]["evaluation_code_sha"],
            "dataset_size": len(ids), "n_images": len(ids), "image_ids": ids,
            "cfg_options": {"data.test.ann_file": run["ann_file"],
                            "data.test.img_prefix": run["img_prefix"]},
            "records": [{"prediction_index": i, "image_id": image_id,
                         "ori_filename": image_id + ".png"} for i, image_id in enumerate(ids)],
        })

    @staticmethod
    def arrays(image_id):
        return {
            "features": np.array([[1., 2.], [2., 1.], [3., 4.]]),
            "boxes": np.array([[0., 0., 2., 2., 0.], [1., 1., 2., 2., 0.], [2., 2., 2., 2., 0.]]),
            "scores": np.array([.2, .8, .9]), "labels": np.array([0, 1, 0]),
            "proposal_indices": np.array([0, 1, 2]), "flat_indices": np.array([0, 3, 4]),
            "detection_indices": np.arange(3), "image_ids": np.full(3, image_id),
        }

    def export_fixture(self, run):
        root = Path(run["out_dir"])
        root.mkdir(parents=True)
        records = []
        for image_id in run["image_ids"]:
            np.savez(root / f"{image_id}.npz", **self.arrays(image_id))
            records.append({"image_id": image_id, "feature_file": image_id + ".npz",
                            "n_detections": 3, "n_roi": 3})
        index = {
            "schema": SCHEMA, "status": "complete", "run": run, "records": records,
            "classes": ["class0", "class1"], "rescale": True,
            "test_cfg": {"score_thr": .05}, "feature_point": FEATURE_POINT,
            "code_commit": run.get("export_code_sha", "accepted-legacy-code"),
        }
        if "native_prediction" in run:
            self.native_fixture(run)
            index.update(native_prediction=native_binding_evidence(run), feature_version=FEATURE_VERSION)
        write_json(root / "index.json", index)

    def test_native_alignment_keeps_low_scores_and_refuses_count_only_evidence(self):
        out = self.prepare(("SFYOLO",))
        plan = read_json(out / "qualitative_seed44.json")
        run = next(r for r in plan["runs"] if r["method"] == "SFYOLO" and r["role"] == "student")
        self.assertEqual(native_binding_evidence(run)["status"], "pending")
        self.export_fixture(run)
        index, records = load_export(run)
        _, arrays = next(records)
        self.assertEqual(len(arrays["scores"]), 3)
        self.assertEqual(arrays["scores"][0], .2)
        per_class = [np.column_stack((arrays["boxes"][arrays["labels"] == label],
                                      arrays["scores"][arrays["labels"] == label])) for label in range(2)]
        validate_native_predictions(arrays, per_class)
        per_class[0][0, -1] = .21
        with self.assertRaisesRegex(ValueError, "bound native"):
            validate_native_predictions(arrays, per_class)
        index["feature_version"] = "unbound-version"
        write_json(Path(run["out_dir"]) / "index.json", index)
        with self.assertRaisesRegex(ValueError, "feature version"):
            load_export(run)
        sidecar = Path(run["native_prediction"]["eval_dir"]) / "predictions.pkl.image_ids.json"
        data = read_json(sidecar)
        data["records"][0]["image_id"] = data["image_ids"][1]
        write_json(sidecar, data)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            native_binding_evidence(run)

    def test_native_execution_revision_and_last_status_not_stale_preparation(self):
        out = self.prepare(("IRG",))
        plan = read_json(out / "qualitative_seed42.json")
        run = next(r for r in plan["runs"] if r["method"] == "IRG")
        self.native_fixture(run)
        evidence = native_binding_evidence(run)
        self.assertEqual(evidence["training_code_sha"], "actual-execution-revision")
        self.assertNotEqual(evidence["training_code_sha"], evidence["prepared_training_code_sha"])
        root = Path(run["native_prediction"]["eval_dir"])
        with (root / "eval_status").open("a") as stream:
            stream.write("eval_exit=1 failed-retry\n")
        self.assertEqual(native_binding_evidence(run)["status"], "pending")
        (root / "eval_status").write_text(
            f"eval_exit=0 name={run['method']} domain={run['domain']} seed={run['seed']} "
            f"role={run['role']} checkpoint={run['checkpoint']}\n")
        execution = read_json(root / "execution.json")
        execution["training_code_sha"] = "different-execution"
        write_json(root / "execution.json", execution)
        with self.assertRaisesRegex(ValueError, "binding mismatch"):
            native_binding_evidence(run)

    def test_port_roi_preserves_old_capacity_and_admits_new_owned_seven(self):
        from experiments.comparison.dior_recovery.extract_roi_pre_fc_cls import require_owned_gpu

        run = {"allowed_gpus": [4, 5, 6]}
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": "6"}):
            require_owned_gpu(run)
        for device in ("7", "4,5", ""):
            with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": device}):
                with self.assertRaisesRegex(ValueError, "bound physical"):
                    require_owned_gpu(run)
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "0", "CUDA_VISIBLE_DEVICES": "4"}):
            with self.assertRaisesRegex(RuntimeError, "actual GPU lock"):
                require_owned_gpu(run)
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": "7"}):
            require_owned_gpu({"allowed_gpus": [4, 5, 6, 7]})
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": "3"}):
            with self.assertRaisesRegex(ValueError, "bound physical"):
                require_owned_gpu({"allowed_gpus": [4, 5, 6, 7]})

    def test_report_reuses_bf_qualitative_references_without_recollecting_quantities(self):
        from experiments.comparison.final_report import build_report

        out = self.prepare()
        manifest = self.root / "report-manifest.json"
        write_json(manifest, {
            "schema": "iraod-comparison-report-v1", "methods": ["A", *PORT_METHODS],
            "roles": ["ema"], "cells": [], "checkpoints": [],
            "source_ids": {"RSAR": "source-RSAR", "DIOR": "source-DIOR"},
            "qualitative_plan": str(out / "qualitative_seed43.json"),
        })
        with patch("experiments.comparison.final_report.collect_quantitative",
                   return_value=([], [], [], ["ema"])) as collect_quant:
            report = build_report(manifest, self.root / "report", metadata_only=True)
        collect_quant.assert_called_once()
        self.assertEqual(report["qualitative_reference_methods"], list("BCDEF"))
        self.assertEqual(report["qualitative_adaptation_seed"], 43)
        self.assertEqual(report["qualitative_coverage"]["vis_expected_image_roles"], 6720)
        self.assertEqual(report["status"], "partial")

    def test_streaming_collection_is_method_aware_and_honestly_partial(self):
        out = self.prepare()
        plan_path = out / "qualitative_seed43.json"
        plan = read_json(plan_path)
        chosen = next(r for r in plan["runs"] if r["method"] == "SFYOLO")
        self.export_fixture(chosen)
        seen = []
        rows, embeddings, coverage = qualitative_evidence(
            {"qualitative_plan": str(plan_path), "inspect_roi_run_ids": [chosen["run_id"]]}, seen.append)
        self.assertEqual(rows, [])
        self.assertEqual(len(seen), len(chosen["image_ids"]))
        self.assertEqual(coverage["roi_complete_groups"], 1)
        self.assertEqual(coverage["roi_detection_rows"], 3 * len(chosen["image_ids"]))
        self.assertEqual(coverage["vis_expected_image_roles"], 6720)
        self.assertEqual(coverage["methods"], [*"ABCDEF", *PORT_METHODS])
        self.assertEqual(len(embeddings), 24)
        group = next(g for g in coverage["roi_group_index"] if g["inspected"])
        self.assertEqual((group["method"], group["seed"], group["budget_group"]),
                         ("SFYOLO", 43, "extended_two_epoch_TAM"))
        summary = collect_completion(plan_path, out / "tsne/seed_43", self.root / "summary")
        self.assertEqual(summary["qualitative_status"], "partial")
        self.assertEqual(summary["quantitative_status"], "pending_separate_evidence")

    def test_joint_producer_and_consumer_all_methods_with_synthetic_estimator(self):
        """Exercise real reservoir/export/normalization plumbing; estimator is explicitly a stub."""
        out = self.prepare()
        plan = read_json(out / "qualitative_seed44.json")
        runs = comparison_runs(plan, "DIOR", "cloudy", "student")
        for run in runs:
            self.export_fixture(run)

        class SyntheticEstimator:
            def __init__(self, **params):
                self.params = params
                self.kl_divergence_ = 0.

            def get_params(self):
                return self.params

            def fit_transform(self, features):
                return features[:, :2].copy()

        figure, axes = MagicMock(), MagicMock()
        figure.savefig.side_effect = lambda path, **kw: Path(path).write_bytes(b"synthetic panel fixture")
        plt = MagicMock()
        plt.subplots.return_value = figure, axes
        modules = {
            "matplotlib": types.SimpleNamespace(use=lambda backend: None, pyplot=plt),
            "matplotlib.pyplot": plt,
            "sklearn": types.SimpleNamespace(__version__="synthetic-test-double"),
            "sklearn.manifold": types.SimpleNamespace(TSNE=SyntheticEstimator),
        }
        directory = out / "synthetic-embedding"
        with patch.dict(sys.modules, modules):
            coords, points = joint_tsne(plan, "DIOR", "cloudy", "student", directory, cap=3, perplexity=2)
            with self.assertRaises(FileExistsError):
                joint_tsne(plan, "DIOR", "cloudy", "student", directory, cap=3, perplexity=2)
        self.assertEqual(coords.shape, (33, 2))
        self.assertTrue(all(p["score"] > .3 for p in points))
        self.assertEqual({p["seed"] for p in points if p["method"] == "A"}, {42})
        self.assertEqual({p["seed"] for p in points if p["method"] != "A"}, {44})
        entry = {"dataset": "DIOR", "domain": "cloudy", "comparison": "student",
                 "directory": str(directory)}
        self.assertEqual(inspect_embedding(entry, plan)["status"], "complete")
        protocol = read_json(directory / "protocol.json")
        self.assertEqual(protocol["budget_labels"]["SFYOLO"]["final_checkpoint_iteration"], 369)
        self.assertFalse(protocol["budget_labels"]["SFYOLO"]["rank_with_common_one_epoch"])
        # Point/feature evidence must detect mutation, not accept matching counts.
        feature_path = Path(points[-1]["feature_file"])
        with np.load(feature_path) as saved:
            data = {k: saved[k] for k in saved.files}
        data["features"] += 1
        np.savez(feature_path, **data)
        with self.assertRaisesRegex(ValueError, "feature differs"):
            inspect_embedding(entry, plan)

    def test_legacy_plan_and_completion_reuse_remain_explicit(self):
        validate_plan(self.plans[42])
        self.assertEqual(len(comparison_runs(self.plans[42], "RSAR", "clean", "ema")), 6)
        self.assertEqual(len(comparison_runs(self.plans[43], "DIOR", "clean", "student")), 6)
        with self.assertRaisesRegex(ValueError, "scoped inspection"):
            reuse_completed_evidence({"qualitative_evidence": "/not-read", "inspect_roi_run_ids": []})

    def test_declared_method_csv_uses_explicit_output_not_run_id_depth(self):
        from experiments.comparison.result_completion import main

        out = self.prepare(("SFYOLO",))
        target = self.root / "explicit-coverage.csv"
        argv = ["result_completion", "collect", "--plan", str(out / "qualitative_seed44.json"),
                "--out", str(target)]
        with patch.object(sys, "argv", argv), patch("builtins.print"):
            main()
            with self.assertRaises(FileExistsError):
                main()
        self.assertTrue(target.is_file())
        self.assertFalse((out / "roi/DIOR/coverage.csv").exists())

    def test_final_report_accepts_explicit_qualitative_scope_not_only_quantitative(self):
        from experiments.comparison.final_report import build_report

        out = self.prepare(("SFYOLO",))
        manifest = self.root / "report-input.json"
        write_json(manifest, {
            "schema": "iraod-comparison-report-v1", "methods": [*"ABCDEF", "SFYOLO"],
            "qualitative_plan": str(out / "qualitative_seed44.json"),
            "source_ids": {}, "checkpoints": [],
        })
        with patch("experiments.comparison.final_report.collect_quantitative",
                   return_value=([], [], [], ("ema", "student"))), patch(
                "experiments.comparison.final_report.historical_paths", return_value=[]):
            report = build_report(manifest, self.root / "report", metadata_only=True)
        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["qualitative_coverage"]["methods"], [*"ABCDEF", "SFYOLO"])
        self.assertEqual(report["qualitative_coverage"]["adaptation_seed"], 44)
        self.assertEqual(report["qualitative_coverage"]["vis_expected_image_roles"], 4160)
        self.assertEqual(len(report["embedding_index"]), 24)

    @unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("matplotlib"),
                         "Real t-SNE requires the parent's provisioned CPU environment")
    def test_real_cpu_joint_tsne_eleven_panels(self):
        out = self.prepare()
        plan = read_json(out / "qualitative_seed43.json")
        for run in comparison_runs(plan, "DIOR", "clean", "ema"):
            self.export_fixture(run)
        directory = out / "real-tsne-synthetic-features"
        coords, _ = joint_tsne(plan, "DIOR", "clean", "ema", directory, cap=3, perplexity=2)
        self.assertEqual(coords.shape, (33, 2))
        result = inspect_embedding(
            {"dataset": "DIOR", "domain": "clean", "comparison": "ema", "directory": str(directory)},
            plan)
        self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
