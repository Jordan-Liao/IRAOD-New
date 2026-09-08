"""Finite Student jobs and seeded qualitative source reuse, without GPU execution."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.comparison.extension_manifest import prepare, seeded_plan
from experiments.comparison.finite_resumer import Cell, load_cells, pane_job, runner_command
from experiments.comparison.joint_tsne import joint_tsne
from experiments.comparison.report_qualitative import inspect_embedding, validate_plan
from tools.tests import test_finite_resumer as finite_tests


class StudentFiniteTest(unittest.TestCase):
    def test_five_port_student_rows_are_single_gpu_and_controls_remain_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "student.list"
            for method in ("IRG", "LPLD", "SFUT", "AASFOD", "SFYOLO"):
                path.write_text(f"DIOR clean 42 {method} student\n")
                cell = Cell("DIOR", "clean", 42, method, "student")
                self.assertEqual(load_cells([], [path]), {cell: False})
                self.assertEqual(cell.width, 1)
                self.assertEqual(cell.model, Cell("DIOR", "clean", 42, method))
            for method in ("A", "B_REG", "F_text_only", "F_veto_only", "ORACLE"):
                path.write_text(f"DIOR clean 42 {method} student\n")
                with self.assertRaises(ValueError):
                    load_cells([], [path])
            self.assertEqual(Cell("DIOR", "clean", 42, "F", "student").width, 1)
            self.assertEqual(Cell("DIOR", "clean", 42, "F", "student").session("eval"),
                             "xafS-DIOR-clean-42-F")

    def test_real_pidfd_serializes_training_and_both_roles_for_one_model(self):
        fixture = finite_tests.FiniteResumerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        model = Cell("DIOR", "clean", 42, "IRG")
        student = Cell("DIOR", "clean", 42, "IRG", "student")
        process, directory = fixture.start([model], "both-roles", eval_cells=[student])
        for expected in (model.session("train"), model.session("eval"), student.session("eval")):
            event = fixture.next_start()
            self.assertEqual(event["name"], expected)
            self.assertEqual(len(fixture.live), 1)
            self.assertEqual(len(event["gpus"]), 1)
            fixture.release(event["name"])
        self.assertEqual(process.wait(timeout=10), 0)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len([e for e in fixture.events if e["phase"] == "train"
                              and e["event"] == "start"]), 1)

    def test_student_only_scope_adopts_original_ema_training_without_relaunch(self):
        fixture = finite_tests.FiniteResumerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        model = Cell("DIOR", "clean", 42, "IRG")
        student = Cell("DIOR", "clean", 42, "IRG", "student")
        command = shlex.join(runner_command(fixture.q, model, "train", (5,)))
        subprocess.run(["tmux", "new-session", "-d", "-s", model.session("train"), command],
                       env=fixture.env, check=True)
        event = fixture.next_start()
        process, directory = fixture.start([], "adopt-model", eval_cells=[student])
        # Exercise discovery of a model not itself listed in this Student-only scope.
        from experiments.comparison.finite_resumer import TmuxBackend, take_lock
        lock_path = fixture.q / "cell_locks" / (model.session("train") + ".lock")
        with patch.dict(os.environ, fixture.env):
            backend = TmuxBackend(fixture.q, fixture.root / "adoption-probe")
            handles = backend.discover({student: False}, {})
            self.assertEqual([(h["cell"], h["phase"]) for h in handles], [(model, "train")])
            self.assertIsNone(take_lock(lock_path))
            for handle in handles:
                backend.close(handle)
            backend.selector.close()
        fixture.release(event["name"])
        event = fixture.next_start()
        self.assertEqual(event["name"], student.session("eval"))
        fixture.release(event["name"])
        self.assertEqual(process.wait(timeout=10), 0)
        self.assertEqual(json.loads((directory / "state.json").read_text())["status"], "complete")

    def test_student_native_finite_job_reuses_training_but_not_ema_evaluation(self):
        fixture = finite_tests.FiniteResumerTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        core = Cell("DIOR", "cloudy", 44, "C")
        student = Cell("DIOR", "cloudy", 44, "C", "student")
        fixture.successful_train_files(core)
        process, _ = fixture.start([], "core-eval", eval_cells=[core])
        fixture.finish(process, 1)
        original = fixture.q / "artifacts/DIOR/cloudy/44/C/eval_full_cloudy_ids_v1"
        sidecar_bytes = (original / "predictions.pkl.image_ids.json").read_bytes()
        process, directory = fixture.start([], "student-eval", eval_cells=[student])
        event = fixture.next_start()
        self.assertEqual(event["name"], student.session("eval"))
        self.assertEqual(event["phase"], "eval")
        fixture.release(event["name"])
        self.assertEqual(process.wait(timeout=10), 0)
        state = json.loads((directory / "state.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertFalse(state["cells"][0]["train_requested"])
        self.assertEqual(state["cells"][0]["cell"]["role"], "student")
        saved = json.loads((original.with_name("eval_full_cloudy_student_ids_v1")
                            / "predictions.pkl.image_ids.json").read_text())
        self.assertTrue(saved["checkpoint"].endswith("iter_185.pth"))
        self.assertEqual((original / "predictions.pkl.image_ids.json").read_bytes(), sidecar_bytes)

    def test_role_input_and_session_bindings_are_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "student.list"
            path.write_text("RSAR clean 43 F student\n")
            cell = Cell("RSAR", "clean", 43, "F", "student")
            self.assertEqual(load_cells([], [path]), {cell: False})
            with self.assertRaisesRegex(ValueError, "outside"):
                load_cells([path], [])
            command = runner_command(root, cell, "eval", (5,))
            self.assertEqual(command[-1], "student")
            actual = pane_job(cell.session("eval"), {"command": " ".join(command)}, root)
            self.assertEqual(actual[0], cell)
            legacy = Path(root) / "owner"
            legacy.mkdir()
            (Path(root) / "paths.py").write_text(f"LEGACY_STUDENT_QUEUE = {str(legacy)!r}\n")
            old_command = f"bash {legacy}/run_eval_student.sh 5 RSAR clean 43 F"
            actual = pane_job("xafS-RSAR-clean-43-F", {"command": old_command}, root)
            self.assertEqual(actual[0], cell)
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                runner_command(root, cell, "train", (4, 5))


class SeededQualitativeTest(unittest.TestCase):
    def test_prepare_emits_only_new_student_and_seeded_non_source_jobs(self):
        from tools.tests.test_extension_training import ExtensionTrainingTest

        fixture = ExtensionTrainingTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.plan = json.loads(fixture.base.read_text())
        source = fixture.root / "paths.py"
        source.write_text(
            f"ROOT = {str(fixture.root / 'weights')!r}\n"
            "EXPECT_PRED = {'RSAR':8538,'DIOR':11738}\n"
            "def student_path(ds, domain, seed, method):\n"
            "    return f'{ROOT}/{ds}/{domain}/{seed}/{method}/iter_final.pth'\n"
            "def ema_path(ds, domain, seed, method):\n"
            "    return student_path(ds, domain, seed, method).replace('.pth','_ema.pth')\n"
            "def eval_full_dir(ds, domain, seed, method):\n"
            "    return f'{ROOT}/{ds}/{domain}/{seed}/{method}/eval_full_{domain}_ids_v1'\n")
        raw = []
        for run in fixture.plan["runs"]:
            if run["role"] != "ema":
                continue
            for seed in (42, 43, 44):
                path = fixture.root / "weights" / run["dataset"] / run["domain"] / str(seed) / run["method"]
                path.mkdir(parents=True)
                (path / "iter_final.pth").write_bytes(b"fixture retained Student")
                raw.append({
                    "dataset": run["dataset"], "domain": run["domain"], "method": run["method"],
                    "seed": seed, "checkpoint": str(path / "iter_final_ema.pth"),
                    "config": run["config"], "effective_training_code_sha": "fixture-training",
                    "source_id": "fixture-source",
                })
        base, report = fixture.root / "base.json", fixture.root / "core.json"
        base.write_text(json.dumps(fixture.plan))
        report.write_text(json.dumps({"status": "declared_scopes_complete",
                                     "quantitative_complete_cells": 192, "raw_results": raw}))
        artifacts = fixture.root / "existing_owner_root"
        artifacts.mkdir()
        out = prepare(base, report, source, fixture.root / "extension",
                      Path(__file__).resolve().parents[2], sys.executable, artifacts)
        runtime = json.loads((out / "student_queue/runtime.json").read_text())
        self.assertEqual(len(runtime["student_cells"]), 180)
        self.assertTrue(all(r["eval_dir"].endswith(f"eval_full_{r['domain']}_student_ids_v1")
                            for r in runtime["student_cells"].values()))
        self.assertEqual(len((out / "student_queue/student.list").read_text().splitlines()), 180)
        jobs = json.loads((out / "roi_jobs.json").read_text())
        self.assertEqual(len(jobs), 240)
        self.assertFalse(any("/A/" in j["run_id"] for j in jobs))
        self.assertTrue(all(j["out_dir"].startswith(str(artifacts / "qualitative")) for j in jobs))
        self.assertEqual(len(json.loads((out / "embedding_jobs.json").read_text())), 48)
        subprocess.run(["bash", "-n", str(out / "student_queue/run_eval_full.sh")], check=True)

    def test_seeded_plan_reuses_exact_source_objects_and_rejects_mixed_adaptation_seeds(self):
        from tools.tests import test_aligned_roi_completion as roi_tests

        fixture = roi_tests.CompletionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        class Paths:
            @staticmethod
            def ema_path(ds, domain, seed, method):
                return f"/new/{ds}/{domain}/{seed}/{method}/iter_final_ema.pth"

            @staticmethod
            def student_path(ds, domain, seed, method):
                return f"/new/{ds}/{domain}/{seed}/{method}/iter_final.pth"

        plan = seeded_plan(fixture.plan, Paths, 43, fixture.root / "extensions")
        validate_plan(plan)
        for old, new in zip(fixture.plan["runs"], plan["runs"]):
            if old["method"] == "A":
                self.assertEqual(new, old)
            else:
                self.assertEqual(new["seed"], 43)
                self.assertIn("/seed_43/", new["out_dir"])
                self.assertEqual(new["image_ids"], old["image_ids"])
                self.assertEqual(new["visualization_image_ids"], old["visualization_image_ids"])
        plan["runs"][1]["seed"] = 44
        with self.assertRaisesRegex(ValueError, "fixed matrix"):
            validate_plan(plan)

    def test_seed43_joint_embedding_binds_source42_and_actual_per_point_seeds(self):
        from tools.tests import test_aligned_roi_completion as roi_tests

        fixture = roi_tests.CompletionTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.plan["adaptation_seed"] = 43
        for run in fixture.plan["runs"]:
            if run["method"] != "A":
                run["seed"] = 43
        mapping = roi_tests.AlignedMappingTest()
        mapping.setUp()
        arrays, _, _ = mapping.capture(mapping.features, mapping.rois, mapping.delta)
        selected = [r for r in fixture.plan["runs"]
                    if r["dataset"] == "DIOR" and r["domain"] == "clean"
                    and r["role"] in ("source", "ema")]
        for run in selected:
            fixture.export_fixture(run, arrays)
        target = fixture.root / "seed43_embedding"
        _, points = joint_tsne(fixture.plan, "DIOR", "clean", "ema", target, cap=5, perplexity=2)
        self.assertEqual({p["seed"] for p in points if p["method"] == "A"}, {42})
        self.assertEqual({p["seed"] for p in points if p["method"] != "A"}, {43})
        result = inspect_embedding({"dataset": "DIOR", "domain": "clean",
                                    "comparison": "ema", "directory": str(target)}, fixture.plan)
        self.assertEqual(result["status"], "complete")


if __name__ == "__main__":
    unittest.main()
