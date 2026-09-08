"""CPU-only finite port bindings and mocked train/native-eval dispatch."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.comparison import extension_training as training
from experiments.comparison import extension_manifest
from experiments.comparison import b_regression
from experiments.comparison.collect_report_manifest import load_resolver
from experiments.comparison.finite_resumer import Cell, load_cells, train_state
from experiments.comparison.result_completion import DOMAINS, ROLES, SCHEMA, write_json
from experiments.comparison.report_inputs import CLASSES, EXPECTED_IMAGES
from experiments.comparison.b_regression import TRAINING_CODE_SHA as B_CODE_SHA


class ExtensionTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="port fixtures ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "new metadata"
        self.artifacts = self.root / "formal outputs"
        self.eval_code = self.root / "native331d"
        self.eval_code.mkdir()
        (self.eval_code / "test.py").write_text("# never executed\n")
        self.python = self.root / "venv/bin/python"
        self.python.parent.mkdir(parents=True)
        self.python.symlink_to(sys.executable)
        self.core = self.root / "core"
        self.core.mkdir()
        self.paths = self.core / "paths.py"
        self.paths.write_text(
            f"ROOT = {str(self.core)!r}\n"
            "def ema_path(ds, domain, seed, method):\n"
            "    assert seed == '42' and method == 'A'\n"
            "    return f'{ROOT}/{ds}_source.pth'\n")
        (self.core / "with_gpu_lock.sh").write_text(
            "#!/usr/bin/env bash\nLOCKDIR=/tmp/fixture_shared_gpu_locks\n")
        runs, rows = [], []
        for ds, domains in DOMAINS.items():
            source = self.core / f"{ds}_source.pth"
            source.write_bytes(b"fixed source fixture")
            config = self.eval_code / f"{ds}_source.py"
            config.write_text("# native source architecture fixture\n")
            ids = ([str(i) for i in range(11726, 11726 + 11738)] if ds == "DIOR"
                   else [f"rsar_{i}" for i in range(8538)])
            ann = self.core / f"{ds}_test_annotations"
            ann.write_bytes(b"Must not be parsed or scanned during preparation")
            for domain in domains:
                images = self.root / "data" / ds / domain / "test"
                if ds == "RSAR":
                    images = images / "images"
                    val = images.parent.parent / "val/images"
                else:
                    val = images.with_name("val")
                images.mkdir(parents=True)
                val.mkdir(parents=True)
                for method, role in ROLES:
                    runs.append({
                        "run_id": f"{ds}/{domain}/{method}/{role}", "dataset": ds,
                        "domain": domain, "method": method, "role": role, "seed": 42,
                        "scope": "full_test", "checkpoint_domain": "source" if method == "A" else domain,
                        "checkpoint": str(source) if method == "A" else "/unused/core_final.pth",
                        "config": "/unused/plan_config.py", "ann_file": str(ann),
                        "img_prefix": str(images), "image_ids": ids,
                        "visualization_image_ids": ids[:32 if ds == "RSAR" else 16],
                    })
                rows.append({
                    "dataset": ds, "domain": domain, "seed": 42, "method": "A",
                    "role": "ema", "status": "complete", "checkpoint": str(source),
                    "config": str(config), "source_id": f"fixed-{ds}-source-id",
                    "effective_training_code_sha": "old-source-producer",
                })
                for seed in (42, 43, 44):
                    for method in "BCDEF":
                        rows.append({"dataset": ds, "domain": domain, "seed": seed,
                                     "method": method, "status": "complete"})
        self.base = self.root / "base.json"
        self.report = self.root / "report.json"
        write_json(self.base, {"schema": SCHEMA, "adaptation_seed": 42, "runs": runs})
        write_json(self.report, {"status": "declared_scopes_complete",
                                "quantitative_complete_cells": 192, "raw_results": rows})
        self.sha = patch.object(training, "code_sha", side_effect=lambda code:
                                training.EVALUATION_SHA if code == self.eval_code else "new-producer")
        self.sha.start()
        self.addCleanup(self.sha.stop)

    def prepare(self, methods=("IRG",), **overrides):
        kwargs = dict(base_plan=self.base, core_report=self.report, core_paths=self.paths,
                      out_dir=self.queue, artifact_root=self.artifacts,
                      eval_code=self.eval_code, python=self.python, methods=methods)
        kwargs.update(overrides)
        return training.prepare(**kwargs)

    def runtime(self):
        return json.loads((self.queue / "runtime.json").read_text())

    def cell(self, ds="DIOR", domain="cloudy", seed=44, method="IRG"):
        return self.runtime()["cells"][f"{ds}/{domain}/{seed}/{method}"]

    def train(self, ds="DIOR", domain="cloudy", seed=44, method="IRG", gpu=4):
        return training.train(self.queue, gpu, ds, domain, seed, method)

    def complete(self, cell):
        for field in ("student_checkpoint", "checkpoint"):
            Path(cell[field]).write_bytes(b"exact final fixture")
        return subprocess.CompletedProcess([], 0)

    def test_shared_native_evaluator_preserves_actual_ema_and_student_roles(self):
        runtime = dict(python=sys.executable, evaluation_code=str(self.eval_code),
                       evaluation_code_sha=training.EVALUATION_SHA)
        for role in ("ema", "student"):
            out = self.root / ("native-" + role)
            checkpoint = "/fixed/iter_266" + ("_ema" if role == "ema" else "") + ".pth"
            cell = dict(dataset="RSAR", domain="clean", seed=42, method="B", role=role,
                        checkpoint=checkpoint, config="/fixed/config.py",
                        ann_file="/fixed/test/annfiles", img_prefix="/fixed/test/images",
                        training_code_sha="actual-producer", eval_dir=str(out))

            def native(command, **kwargs):
                kwargs["stdout"].write(
                    "| class | gts | dets | recall | ap |\n"
                    + "".join(f"| {c} | 1 | 1 | .1 | .1 |\n" for c in CLASSES["RSAR"])
                    + "| mAP | | | | .1 |\n")
                write_json(out / "predictions.pkl.image_ids.json", {
                    "checkpoint": checkpoint, "evaluation_code_sha": training.EVALUATION_SHA,
                    "n_images": EXPECTED_IMAGES["RSAR"]})
                return subprocess.CompletedProcess(command, 0)

            with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                    patch.object(extension_manifest.subprocess, "run", side_effect=native):
                extension_manifest.evaluate_binding(cell, runtime, 4)
            status = (out / "eval_status").read_text()
            self.assertIn(f"role={role}", status)
            self.assertIn(f"checkpoint={checkpoint}", status)
            self.assertEqual(json.loads((out / "execution.json").read_text())["role"], role)

    def test_unrelated_metadata_import_does_not_require_mmcv_or_opencv(self):
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.modules['mmcv']=None; "
             "from experiments.comparison import extension_training; "
             "assert 'B_REG' in extension_training.METHODS"],
            cwd=training.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_b_reg_uses_frozen_code_and_persists_only_regression_config_diff(self):
        (self.core / "train.py").write_text("# frozen B training entry; never executed\n")

        def b_spec(paths, dataset, overrides):
            self.assertNotIn("model.cfg.use_bbox_reg", overrides)
            self.assertEqual(overrides["data.samples_per_gpu"], 32)
            self.assertEqual(overrides["optimizer.lr"], .02)
            reference = self.core / f"{dataset}_B.py"
            return dict(
                method="B_REG", reference_config=str(reference), training_code=str(self.core),
                training_code_sha=B_CODE_SHA,
                overlay_text=f"_base_ = {str(reference)!r}\nmodel = dict(cfg=dict(use_bbox_reg=True))\n",
                config_diff=[dict(path="model.cfg.use_bbox_reg", before=False, after=True)],
                operational_note="Only experiment/output identity differs operationally")

        with patch.object(b_regression, "build_b_regression_spec", side_effect=b_spec) as build:
            self.prepare(("B_REG",))
        self.assertEqual(build.call_count, 36)
        audits = json.loads((self.queue / "b_regression_config_diff.json").read_text())
        self.assertEqual(len(audits), 36)
        self.assertTrue(all(a["config_diff"] == [
            dict(path="model.cfg.use_bbox_reg", before=False, after=True)] for a in audits.values()))
        cell = self.cell(method="B_REG")
        self.assertEqual(cell["training_code"], str(self.core))
        self.assertEqual(cell["training_code_sha"], B_CODE_SHA)
        self.assertEqual(cell["config"], str(self.queue / "b_reg_dior.py"))
        self.assertTrue(Path(cell["config"]).is_file())
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CGA_SCORER": "poison",
                                      "SARCLIP_LORA": "/not-allowed.pth"}), \
                patch.object(training, "code_sha",
                             side_effect=lambda code: B_CODE_SHA if code == self.core else "orchestrator"), \
                patch.object(training.subprocess, "run", side_effect=lambda *a, **kw: self.complete(cell)) as run:
            self.train(method="B_REG")
        self.assertEqual(run.call_args.args[0][1], str(self.core / "train.py"))
        self.assertEqual(run.call_args.kwargs["cwd"], self.core)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["PYTHONPATH"], str(self.core))
        self.assertEqual((env["CGA_SCORER"], env["CGA_BACKEND"], env["CGA_FILTER_MODE"]),
                         ("none", "none", "none"))
        self.assertNotIn("SARCLIP_LORA", env)
        execution = json.loads((Path(cell["method_dir"]) / "execution.json").read_text())
        self.assertEqual(execution["training_code_sha"], B_CODE_SHA)
        self.assertEqual(execution["orchestration_code_sha"], "orchestrator")
        with patch.object(training, "evaluate_binding") as evaluate:
            training.evaluate(self.queue, 4, "DIOR", "cloudy", 44, "B_REG")
        self.assertEqual(evaluate.call_args.args[0]["training_code_sha"], B_CODE_SHA)
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                patch.object(training, "code_sha", return_value="changed-frozen-code"), \
                self.assertRaisesRegex(ValueError, "frozen B training code changed"):
            self.train(domain="clean", seed=42, method="B_REG")

    def test_prepare_exact_selected_counts_paths_and_no_artifact_writes(self):
        original = {p: p.read_bytes() for p in self.core.iterdir() if p.is_file()}
        for methods in (("IRG",), ("LPLD", "SFUT"), training.PORT_METHODS):
            with self.subTest(methods=methods):
                self.queue = self.root / ("metadata " + "-".join(methods))
                with patch.object(training.subprocess, "run") as run, \
                        patch.object(Path, "glob", side_effect=AssertionError("No data scans")):
                    self.prepare(methods)
                run.assert_not_called()
                runtime = self.runtime()
                self.assertEqual(runtime["train_cells"], 36 * len(methods))
                self.assertEqual(runtime["eval_cells"], 36 * len(methods))
                self.assertEqual(runtime["source_training_cells"], 0)
                self.assertEqual(runtime["status"], "prepared_not_execution_evidence")
                self.assertEqual(runtime["python"], str(self.python))
                self.assertEqual(runtime["evaluation_code_sha"], training.EVALUATION_SHA)
                cells = runtime["cells"]
                self.assertEqual(json.loads((self.queue / "cells.json").read_text()), cells)
                lines = (self.queue / "train.list").read_text().splitlines()
                self.assertEqual(lines, (self.queue / "eval.list").read_text().splitlines())
                self.assertEqual(len(set(lines)), 36 * len(methods))
                self.assertTrue(all(len(line.split()) == 4 for line in lines))
                self.assertEqual({line.split()[3] for line in lines}, set(methods))
                admitted = load_cells([self.queue / "train.list"], [self.queue / "eval.list"])
                self.assertEqual(len(admitted), 36 * len(methods))
                self.assertTrue(all(admitted.values()))
                self.assertTrue(all(c.method in methods and c.role == "ema" and c.width == 1
                                    for c in admitted))
                resolver = load_resolver(self.queue / "paths.py")
                self.assertEqual(resolver.EXPECT_PRED, {"RSAR": 8538, "DIOR": 11738})
                self.assertEqual(resolver.EVALUATION_CODE_SHA, training.EVALUATION_SHA)
                for cell in cells.values():
                    ds, domain, seed, method = (cell[k] for k in ("dataset", "domain", "seed", "method"))
                    root = self.artifacts / ds.lower() / domain / f"seed_{seed}"
                    method_dir = root / "methods" / method
                    final = 266 if ds == "RSAR" else 185
                    args = (ds, domain, str(seed), method)
                    self.assertEqual(resolver.root(*args[:3]), str(root))
                    self.assertEqual(resolver.method_dir(*args), str(method_dir))
                    self.assertEqual(resolver.student_path(*args), str(method_dir / f"work/iter_{final}.pth"))
                    self.assertEqual(resolver.ema_path(*args), str(method_dir / f"work/iter_{final}_ema.pth"))
                    self.assertEqual(resolver.eval_full_dir(*args),
                                     str(method_dir / f"eval_full_{domain}_ids_v1"))
                    self.assertEqual(cell["terminal_status"], str(method_dir / "terminal_status"))
                    self.assertEqual(cell["source_checkpoint"], str(self.core / f"{ds}_source.pth"))
                    self.assertEqual(cell["source_id"], f"fixed-{ds}-source-id")
                    self.assertEqual(cell["source_seed"], 42)
                    self.assertEqual(cell["config"], str(training.ROOT /
                                     f"configs/unbiased_teacher/sfod/extensions/{method.lower()}_{ds.lower()}.py"))
                    self.assertEqual(cell["eval_config"], str(self.eval_code / f"{ds}_source.py"))
                    self.assertEqual(cell["unlabeled_epoch_size"], 8467 if ds == "RSAR" else 5863)
                    self.assertTrue(cell["target_val"].endswith("/val/images" if ds == "RSAR" else "/val"))
                self.assertEqual((self.queue / "with_gpu_lock.sh").resolve(),
                                 self.core / "with_gpu_lock.sh")
                for filename in ("run_train_1gpu.sh", "run_eval_full.sh", "with_gpu_lock.sh"):
                    subprocess.run(["bash", "-n", str(self.queue / filename)], check=True)
                for filename in ("run_train_1gpu.sh", "run_eval_full.sh"):
                    self.assertTrue(os.access(self.queue / filename, os.X_OK))
                    self.assertIn(str(training.ROOT), (self.queue / filename).read_text())
                    self.assertIn("export PYTHONNOUSERSITE=1", (self.queue / filename).read_text())
        self.assertFalse(self.artifacts.exists())
        self.assertEqual({p: p.read_bytes() for p in self.core.iterdir() if p.is_file()}, original)

    def test_cli_requires_explicit_methods(self):
        args = ["extension_training", "prepare"]
        for name in ("base-plan", "core-report", "core-paths", "out-dir", "artifact-root",
                     "eval-code", "python"):
            args += ["--" + name, "fixture"]
        with patch.object(sys, "argv", args), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            training.main()
        self.assertEqual(error.exception.code, 2)
        with patch.object(sys, "argv", args + ["--method", "IRG", "--method", "SFUT"]), \
                patch.object(training, "prepare") as prepare:
            prepare.return_value = None
            training.main()
        self.assertEqual(prepare.call_args.kwargs["methods"], ["IRG", "SFUT"])

    def test_rejects_unbound_inputs_before_metadata_creation(self):
        with self.assertRaisesRegex(ValueError, "Explicitly"):
            self.prepare(())
        report = json.loads(self.report.read_text())
        source = next(row for row in report["raw_results"] if row["method"] == "A")
        source["checkpoint"] = "/partial/iter_2.pth"
        write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.prepare()
        self.assertFalse(self.queue.exists())
        source["checkpoint"] = str(self.core / "RSAR_source.pth")
        write_json(self.report, report)
        (self.core / "RSAR_source.pth").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Missing nonempty"):
            self.prepare()
        self.assertFalse(self.queue.exists())

    def test_rejects_existing_roots_and_wrong_native_code(self):
        self.artifacts.mkdir()
        with self.assertRaisesRegex(ValueError, "NEW"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "separate"):
            self.prepare(artifact_root=self.queue / "nested")
        with patch.object(training, "code_sha", return_value="wrong-head"):
            with self.assertRaisesRegex(ValueError, "unchanged331d"):
                self.prepare(artifact_root=self.root / "other formal root")
        self.assertFalse(self.queue.exists())

    def test_rejects_nonseed42_plan_and_incomplete_report(self):
        report = json.loads(self.report.read_text())
        report["quantitative_complete_cells"] = 191
        write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "192-cell"):
            self.prepare()
        plan = json.loads(self.base.read_text())
        plan["adaptation_seed"] = 43
        for run in plan["runs"]:
            if run["method"] != "A":
                run["seed"] = 43
        write_json(self.base, plan)
        with self.assertRaisesRegex(ValueError, "seed42"):
            self.prepare()
        self.assertFalse(self.queue.exists())

    def test_target_val_only_known_mapping_and_existing_directories(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            training.target_val("RSAR", "clean", str(self.root / "arbitrary/images"))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            training.target_val("DIOR", "clean", str(self.root / "cloudy/test"))
        missing = self.root / "missing/clean/test"
        missing.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Missing image directory"):
            training.target_val("DIOR", "clean", str(missing))

    def test_train_all_methods_four_losses_env_and_exact_final_success(self):
        self.prepare(training.PORT_METHODS)
        ambient = {"IRAOD_GPU_LOCKED": "1", "CGA_ENABLE": "1", "SARCLIP_MODEL": "old",
                   "VLST_ENABLE": "1", "LD_LIBRARY_PATH": "/existing/lib", "DIOR_EXTRA": "regressionFalse"}
        for ds, domain in (("DIOR", "cloudy"), ("RSAR", "clean")):
            for method in training.PORT_METHODS:
                with self.subTest(dataset=ds, method=method):
                    cell = self.cell(ds, domain, method=method)
                    with patch.dict(os.environ, ambient), \
                            patch.object(training, "code_sha", return_value="actual-new-training-head"), \
                            patch.object(training.subprocess, "run",
                                         side_effect=lambda *a, **kw: self.complete(cell)) as run:
                        self.train(ds, domain, method=method)
                    command = run.call_args.args[0]
                    opts = command[command.index("--cfg-options") + 1:]
                    for option in (
                        "data.samples_per_gpu=32", "optimizer.lr=0.02",
                        "model.cfg.strict_source_free=True", "model.cfg.weight_l=0",
                        "model.cfg.weight_u=1", "model.cfg.use_bbox_reg=True",
                        "data.train.type=StrictSourceFreeDOTADataset",
                        "data.train.img_prefix=" + cell["target_val"],
                        "data.train.unlabeled_epoch_size=" + str(cell["unlabeled_epoch_size"]),
                        "load_from=" + cell["source_checkpoint"],
                        "model.ema_ckpt=" + cell["source_checkpoint"], "corrupt=" + domain,
                        "checkpoint_config.max_keep_ckpts=2", "checkpoint_config.save_last=True",
                    ):
                        self.assertIn(option, opts)
                    self.assertEqual(command[:3], [str(self.python), str(training.ROOT / "train.py"), cell["config"]])
                    self.assertIn("--no-validate", command)
                    self.assertIn("--deterministic", command)
                    self.assertEqual(command[command.index("--gpus") + 1], "1")
                    self.assertEqual(command[command.index("--seed") + 1], "44")
                    self.assertFalse(any("False" in value for value in opts))
                    self.assertFalse(any("ann_file" in value for value in opts))
                    self.assertFalse(any("lr_config" in value for value in opts))
                    env = run.call_args.kwargs["env"]
                    self.assertFalse(any(k.startswith(("CGA_", "SARCLIP_", "VLST_")) for k in env))
                    prefix = str(self.python.parent.parent)
                    self.assertEqual(env["CONDA_PREFIX"], prefix)
                    self.assertEqual(env["LD_LIBRARY_PATH"], prefix + "/lib:/existing/lib")
                    for key, value in {
                        "PYTHONPATH": str(training.ROOT), "CUDA_VISIBLE_DEVICES": "4",
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONNOUSERSITE": "1",
                        "PYTHONUNBUFFERED": "1", "IRAOD_RUNTIME_READY": "1",
                    }.items():
                        self.assertEqual(env[key], value)
                    self.assertEqual(run.call_args.kwargs["cwd"], training.ROOT)
                    self.assertEqual(run.call_args.kwargs["stderr"], subprocess.STDOUT)
                    self.assertTrue((Path(cell["method_dir"]) / "train.log").is_file())
                    execution = json.loads((Path(cell["method_dir"]) / "execution.json").read_text())
                    self.assertEqual(execution["training_code_sha"], "actual-new-training-head")
                    self.assertEqual(execution["source_id"], cell["source_id"])
                    self.assertEqual(execution["command"], command)
                    resolver = load_resolver(self.queue / "paths.py")
                    self.assertEqual(train_state(self.queue, resolver, Cell(ds, domain, 44, method)), "complete")
                    with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                            patch.object(training.subprocess, "run") as run:
                        with self.assertRaises(FileExistsError):
                            self.train(ds, domain, method=method)
                    run.assert_not_called()

    def test_train_requires_approved_gpu_and_lock_without_writes(self):
        self.prepare()
        with patch.object(training.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "GPU4-7"):
                self.train(gpu=3)
            with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": ""}):
                with self.assertRaisesRegex(RuntimeError, "shared GPU lock"):
                    self.train()
        run.assert_not_called()
        self.assertFalse(self.artifacts.exists())

    def test_train_nonzero_missing_empty_or_smoke_finals_cannot_succeed(self):
        self.prepare(training.PORT_METHODS)
        cases = ((42, "IRG", "nonzero"), (43, "IRG", "missing"), (44, "IRG", "smoke"),
                 (42, "LPLD", "empty-ema"), (43, "LPLD", "missing-student"), (44, "LPLD", "spawn"))
        for seed, method, case in cases:
            with self.subTest(case=case):
                cell = self.cell(seed=seed, method=method)

                def execute(*args, **kwargs):
                    if case == "spawn":
                        raise OSError("fixture spawn failure")
                    if case == "nonzero":
                        self.complete(cell)
                        return subprocess.CompletedProcess([], 9)
                    if case == "smoke":
                        for suffix in ("", "_ema"):
                            (Path(cell["work_dir"]) / f"iter_2{suffix}.pth").write_bytes(b"smoke")
                    if case == "empty-ema":
                        Path(cell["student_checkpoint"]).write_bytes(b"student")
                        Path(cell["checkpoint"]).touch()
                    if case == "missing-student":
                        Path(cell["checkpoint"]).write_bytes(b"ema")
                    return subprocess.CompletedProcess([], 0)

                error_type = (subprocess.CalledProcessError if case == "nonzero"
                              else OSError if case == "spawn" else ValueError)
                with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                        patch.object(training.subprocess, "run", side_effect=execute):
                    with self.assertRaises(error_type):
                        self.train(seed=seed, method=method)
                expected_rc = 9 if case == "nonzero" else 1
                self.assertEqual(Path(cell["terminal_status"]).read_text(),
                                 f"tmux_wrap_exit={expected_rc}\n")
                resolver = load_resolver(self.queue / "paths.py")
                self.assertNotEqual(train_state(self.queue, resolver, Cell("DIOR", "cloudy", seed, method)),
                                    "complete")

    def test_eval_delegates_native_ema_with_actual_training_provenance(self):
        self.prepare()
        cell = self.cell()
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}), \
                patch.object(training, "code_sha", return_value="actual-producer-after-prepare"), \
                patch.object(training.subprocess, "run", side_effect=lambda *a, **kw: self.complete(cell)):
            self.train()
        with patch.object(training, "evaluate_binding") as native:
            training.evaluate(self.queue, 5, "DIOR", "cloudy", 44, "IRG")
        binding, runtime, gpu = native.call_args.args
        self.assertEqual(gpu, 5)
        self.assertEqual(binding["role"], "ema")
        self.assertEqual(binding["checkpoint"], cell["checkpoint"])
        self.assertEqual(binding["config"], cell["eval_config"])
        self.assertEqual(binding["source_id"], cell["source_id"])
        self.assertEqual(binding["training_code_sha"], "actual-producer-after-prepare")
        self.assertEqual(binding["ann_file"], cell["ann_file"])
        self.assertEqual(binding["img_prefix"], cell["img_prefix"])
        self.assertEqual(binding["eval_dir"], cell["eval_dir"])
        self.assertEqual(runtime["python"], str(self.python))
        self.assertEqual(runtime["evaluation_code"], str(self.eval_code))
        self.assertEqual(runtime["evaluation_code_sha"], training.EVALUATION_SHA)
        self.assertFalse(Path(cell["eval_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
