from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.cga_research import build_data_manifest
from tools.cga_research import evaluate_final_test_predictions as final_evaluator
from tools.cga_research.evaluate_final_test_predictions import (
    FinalTestEvaluationError,
    REQUIRED_RUNTIME_FILES,
    _atomic_create_json,
    _manifest_bound_test_config,
    _reverify_prediction_identity,
    authorize_final_test_run,
    run,
    validate_final_test_dataset_paths,
)
from tools.cga_research.final_test_manifest import (
    FROZEN_STATE,
    ManifestError,
    create_manifest,
    sha256_file,
    sidecar_path,
)


class FinalTestPredictionsAuthorizationTests(unittest.TestCase):
    METHODS = ("no_cga", "candidate")
    SEED = 41

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="iraod_final_predictions_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.annotation_root = self.root / "dataset" / "test" / "annfiles"
        self.image_root = self.root / "dataset" / "test" / "images"
        self.annotation_root.mkdir(parents=True)
        self.image_root.mkdir(parents=True)
        (self.annotation_root / "sample.txt").write_text(
            "0 0 4 0 4 2 0 2 ship 0\n", encoding="utf-8"
        )
        (self.image_root / "sample.png").write_bytes(b"synthetic image")
        config = (
            "classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')\n"
            "model = dict()\n"
            "data = dict(test=dict(\n"
            "    type='DOTADataset',\n"
            f"    ann_file={str(self.annotation_root)!r},\n"
            f"    img_prefix={str(self.image_root)!r},\n"
            "    classes=classes, pipeline=[], version='le90'))\n"
        )
        self.config = self._write("configs/final.py", config.encode("utf-8"))
        self.evidence = self._write("reports/selection.json", b'{"fixed":true}\n')
        self.data_manifest = self.root / "state" / "data_manifest.json"
        build_data_manifest.build_manifest(
            ann_root=self.annotation_root,
            image_root=self.image_root,
            split="test",
            corruption="chaff",
            class_order=(
                "ship",
                "aircraft",
                "car",
                "tank",
                "bridge",
                "harbor",
            ),
            output=self.data_manifest,
        )
        self.runtime_marker = self._write(
            "state/runtime_marker.txt", b"frozen runtime\n"
        )
        self.draft = self.root / "draft.json"
        self.manifest = self.root / "frozen" / "final_test_manifest.json"
        self.payload = {
            "schema_version": 2,
            "state": FROZEN_STATE,
            "project_root": str(self.root),
            "arms": [self._arm(method) for method in self.METHODS],
            "aggregation": {
                "required_seed_set": [self.SEED],
                "paired_comparator": "no_cga",
                "exclude_failed_seed": False,
                "mean": True,
                "sample_std_ddof": 1,
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260715,
                "paired_t_test": True,
                "exact_sign_flip_permutation": True,
                "cohens_dz": True,
                "holm_correction": True,
            },
            "selection": {
                "evidence": [self._artifact(self.evidence)],
                "config": self._artifact(self.config),
                "runtime_files": [
                    self._artifact(path)
                    for path in sorted(
                        set(REQUIRED_RUNTIME_FILES).union(
                            final_evaluator.runner_tool.REQUIRED_RUNTIME_FILES
                        ),
                        key=str,
                    )
                ]
                + [self._artifact(self.runtime_marker)],
                "data_manifest": self._artifact(self.data_manifest),
            },
        }
        self._freeze()
        self.selected_output = (
            self.root / self.payload["arms"][0]["runs"][0]["output_dir"]
        )
        self.selected_output.mkdir(parents=True)
        self.predictions = self.selected_output / "predictions.pkl"
        self.predictions.write_bytes(b"trusted generated predictions")
        self.output = self.selected_output / "per_class_ap.json"

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _artifact(self, path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256_file(path)}

    def _run_entry(self, method: str) -> dict[str, object]:
        checkpoint = (
            self.root / "runs" / method / f"seed_{self.SEED}" / "iter_4235_ema.pth"
        )
        checkpoint.parent.mkdir(parents=True)
        torch.save(
            {
                "meta": {"iter": 4235, "epoch": 1, "seed": self.SEED},
                "state_dict": {},
            },
            checkpoint,
        )
        return {
            "seed": self.SEED,
            "training_fingerprint": hashlib.sha256(
                f"{method}:{self.SEED}".encode("utf-8")
            ).hexdigest(),
            "checkpoint": {
                "path": str(checkpoint.relative_to(self.root)),
                "size_bytes": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "meta": {"iter": 4235, "epoch": 1, "seed": self.SEED},
            },
            "output_dir": str(
                (
                    self.manifest.parent
                    / final_evaluator.runner_tool.OUTPUT_ROOT_NAME
                    / method
                    / f"seed_{self.SEED}"
                ).relative_to(self.root)
            ),
        }

    def _arm(self, method: str) -> dict[str, object]:
        return {
            "method": method,
            "method_environment": {"CGA_SCORER": "none"},
            "hyperparameters": {"corrupt": "chaff"},
            "runs": [self._run_entry(method)],
        }

    def _freeze(self) -> None:
        self.draft.write_text(json.dumps(self.payload), encoding="utf-8")
        create_manifest(self.draft, self.manifest)

    def _freeze_with_data_manifest(
        self,
        *,
        name: str,
        split: str = "test",
        corruption: str = "chaff",
        class_order=("ship", "aircraft", "car", "tank", "bridge", "harbor"),
    ) -> Path:
        data_manifest = self.root / "state" / f"{name}_data_manifest.json"
        build_data_manifest.build_manifest(
            ann_root=self.annotation_root,
            image_root=self.image_root,
            split=split,
            corruption=corruption,
            class_order=class_order,
            output=data_manifest,
        )
        payload = json.loads(json.dumps(self.payload))
        payload["selection"]["data_manifest"] = self._artifact(data_manifest)
        for arm in payload["arms"]:
            arm["runs"][0][
                "output_dir"
            ] = f"{name}_outputs/{arm['method']}/seed_{self.SEED}"
        draft = self.root / "state" / f"{name}_draft.json"
        manifest = self.root / "state" / f"{name}_manifest.json"
        draft.write_text(json.dumps(payload), encoding="utf-8")
        create_manifest(draft, manifest)
        return manifest

    def _authorize(self, **overrides):
        arguments = {
            "manifest_path": self.manifest,
            "method": "no_cga",
            "seed": self.SEED,
            "predictions_path": self.predictions,
            "output_path": self.output,
        }
        arguments.update(overrides)
        with (
            mock.patch.object(
                final_evaluator.runner_tool,
                "registry_path_for_manifest",
                create=True,
                return_value=self.manifest.parent / "final_test_registry.jsonl",
            ),
            mock.patch.object(
                final_evaluator.runner_tool,
                "verify_completed_run",
                create=True,
                return_value=self._completed_record(),
            ),
        ):
            return authorize_final_test_run(**arguments)

    def _completed_record(self) -> dict[str, object]:
        selected_run = self.payload["arms"][0]["runs"][0]
        checkpoint = selected_run["checkpoint"]
        return {
            "event": "completed",
            "manifest_path": str(self.manifest),
            "manifest_sha256": sha256_file(self.manifest),
            "method": "no_cga",
            "seed": self.SEED,
            "training_fingerprint": selected_run["training_fingerprint"],
            "checkpoint_path": str((self.root / checkpoint["path"]).resolve()),
            "checkpoint_sha256": checkpoint["sha256"],
            "output_dir": str(self.selected_output),
            "returncode": 0,
            "attempt": 1,
            "artifacts": {
                "predictions.pkl": {
                    "path": str(self.predictions),
                    "sha256": sha256_file(self.predictions),
                    "size_bytes": self.predictions.stat().st_size,
                }
            },
        }

    def test_exact_frozen_method_seed_and_paths_are_authorized(self) -> None:
        authorization = self._authorize()
        self.assertEqual(authorization.method, "no_cga")
        self.assertEqual(authorization.seed, self.SEED)
        self.assertEqual(authorization.predictions_path, self.predictions)
        self.assertEqual(authorization.output_path, self.output)
        self.assertEqual(len(authorization.manifest_sha256), 64)

    def test_method_and_seed_must_uniquely_match_frozen_run(self) -> None:
        with self.assertRaisesRegex(FinalTestEvaluationError, "found 0"):
            self._authorize(method="missing")
        with self.assertRaisesRegex(FinalTestEvaluationError, "found 0"):
            self._authorize(seed=42)

    def test_predictions_must_be_exact_registered_basename_and_not_symlink(
        self,
    ) -> None:
        outside = self._write("outside/predictions.pkl", b"wrong")
        renamed = self._write("final_outputs/no_cga/seed_41/renamed.pkl", b"wrong")
        for path in (outside, renamed):
            with self.subTest(path=path), self.assertRaisesRegex(
                FinalTestEvaluationError, "must exactly equal"
            ):
                self._authorize(predictions_path=path)

        self.predictions.unlink()
        self.predictions.symlink_to(outside)
        with self.assertRaisesRegex(FinalTestEvaluationError, "must not be a symlink"):
            self._authorize()

    def test_completed_registry_identity_is_required_and_rechecked(self) -> None:
        with (
            mock.patch.object(
                final_evaluator.runner_tool,
                "registry_path_for_manifest",
                create=True,
                return_value=self.manifest.parent / "final_test_registry.jsonl",
            ),
            mock.patch.object(
                final_evaluator.runner_tool,
                "verify_completed_run",
                create=True,
                side_effect=final_evaluator.runner_tool.FinalTestBundleError(
                    "no completed record"
                ),
            ),
            self.assertRaisesRegex(
                FinalTestEvaluationError, "runner completion verification failed"
            ),
        ):
            authorize_final_test_run(
                manifest_path=self.manifest,
                method="no_cga",
                seed=self.SEED,
                predictions_path=self.predictions,
                output_path=self.output,
            )

        authorization = self._authorize()
        self.predictions.write_bytes(b"replacement predictions")
        with self.assertRaisesRegex(
            FinalTestEvaluationError, "changed after runner completion"
        ):
            _reverify_prediction_identity(authorization)

    def test_verified_snapshot_fd_defeats_path_replacement_before_pickle(self) -> None:
        authorized_value = {"source": "authorized inode"}
        replacement_value = {"source": "replacement path"}
        authorized_bytes = pickle.dumps(authorized_value)
        self.predictions.write_bytes(authorized_bytes)
        authorization = self._authorize()
        replacement = self._write(
            "replacement/predictions.pkl", pickle.dumps(replacement_value)
        )
        original_load = pickle.load

        def replace_path_then_load(handle):
            os.replace(replacement, self.predictions)
            return original_load(handle)

        with mock.patch.object(
            final_evaluator.pickle,
            "load",
            side_effect=replace_path_then_load,
        ):
            loaded, digest, size = final_evaluator._load_verified_prediction_pickle(
                authorization
            )

        self.assertEqual(loaded, authorized_value)
        self.assertEqual(digest, hashlib.sha256(authorized_bytes).hexdigest())
        self.assertEqual(size, len(authorized_bytes))
        self.assertEqual(pickle.loads(self.predictions.read_bytes()), replacement_value)

    def test_post_authorization_symlink_replacement_is_rejected(self) -> None:
        self.predictions.write_bytes(pickle.dumps({"source": "authorized"}))
        authorization = self._authorize()
        outside = self._write(
            "replacement/outside.pkl", pickle.dumps({"source": "outside"})
        )
        self.predictions.unlink()
        self.predictions.symlink_to(outside)

        with self.assertRaisesRegex(
            FinalTestEvaluationError, "cannot securely open authorized predictions"
        ):
            final_evaluator._load_verified_prediction_pickle(authorization)

    def test_registry_path_is_fixed_next_to_manifest(self) -> None:
        with (
            mock.patch.object(
                final_evaluator.runner_tool,
                "registry_path_for_manifest",
                create=True,
                return_value=self.root / "alternate_registry.jsonl",
            ),
            mock.patch.object(
                final_evaluator.runner_tool,
                "verify_completed_run",
                create=True,
                return_value=self._completed_record(),
            ),
            self.assertRaisesRegex(
                FinalTestEvaluationError, "fixed manifest-local registry"
            ),
        ):
            authorize_final_test_run(
                manifest_path=self.manifest,
                method="no_cga",
                seed=self.SEED,
                predictions_path=self.predictions,
                output_path=self.output,
            )

    def test_actual_runner_completion_is_bound_and_postprocess_is_resumable(
        self,
    ) -> None:
        runner = final_evaluator.runner_tool
        shutil.rmtree(self.selected_output)
        bundle = runner.load_frozen_bundle(self.manifest, verify_data=True)
        selected_run = next(item for item in bundle.runs if item.method == "no_cga")
        registry = runner.AppendOnlyRegistry(
            runner.registry_path_for_manifest(self.manifest)
        )
        gpu = runner.gpu_scheduler.GPUInfo(
            index=0,
            uuid="GPU-synthetic-0",
            name="Synthetic GPU",
            memory_used_mib=0,
            memory_free_mib=24000,
            memory_total_mib=24000,
            utilization_percent=0,
        )

        class FakeProcess:
            pid = 4321

            @staticmethod
            def poll():
                return 0

            @staticmethod
            def wait(timeout=None):
                del timeout
                return 0

            @staticmethod
            def terminate():
                return None

            @staticmethod
            def kill():
                return None

        def successful_process(command, **kwargs):
            prediction_path = Path(command[command.index("--out") + 1])
            prediction_path.write_bytes(b"runner-produced predictions")
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "eval_20260715_123456.json").write_text(
                '{"mAP": 0.5}\n', encoding="utf-8"
            )
            kwargs["stdout"].write(b"synthetic test.py completed\n")
            return FakeProcess()

        def compute_apps():
            return [
                runner.run_experiment.ComputeApp(
                    pid=FakeProcess.pid,
                    gpu_uuid=gpu.uuid,
                    used_memory_mib=128,
                )
            ]

        completed = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=selected_run,
            gpu=gpu,
            registry=registry,
            python=Path(sys.executable),
            max_attempts=1,
            base_environment={},
            popen_factory=successful_process,
            compute_apps_probe=compute_apps,
            process_start_probe=lambda pid: 123456,
            monitor_interval=0.01,
            run_timeout=10,
            stall_timeout=10,
            terminate_grace=0.01,
        )
        self.assertEqual(completed.status, "completed")

        authorization = authorize_final_test_run(
            manifest_path=self.manifest,
            method="no_cga",
            seed=self.SEED,
            predictions_path=self.predictions,
            output_path=self.output,
        )
        self.assertEqual(authorization.completion_record["event"], "completed")
        _atomic_create_json(self.output, {"status": "synthetic completed"})
        verified_after_postprocess = runner.verify_completed_run(
            self.manifest, "no_cga", self.SEED
        )
        self.assertEqual(verified_after_postprocess["event"], "completed")

        skipped = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=selected_run,
            gpu=gpu,
            registry=registry,
            python=Path(sys.executable),
            max_attempts=1,
            base_environment={},
            popen_factory=mock.Mock(
                side_effect=AssertionError("completed run was relaunched")
            ),
            compute_apps_probe=mock.Mock(
                side_effect=AssertionError("completed run queried GPU apps")
            ),
            process_start_probe=lambda pid: 123456,
        )
        self.assertEqual(skipped.status, "skipped_verified")

        self.predictions.unlink()
        self.predictions.write_bytes(b"tampered canonical predictions")
        with self.assertRaisesRegex(
            FinalTestEvaluationError, "changed after runner completion"
        ):
            _reverify_prediction_identity(authorization)
        with self.assertRaises(final_evaluator.runner_tool.FinalTestBundleError):
            runner.verify_completed_run(self.manifest, "no_cga", self.SEED)

    def test_output_must_be_new_json_directly_inside_selected_output_dir(self) -> None:
        outside = self.root / "outside.json"
        nested = self.selected_output / "nested" / "result.json"
        wrong_suffix = self.selected_output / "result.txt"
        wrong_basename = self.selected_output / "result.json"
        for path, message in (
            (outside, "direct child"),
            (nested, "direct child"),
            (wrong_suffix, "end with .json"),
            (wrong_basename, "basename must equal"),
        ):
            with self.subTest(path=path), self.assertRaisesRegex(
                FinalTestEvaluationError, message
            ):
                self._authorize(output_path=path)

        self.output.write_text("already exists", encoding="utf-8")
        with self.assertRaisesRegex(FinalTestEvaluationError, "must be a new"):
            self._authorize()

    def test_atomic_result_publication_never_replaces_existing_evidence(self) -> None:
        result = self.root / "atomic" / "result.json"
        result.parent.mkdir()
        _atomic_create_json(result, {"mAP": 0.5})
        original = result.read_bytes()
        with self.assertRaisesRegex(FinalTestEvaluationError, "refusing to replace"):
            _atomic_create_json(result, {"mAP": 0.9})
        self.assertEqual(result.read_bytes(), original)

    def test_manifest_sidecar_and_frozen_artifact_drift_fail(self) -> None:
        sidecar = sidecar_path(self.manifest)
        original_sidecar = sidecar.read_bytes()
        sidecar.write_text("0" * 64 + "  final_test_manifest.json\n")
        with self.assertRaisesRegex(ManifestError, "sidecar mismatch"):
            self._authorize()
        sidecar.write_bytes(original_sidecar)

        self.runtime_marker.write_text("drifted", encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "runtime_files.*mismatch"):
            self._authorize()

    def test_required_runtime_tools_must_be_frozen(self) -> None:
        with tempfile.TemporaryDirectory(prefix="iraod_missing_runtime_") as temporary:
            other_root = Path(temporary).resolve()
            payload = json.loads(json.dumps(self.payload))
            payload["project_root"] = str(self.root)
            payload["selection"]["runtime_files"] = [
                self._artifact(self.runtime_marker)
            ]
            for arm in payload["arms"]:
                arm["runs"][0][
                    "output_dir"
                ] = f"missing_runtime_outputs/{arm['method']}/seed_{self.SEED}"
            draft = other_root / "draft.json"
            manifest = other_root / "manifest.json"
            draft.write_text(json.dumps(payload), encoding="utf-8")
            create_manifest(draft, manifest)
            with self.assertRaisesRegex(
                FinalTestEvaluationError, "does not register required runtime files"
            ):
                authorize_final_test_run(
                    manifest_path=manifest,
                    method="no_cga",
                    seed=self.SEED,
                    predictions_path=self.predictions,
                    output_path=self.output,
                )

    def test_pickle_loader_is_not_reached_when_full_verify_fails(self) -> None:
        self.runtime_marker.write_text("drifted before evaluation", encoding="utf-8")
        args = argparse.Namespace(
            manifest=self.manifest,
            method="no_cga",
            seed=self.SEED,
            predictions=self.predictions,
            output=self.output,
            trust_pickle=True,
            nproc=1,
        )
        with mock.patch(
            "tools.cga_research.evaluate_final_test_predictions."
            "_load_verified_prediction_pickle"
        ) as loader:
            with self.assertRaises(ManifestError):
                run(args)
            loader.assert_not_called()

    def test_dataset_byte_drift_fails_before_pickle_loading(self) -> None:
        (self.annotation_root / "sample.txt").write_text(
            "0 0 4 0 4 2 0 2 aircraft 0\n", encoding="utf-8"
        )
        args = argparse.Namespace(
            manifest=self.manifest,
            method="no_cga",
            seed=self.SEED,
            predictions=self.predictions,
            output=self.output,
            trust_pickle=True,
            nproc=1,
        )
        with mock.patch(
            "tools.cga_research.evaluate_final_test_predictions."
            "_load_verified_prediction_pickle"
        ) as loader:
            with self.assertRaisesRegex(
                FinalTestEvaluationError, "dataset manifest verification failed"
            ):
                run(args)
            loader.assert_not_called()

    def test_data_manifest_semantics_are_bound_to_arm_and_rsar_order(self) -> None:
        wrong_split = self._freeze_with_data_manifest(name="wrong_split", split="val")
        with self.assertRaisesRegex(
            FinalTestEvaluationError, "split must equal 'test'"
        ):
            authorize_final_test_run(
                manifest_path=wrong_split,
                method="no_cga",
                seed=self.SEED,
                predictions_path=self.predictions,
                output_path=self.output,
            )

        wrong_corruption = self._freeze_with_data_manifest(
            name="wrong_corruption", corruption="fog"
        )
        with self.assertRaisesRegex(
            FinalTestEvaluationError, "corruption must exactly match"
        ):
            authorize_final_test_run(
                manifest_path=wrong_corruption,
                method="no_cga",
                seed=self.SEED,
                predictions_path=self.predictions,
                output_path=self.output,
            )

        wrong_order = self._freeze_with_data_manifest(
            name="wrong_order",
            class_order=("aircraft", "ship", "car", "tank", "bridge", "harbor"),
        )
        with self.assertRaisesRegex(FinalTestEvaluationError, "class_order mismatch"):
            authorize_final_test_run(
                manifest_path=wrong_order,
                method="no_cga",
                seed=self.SEED,
                predictions_path=self.predictions,
                output_path=self.output,
            )

    def test_manifest_roots_override_mismatched_config_paths(self) -> None:
        alternate_ann = self.root / "alternate" / "test" / "annfiles"
        alternate_images = self.root / "alternate" / "test" / "images"
        alternate_ann.mkdir(parents=True)
        alternate_images.mkdir(parents=True)
        bound, ann_file, img_prefix = _manifest_bound_test_config(
            {
                "type": "DOTADataset",
                "ann_file": str(alternate_ann),
                "img_prefix": str(alternate_images),
                "classes": ("wrong",),
            },
            annotation_root=self.annotation_root,
            image_root=self.image_root,
        )
        self.assertEqual(ann_file, self.annotation_root)
        self.assertEqual(img_prefix, self.image_root)
        self.assertEqual(bound["ann_file"], str(self.annotation_root))
        self.assertEqual(bound["img_prefix"], str(self.image_root))
        self.assertEqual(
            bound["classes"],
            ("ship", "aircraft", "car", "tank", "bridge", "harbor"),
        )
        with self.assertRaisesRegex(FinalTestEvaluationError, "dataset wrappers"):
            _manifest_bound_test_config(
                {
                    "type": "ConcatDataset",
                    "datasets": [
                        {
                            "type": "DOTADataset",
                            "ann_file": str(alternate_ann),
                            "img_prefix": str(alternate_images),
                        }
                    ],
                },
                annotation_root=self.annotation_root,
                image_root=self.image_root,
            )

    def test_final_dataset_paths_require_exact_test_component(self) -> None:
        test_ann = self.root / "dataset" / "test" / "annfiles"
        test_img = self.root / "dataset" / "test" / "images"
        val_ann = self.root / "dataset" / "val" / "annfiles"
        for path in (test_ann, test_img, val_ann):
            path.mkdir(parents=True, exist_ok=True)

        ann, img = validate_final_test_dataset_paths(
            {"ann_file": str(test_ann), "img_prefix": str(test_img)}
        )
        self.assertEqual(ann, test_ann)
        self.assertEqual(img, test_img)
        with self.assertRaisesRegex(FinalTestEvaluationError, "exact 'test'"):
            validate_final_test_dataset_paths(
                {"ann_file": str(val_ann), "img_prefix": str(test_img)}
            )


if __name__ == "__main__":
    unittest.main()
