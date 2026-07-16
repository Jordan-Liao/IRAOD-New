from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.cga_research import aggregate_final_test_results as aggregate_tool
from tools.cga_research import build_data_manifest
from tools.cga_research import evaluate_final_test_predictions as evaluator_tool
from tools.cga_research import final_test_manifest as manifest_tool
from tools.cga_research import run_final_test_bundle as runner_tool


class SyntheticFinalTestAggregationTests(unittest.TestCase):
    METHODS = ("no_cga", "legacy", "bounded_penalty")
    SEEDS = (41, 42, 43)
    MAPS = {
        "no_cga": (0.600, 0.620, 0.640),
        "legacy": (0.610, 0.625, 0.650),
        "bounded_penalty": (0.605, 0.630, 0.655),
    }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="iraod_synthetic_final_aggregate_"
        )
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.annotation_root = self.root / "synthetic_dataset" / "test" / "annfiles"
        self.image_root = self.root / "synthetic_dataset" / "test" / "images"
        self.annotation_root.mkdir(parents=True)
        self.image_root.mkdir(parents=True)
        (self.annotation_root / "sample.txt").write_text(
            "0 0 4 0 4 2 0 2 ship 0\n", encoding="utf-8"
        )
        (self.image_root / "sample.png").write_bytes(b"synthetic image only")
        self.config = self._write(
            "configs/synthetic_final.py",
            (
                "classes = ('ship', 'aircraft', 'car', 'tank', 'bridge', "
                "'harbor')\n"
                "model = dict()\n"
                "data = dict(test=dict(type='DOTADataset', "
                f"ann_file={str(self.annotation_root)!r}, "
                f"img_prefix={str(self.image_root)!r}, classes=classes, "
                "pipeline=[], version='le90'))\n"
            ).encode("utf-8"),
        )
        self.evidence = self._write(
            "reports/preregistered_selection.json", b'{"fixed":true}\n'
        )
        self.data_manifest = self.root / "state" / "data_manifest.json"
        build_data_manifest.build_manifest(
            ann_root=self.annotation_root,
            image_root=self.image_root,
            split="test",
            corruption="chaff",
            class_order=aggregate_tool.RSAR_CLASSES,
            output=self.data_manifest,
        )
        self.manifest = self.root / "frozen" / "final_test_manifest.json"
        self.draft = self.root / "state" / "draft.json"
        self.payload = self._payload()
        self.draft.write_text(json.dumps(self.payload), encoding="utf-8")
        manifest_tool.create_manifest(self.draft, self.manifest)
        self.manifest_sha256 = manifest_tool.sha256_file(self.manifest)
        self.completions: dict[tuple[str, int], dict[str, object]] = {}
        self.report_paths: dict[tuple[str, int], Path] = {}
        for arm in self.payload["arms"]:
            for index, run in enumerate(arm["runs"]):
                self._create_completed_run(arm, run, self.MAPS[arm["method"]][index])

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _artifact(self, path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": manifest_tool.sha256_file(path)}

    def _checkpoint(self, method: str, seed: int) -> dict[str, object]:
        path = self.root / "runs" / method / f"seed_{seed}" / "iter_4235_ema.pth"
        path.parent.mkdir(parents=True)
        torch.save(
            {
                "meta": {"iter": 4235, "epoch": 1, "seed": seed},
                "state_dict": {},
            },
            path,
        )
        return {
            "path": str(path.relative_to(self.root)),
            "size_bytes": path.stat().st_size,
            "sha256": manifest_tool.sha256_file(path),
            "meta": {"iter": 4235, "epoch": 1, "seed": seed},
        }

    def _arm(self, method: str) -> dict[str, object]:
        environment = (
            {"CGA_SCORER": "none"}
            if method == "no_cga"
            else {"CGA_SCORER": "sarclip", "CGA_FILTER_MODE": method}
        )
        return {
            "method": method,
            "method_environment": environment,
            "hyperparameters": {"corrupt": "chaff", "model.cfg.score_thr": 0.9},
            "runs": [
                {
                    "seed": seed,
                    "training_fingerprint": self._hash_text(f"{method}:{seed}"),
                    "checkpoint": self._checkpoint(method, seed),
                    "output_dir": str(
                        (
                            self.manifest.parent
                            / runner_tool.OUTPUT_ROOT_NAME
                            / method
                            / f"seed_{seed}"
                        ).relative_to(self.root)
                    ),
                }
                for seed in self.SEEDS
            ],
        }

    def _payload(self) -> dict[str, object]:
        runtime_files = set(aggregate_tool.REQUIRED_RUNTIME_FILES)
        runtime_files.update(evaluator_tool.REQUIRED_RUNTIME_FILES)
        runtime_files.update(runner_tool.REQUIRED_RUNTIME_FILES)
        return {
            "schema_version": 2,
            "state": manifest_tool.FROZEN_STATE,
            "project_root": str(self.root),
            "arms": [self._arm(method) for method in self.METHODS],
            "aggregation": {
                "required_seed_set": list(self.SEEDS),
                "paired_comparator": "no_cga",
                "exclude_failed_seed": False,
                "mean": True,
                "sample_std_ddof": 1,
                "bootstrap_samples": 1000,
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
                    self._artifact(path) for path in sorted(runtime_files, key=str)
                ],
                "data_manifest": self._artifact(self.data_manifest),
            },
        }

    def _create_completed_run(
        self,
        arm: dict[str, object],
        run: dict[str, object],
        mean_ap: float,
    ) -> None:
        method = arm["method"]
        seed = run["seed"]
        output_dir = self.root / run["output_dir"]
        output_dir.mkdir(parents=True)
        predictions = output_dir / runner_tool.PREDICTIONS_BASENAME
        predictions.write_bytes(f"synthetic:{method}:{seed}".encode("utf-8"))
        prediction_identity = {
            "path": str(predictions),
            "sha256": manifest_tool.sha256_file(predictions),
            "size_bytes": predictions.stat().st_size,
        }
        completion = {
            "event": "completed",
            "manifest_path": str(self.manifest),
            "manifest_sha256": self.manifest_sha256,
            "method": method,
            "seed": seed,
            "training_fingerprint": run["training_fingerprint"],
            "checkpoint_path": str(self.root / run["checkpoint"]["path"]),
            "checkpoint_sha256": run["checkpoint"]["sha256"],
            "output_dir": str(output_dir),
            "returncode": 0,
            "attempt": 1,
            "artifacts": {
                runner_tool.PREDICTIONS_BASENAME: prediction_identity,
                runner_tool.STDOUT_BASENAME: {
                    "path": str(output_dir / runner_tool.STDOUT_BASENAME),
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                },
                runner_tool.ONLINE_METADATA_BASENAME: {
                    "path": str(output_dir / runner_tool.ONLINE_METADATA_BASENAME),
                    "sha256": "1" * 64,
                    "size_bytes": 1,
                },
                runner_tool.FRAMEWORK_EVAL_BASENAME: {
                    "path": str(output_dir / runner_tool.FRAMEWORK_EVAL_BASENAME),
                    "sha256": "2" * 64,
                    "size_bytes": 1,
                },
            },
            "gpu_index": seed % 2,
            "gpu_uuid": f"GPU-synthetic-{seed % 2}",
            "gpu_name": "Synthetic GPU",
            "gpu_verified": True,
            "recorded_at_utc": "2026-07-15T00:00:00+00:00",
            "run_identity": self._hash_text(f"run:{method}:{seed}"),
        }
        report_path = output_dir / runner_tool.POSTPROCESS_BASENAME
        report = self._report(arm, run, completion, report_path, mean_ap)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        identity = (method, seed)
        self.completions[identity] = completion
        self.report_paths[identity] = report_path

    def _report(
        self,
        arm: dict[str, object],
        run: dict[str, object],
        completion: dict[str, object],
        report_path: Path,
        mean_ap: float,
    ) -> dict[str, object]:
        method = arm["method"]
        seed = run["seed"]
        offsets = (-0.05, -0.03, -0.01, 0.01, 0.03, 0.05)
        detections = {
            class_name: 10 + index
            for index, class_name in enumerate(aggregate_tool.RSAR_CLASSES)
        }
        per_class = [
            {
                "index": index,
                "class": class_name,
                "ap": mean_ap + offsets[index],
                "num_gts": 2,
                "num_dets": detections[class_name],
                "final_recall": 0.5 + index * 0.05,
            }
            for index, class_name in enumerate(aggregate_tool.RSAR_CLASSES)
        ]
        runtime = evaluator_tool.REQUIRED_RUNTIME_FILES
        predictions = completion["artifacts"][runner_tool.PREDICTIONS_BASENAME]
        return {
            "schema_version": 1,
            "generated_at_utc": "2026-07-15T00:00:00+00:00",
            "status": "completed",
            "authorization": {
                "manifest_path": str(self.manifest),
                "manifest_sha256": self.manifest_sha256,
                "method": method,
                "seed": seed,
                "training_fingerprint": run["training_fingerprint"],
                "checkpoint": run["checkpoint"],
                "method_environment": arm["method_environment"],
                "hyperparameters": arm["hyperparameters"],
                "registered_output_dir": str(report_path.parent),
                "registry_path": str(
                    runner_tool.registry_path_for_manifest(self.manifest)
                ),
                "registry_attempt": completion["attempt"],
            },
            "evaluator": {
                "function": "mmrotate.core.eval_rbbox_map",
                "iou_threshold": 0.5,
                "nproc": 1,
                "model_built": False,
                "inference_run": False,
                "cga_environment": dict(
                    evaluator_tool.offline_evaluator.DISABLED_CGA_ENV
                ),
            },
            "result": {
                "metric": "rotated_bbox_mAP",
                "iou_threshold": 0.5,
                "use_07_metric": True,
                "mean_ap": mean_ap,
                "per_class": per_class,
                "num_images": 1,
                "classes": list(aggregate_tool.RSAR_CLASSES),
                "detections_per_class": detections,
                "num_detections": sum(detections.values()),
                "predictions_semantic_sha256": self._hash_text(
                    f"predictions:{method}:{seed}"
                ),
                "annotations_semantic_sha256": self._hash_text("annotations"),
                "image_order_sha256": self._hash_text("image-order"),
                "first_image_id": "sample",
                "last_image_id": "sample",
            },
            "dataset": {
                "type": "DOTADataset",
                "test_mode": True,
                "ann_file": str(self.annotation_root),
                "img_prefix": str(self.image_root),
                "classes": list(aggregate_tool.RSAR_CLASSES),
                "data_manifest_path": str(self.data_manifest),
                "data_manifest_sha256": manifest_tool.sha256_file(self.data_manifest),
            },
            "provenance": {
                "command": ["synthetic-evaluator"],
                "python_executable": "/synthetic/python",
                "project_root": str(self.root),
                "tool_path": str(Path(evaluator_tool.__file__).resolve()),
                "tool_sha256": manifest_tool.sha256_file(
                    Path(evaluator_tool.__file__).resolve()
                ),
                "offline_evaluator_path": str(runtime[1]),
                "offline_evaluator_sha256": manifest_tool.sha256_file(runtime[1]),
                "manifest_tool_path": str(runtime[2]),
                "manifest_tool_sha256": manifest_tool.sha256_file(runtime[2]),
                "data_manifest_tool_path": str(runtime[3]),
                "data_manifest_tool_sha256": manifest_tool.sha256_file(runtime[3]),
                "runner_tool_path": str(runtime[4]),
                "runner_tool_sha256": manifest_tool.sha256_file(runtime[4]),
                "config_path": str(self.config),
                "config_sha256": manifest_tool.sha256_file(self.config),
                "effective_config_sha256": self._hash_text("effective-config"),
                "predictions_path": predictions["path"],
                "predictions_sha256": predictions["sha256"],
                "predictions_size_bytes": predictions["size_bytes"],
                "output_path": str(report_path),
                "pickle_trust_acknowledged": True,
                "versions": {
                    "mmcv": "synthetic",
                    "mmdet": "synthetic",
                    "mmrotate": "synthetic",
                    "numpy": "synthetic",
                },
            },
        }

    def _verify_completed(self, _manifest: Path, method: str, seed: int):
        return self.completions[(method, seed)]

    def _aggregate(self) -> dict[str, object]:
        with mock.patch.object(
            aggregate_tool.runner_tool,
            "verify_completed_run",
            side_effect=self._verify_completed,
        ) as verifier:
            result = aggregate_tool.aggregate(self.manifest)
        self.assertEqual(verifier.call_count, len(self.METHODS) * len(self.SEEDS))
        return result

    def _rewrite_report(self, identity: tuple[str, int], mutate) -> None:
        path = self.report_paths[identity]
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _assert_no_outputs(self) -> None:
        for basename in aggregate_tool.OUTPUT_BASENAMES:
            self.assertFalse((self.manifest.parent / basename).exists())

    def test_happy_path_writes_complete_descriptive_outputs_once(self) -> None:
        summary = self._aggregate()
        self.assertEqual(summary["rows"], 9)
        self.assertTrue(summary["descriptive_only"])
        self.assertFalse(summary["method_selection_performed"])
        with (self.manifest.parent / "final_test_results.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 9)
        self.assertEqual(
            [name for name in csv_rows[0] if name.startswith("ap_")],
            [f"ap_{name}" for name in aggregate_tool.RSAR_CLASSES],
        )
        jsonl_rows = [
            json.loads(line)
            for line in (self.manifest.parent / "final_test_results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(jsonl_rows), 9)
        self.assertIn("predictions", jsonl_rows[0]["provenance"])
        report = json.loads(
            (self.manifest.parent / "final_test_statistical_report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["paired_comparator"], "no_cga")
        self.assertEqual(report["settings"]["bootstrap_seed"], 20260715)
        self.assertEqual(len(report["method_descriptives"]), 3)
        self.assertEqual(len(report["paired_comparisons"]), 2)
        for comparison in report["paired_comparisons"]:
            self.assertEqual(comparison["N"], 3)
            self.assertEqual(comparison["paired_t"]["df"], 2)
            self.assertIsNotNone(comparison["paired_t_p_holm"])
            self.assertEqual(comparison["exact_sign_flip"]["mode"], "exact")
            self.assertIsNotNone(comparison["exact_sign_flip_p_holm"])
            self.assertEqual(len(comparison["pairs"]), 3)
        markdown = (
            self.manifest.parent / "final_test_statistical_report.md"
        ).read_text(encoding="utf-8")
        self.assertIn("No method selection, ranking, tuning", markdown)
        with self.assertRaisesRegex(
            aggregate_tool.FinalTestAggregationError, "refusing to replace"
        ):
            self._aggregate()

    def test_missing_fixed_report_fails_without_partial_outputs(self) -> None:
        self.report_paths[("legacy", 42)].unlink()
        with self.assertRaises(aggregate_tool.FinalTestAggregationError):
            self._aggregate()
        self._assert_no_outputs()

    def test_tampered_predictions_hash_fails_closed(self) -> None:
        self._rewrite_report(
            ("legacy", 42),
            lambda payload: payload["provenance"].__setitem__(
                "predictions_sha256", "f" * 64
            ),
        )
        with self.assertRaisesRegex(
            aggregate_tool.FinalTestAggregationError,
            "provenance predictions_sha256 mismatch",
        ):
            self._aggregate()
        self._assert_no_outputs()

    def test_duplicate_or_reordered_class_result_fails_closed(self) -> None:
        self._rewrite_report(
            ("bounded_penalty", 43),
            lambda payload: payload["result"]["per_class"][1].__setitem__(
                "class", "ship"
            ),
        )
        with self.assertRaisesRegex(
            aggregate_tool.FinalTestAggregationError,
            "duplicate, missing, or reordered classes",
        ):
            self._aggregate()
        self._assert_no_outputs()

    def test_seed_misalignment_fails_closed(self) -> None:
        self._rewrite_report(
            ("legacy", 41),
            lambda payload: payload["authorization"].__setitem__("seed", 42),
        )
        with self.assertRaisesRegex(
            aggregate_tool.FinalTestAggregationError,
            "authorization seed mismatch",
        ):
            self._aggregate()
        self._assert_no_outputs()


if __name__ == "__main__":
    unittest.main()
