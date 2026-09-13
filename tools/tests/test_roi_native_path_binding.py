"""CPU fixtures for the retained /mnt/shared -> staged /home native ROI gate."""

import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

from experiments.comparison import result_completion as completion


class NativePathBindingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.native_prefix = "/mnt/shared/zechuan"
        self.resolver = self.root / "approved_paths.py"
        self.resolver.write_text(
            "def map_path(value):\n"
            "    value = str(value)\n"
            f"    source, target = {self.native_prefix!r}, {str(self.home)!r}\n"
            "    return target + value[len(source):] if "
            "value == source or value.startswith(source + '/') else value\n")
        self.run = {
            "run_id": "DIOR/brightness/seed_42/IRG/ema",
            "dataset": "DIOR", "domain": "brightness", "seed": 42,
            "method": "IRG", "role": "ema", "scope": "full_test",
            "checkpoint_domain": "brightness",
            "checkpoint": str(self.home / "artifacts/iter_185_ema.pth"),
            "config": str(self.home / "artifacts/resolved_config.py"),
            "ann_file": str(self.home / "data/test.txt"),
            "img_prefix": str(self.home / "artifacts/brightness/test"),
            "image_ids": [str(i) for i in range(11726, 11744)],
            "visualization_image_ids": [str(i) for i in range(11726, 11742)],
            "out_dir": str(self.root / "not-exported"),
            "export_code_sha": "old-exporter",
            "allowed_gpus": [3],
            "native_prediction": {
                "eval_dir": str(self.home / "artifacts/eval"),
                "prepared_training_code_sha": "prepared-code",
                "evaluation_code_sha": "native-eval-code",
                "source_id": "source-identity",
            },
        }
        self.eval = Path(self.run["native_prediction"]["eval_dir"])
        self.eval.mkdir(parents=True)
        Path(self.run["checkpoint"]).write_bytes(b"checkpoint")
        Path(self.run["config"]).write_text("model=dict(type='OrientedRCNN')\n")
        Path(self.run["ann_file"]).parent.mkdir()
        Path(self.run["ann_file"]).write_text("\n".join(self.run["image_ids"]))
        Path(self.run["img_prefix"]).mkdir(parents=True)
        (self.eval / "predictions.pkl").write_bytes(pickle.dumps([None] * 18))
        self.execution = {
            **{key: self.run[key] for key in ("dataset", "domain", "seed", "method", "role")},
            "checkpoint": self.recorded("checkpoint"), "config": self.recorded("config"),
            "training_code_sha": "native-training-code",
            "source_id": "source-identity", "evaluation_code_sha": "native-eval-code",
        }
        self.sidecar = {
            "schema": "iraod-prediction-image-order-v1", "status": "complete",
            "origin": "inference_batch_img_metas", "predictions_file": "predictions.pkl",
            "checkpoint": self.recorded("checkpoint"), "config": self.recorded("config"),
            "training_code_sha": "native-training-code",
            "evaluation_code_sha": "native-eval-code",
            "n_images": 18, "dataset_size": 18,
            "image_ids": self.run["image_ids"],
            "records": [
                {"prediction_index": i, "image_id": image, "ori_filename": image + ".jpg"}
                for i, image in enumerate(self.run["image_ids"])
            ],
            "cfg_options": {f"data.test.{key}": self.recorded(key)
                            for key in ("ann_file", "img_prefix")},
        }
        self.write(self.eval / "execution.json", self.execution)
        self.write(self.eval / "predictions.pkl.image_ids.json", self.sidecar)
        self.status()
        sizes = patch.dict(completion.EXPECTED_TEST_IMAGES, {"DIOR": 18})
        sizes.start()
        self.addCleanup(sizes.stop)

    @staticmethod
    def write(path, value):
        Path(path).write_text(json.dumps(value))

    @staticmethod
    def identity(path):
        path = Path(path)
        return {"bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def recorded(self, field):
        return self.run[field].replace(str(self.home), self.native_prefix, 1)

    def status(self, checkpoint=None):
        self.eval.joinpath("eval_status").write_text(
            f"eval_exit=0 name={self.run['method']} domain=brightness seed={self.run['seed']} "
            f"role=ema checkpoint={checkpoint or self.recorded('checkpoint')}\n")

    def path_binding(self, core=False):
        return {
            "resolver": {"path": str(self.resolver),
                         "sha256": self.identity(self.resolver)["sha256"]},
            "artifacts": {
                "checkpoint": self.identity(self.run["checkpoint"]),
                "config": self.identity(self.run["config"]),
                ("native_report" if core else "execution_json"):
                    self.identity(self.root / "report.json" if core else self.eval / "execution.json"),
                "prediction_sidecar": self.identity(self.eval / "predictions.pkl.image_ids.json"),
            },
        }

    def bind(self):
        self.run["native_prediction"]["path_binding"] = self.path_binding()

    def test_authentic_port_path_difference_requires_verified_explicit_binding(self):
        self.assertEqual(completion.native_binding_evidence(self.run)["status"], "pending")
        self.bind()
        evidence = completion.native_binding_evidence(self.run)
        self.assertEqual(evidence["status"], "complete")
        self.assertEqual(evidence["path_binding"], self.run["native_prediction"]["path_binding"])
        self.assertEqual(self.execution["checkpoint"], self.recorded("checkpoint"))

    def test_wrong_resolver_hash_and_same_size_changed_artifact_reject(self):
        self.bind()
        self.resolver.write_text(self.resolver.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "resolver.*SHA256"):
            completion.native_binding_evidence(self.run)
        self.bind()
        Path(self.run["checkpoint"]).write_bytes(b"CHECKPOINT")
        with self.assertRaisesRegex(ValueError, "checkpoint.*identity"):
            completion.native_binding_evidence(self.run)

    def test_unknown_mapping_wrong_tuple_missing_file_and_missing_proof_never_complete(self):
        self.bind()
        self.status("/unapproved/iter_185_ema.pth")
        self.assertEqual(completion.native_binding_evidence(self.run)["status"], "pending")
        self.status()
        changed = copy.deepcopy(self.run)
        changed["seed"] = 43
        self.assertEqual(completion.native_binding_evidence(changed)["status"], "pending")
        Path(self.run["config"]).unlink()
        evidence = completion.native_binding_evidence(self.run)
        self.assertEqual(evidence["status"], "pending")
        self.assertIn(self.run["config"], evidence["missing"])
        Path(self.run["config"]).write_text("model=dict(type='OrientedRCNN')\n")
        del self.run["native_prediction"]["path_binding"]["artifacts"]["prediction_sidecar"]
        with self.assertRaisesRegex(ValueError, "artifact.*proof"):
            completion.native_binding_evidence(self.run)

    def test_sidecar_split_and_execution_identity_remain_strict_after_mapping(self):
        for field in ("config", "training_code_sha", "source_id", "evaluation_code_sha"):
            with self.subTest(field=field):
                altered = {**self.execution, field: "/unapproved/value"}
                self.write(self.eval / "execution.json", altered)
                self.bind()
                with self.assertRaises(ValueError):
                    completion.native_binding_evidence(self.run)
        self.write(self.eval / "execution.json", self.execution)
        self.sidecar["cfg_options"]["data.test.img_prefix"] = "/unapproved/brightness/test"
        self.write(self.eval / "predictions.pkl.image_ids.json", self.sidecar)
        self.bind()
        with self.assertRaisesRegex(ValueError, "split/domain"):
            completion.native_binding_evidence(self.run)

    def test_pending_admission_is_not_no_exception_and_runtime_uses_same_binding(self):
        from experiments.comparison.dior_recovery import extract_roi_pre_fc_cls as exporter

        plan = self.root / "plan.json"
        untouched = {**self.run, "run_id": "unselected"}
        self.write(plan, {"schema": completion.SCHEMA, "runs": [self.run, untouched]})
        proof = self.root / "binding.json"
        self.write(proof, self.path_binding())
        output = self.root / "versioned.json"
        self.status("/unapproved/checkpoint.pth")
        with self.assertRaisesRegex(ValueError, "pending native inputs"):
            admission = completion.bind_native_paths(plan, self.run["run_id"], proof, output)
            self.assertEqual(admission["status"], "complete")
            self.assertEqual(admission["native_prediction"]["status"], "complete")
        self.assertFalse(output.exists())
        self.status()
        completion.bind_native_paths(plan, self.run["run_id"], proof, output)
        prepared = completion.read_json(output)
        bound = prepared["runs"][0]
        self.assertEqual(prepared["runs"][1], untouched)
        with self.assertRaises(FileExistsError):
            completion.bind_native_paths(plan, self.run["run_id"], proof, output)
        with patch("sys.argv", ["extract_roi", "--plan", str(output),
                               "--run-id", self.run["run_id"], "--physical-gpu", "3"]), \
                patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "CUDA_VISIBLE_DEVICES": "3"}), \
                patch("iraod_runtime.ensure_iraod_runtime",
                      side_effect=RuntimeError("GPU boundary reached")) as runtime:
            with self.assertRaisesRegex(RuntimeError, "GPU boundary reached"):
                exporter.main()
            runtime.assert_called_once()
            runtime.reset_mock()
            with patch.object(exporter.subprocess, "check_output", return_value="old-exporter"):
                with self.assertRaisesRegex(ValueError, "Export checkout differs"):
                    exporter.main()
            runtime.assert_not_called()
        self.assertEqual(bound["native_prediction"]["path_binding"], self.path_binding())
        Path(self.run["out_dir"]).mkdir()
        with self.assertRaises(FileExistsError):
            completion.bind_native_paths(plan, self.run["run_id"], proof,
                                         self.root / "another-plan.json")

    def test_core_report_also_uses_explicit_mapping_without_new_execution_record(self):
        self.run.update(method="B", seed=43)
        self.status()
        entry = {
            **self.execution, "method": "B", "seed": 43,
            "effective_training_code_sha": "native-training-code",
            "status": "complete", "problems": [], "n_predictions": 18,
            **{key: str(self.eval / filename).replace(str(self.home), self.native_prefix, 1)
               for key, filename in (("eval_dir", ""), ("eval_status", "eval_status"),
                                     ("predictions", "predictions.pkl"),
                                     ("prediction_image_ids", "predictions.pkl.image_ids.json"))},
        }
        report_file = self.root / "report.json"
        self.write(report_file, {
            "schema": "iraod-comparison-report-v1", "code_commit": "report-code",
            "raw_results": [entry], "source_ids": {"DIOR": "source-identity"},
            "source_provenance": {"DIOR": {
                "weights_sha256": "source-identity", "record_status": "present",
                "actual_source_producer_sha": "source-code"}},
            "checkpoints": [{
                **{key: self.run[key] for key in ("dataset", "domain", "seed", "method", "role")},
                "path": self.recorded("checkpoint"), "verified": True, "selection": "final",
                "iteration": 185, "source_id": "source-identity",
                "bytes": Path(self.run["checkpoint"]).stat().st_size,
            }],
        })
        self.run["native_prediction"].update(
            format="comparison-report-v1", report=str(report_file), report_entry=0,
            report_code_sha="report-code",
            checkpoint_bytes=Path(self.run["checkpoint"]).stat().st_size)
        self.run["native_prediction"]["path_binding"] = self.path_binding(core=True)
        (self.eval / "execution.json").unlink()
        self.assertEqual(completion.native_binding_evidence(self.run)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
