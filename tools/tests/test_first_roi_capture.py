"""CPU replay of the frozen consumer loop; no neural model forward or GPU."""

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.comparison import aligned_roi
from experiments.comparison import capture_first_roi_image as capture_entry


ROOT = Path(__file__).resolve().parents[2]
COUNTS = [16, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 9]


class FirstImageCaptureTest(unittest.TestCase):
    def replay(self, mismatch=False, empty=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "diagnostic"
            directory.mkdir()
            plan = root / "plan.json"
            plan.write_text("{}")
            spec = {
                "scope_id": "cpu-capture-fixture", "plan": str(plan), "run_id": "fixture/ema",
                "image_id": "11726", "native_prediction_index": 0,
                "identity": {"role": "ema"}, "image_path": "fixture/11726.jpg",
                "image_sha256": "fixture-only", "native_class_row_counts": COUNTS,
            }
            run = {
                "out_dir": str(root / "canonical-roi"), "checkpoint": "fixture-checkpoint",
                "config": "fixture-config", "native_prediction": {"eval_dir": "fixture-native"},
                "image_ids": ["11726", "11727"],
            }
            original_run = copy.deepcopy(run)
            labels = np.repeat(np.arange(20, dtype=np.int64), COUNTS)
            dets = np.arange(32 * 6, dtype=np.float32).reshape(32, 6) / 256
            reference = [dets[labels == label].copy() for label in range(20)]
            if mismatch:
                reference[0][0, 0] += np.float32(0.125)
            actual_dets, actual_labels = (dets[:0], labels[:0]) if empty else (dets, labels)
            actual = [actual_dets[actual_labels == label] for label in range(20)]
            capture = aligned_roi.AlignedRoICapture(None, None)

            def reset():
                count = len(actual_dets)
                capture.features = [np.ones((count, 4), dtype=np.float32)]
                capture.detections = [(
                    actual_dets, actual_labels,
                    np.arange(count, dtype=np.int64) * 20 + actual_labels, 20, count)]

            capture.reset = reset
            operands = capture_entry.OperandCapture(spec, run, directory)
            completion = types.SimpleNamespace(load_run=lambda *_: run)
            original_compare = aligned_roi.validate_native_predictions
            checked_before_compare = []

            def checked_compare(arrays, native):
                self.assertTrue((directory / "actual_operands.npz").is_file())
                self.assertTrue((directory / "native_operands.npz").is_file())
                self.assertTrue((directory / "capture.json").is_file())
                checked_before_compare.append(True)
                return original_compare(arrays, native)

            source = ast.parse((ROOT / "experiments/comparison/dior_recovery/"
                                "extract_roi_pre_fc_cls.py").read_text())
            loop = next(node for node in ast.walk(source) if (
                isinstance(node, ast.For) and isinstance(node.target, ast.Tuple)
                and [item.id for item in node.target.elts] == ["image_id", "data"]))
            code = compile(ast.Module(body=[loop], type_ignores=[]),
                           "frozen860-consumer-loop", "exec")
            model = Mock(return_value=[actual])
            records = []
            loader = [
                {"img_metas": [types.SimpleNamespace(data=[[
                    {"ori_filename": f"{image}.jpg", "filename": f"fixture/{image}.jpg"}]])]}
                for image in run["image_ids"]
            ]
            expected_error = ValueError if mismatch or empty else capture_entry.FirstImageCaptured
            with patch.object(aligned_roi, "validate_native_predictions", checked_compare), \
                    capture_entry.observe_consumer(spec, run, operands, completion, aligned_roi), \
                    patch.object(np, "savez_compressed", wraps=np.savez_compressed) as save:
                selected = completion.load_run(plan, spec["run_id"])
                self.assertEqual(run, original_run)
                self.assertEqual(selected["out_dir"], str(directory / "unused_roi_output"))
                namespace = {
                    "capture": capture, "run": selected, "loader": loader, "model": model,
                    "Path": Path, "np": np, "records": records,
                    "predictions": [reference, reference],
                    "prediction_indices": {"11726": 0, "11727": 1},
                    "validate_native_predictions": aligned_roi.validate_native_predictions,
                    "out": directory / "unused_roi_output",
                }
                with self.assertRaises(expected_error):
                    exec(code, namespace)
                self.assertEqual(save.call_count, 2)
            self.assertEqual(model.call_count, 1)
            self.assertEqual(operands.calls, 1)
            self.assertEqual(len(checked_before_compare), 1)
            self.assertEqual(records, [])
            self.assertFalse(Path(run["out_dir"]).exists())
            self.assertFalse((directory / "unused_roi_output/index.json").exists())
            with np.load(directory / "actual_operands.npz", allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"boxes", "scores", "labels"})
                np.testing.assert_array_equal(saved["boxes"], actual_dets[:, :5])
                np.testing.assert_array_equal(saved["scores"], actual_dets[:, 5])
                np.testing.assert_array_equal(saved["labels"], actual_labels)
            with np.load(directory / "native_operands.npz", allow_pickle=False) as saved:
                self.assertEqual(len(saved.files), 20)
                for label, rows in enumerate(reference):
                    np.testing.assert_array_equal(saved[f"class_{label}"], rows)
            self.assertEqual(operands.comparison,
                             "MISMATCH" if mismatch or empty else "MATCH_DIAGNOSTIC_ONLY")

    def test_mismatch_saves_operands_before_real_exception_and_no_second_image(self):
        self.replay(mismatch=True)

    def test_match_also_stops_before_roi_write_index_or_second_image(self):
        self.replay()

    def test_empty_actual_output_is_saved_and_remains_strict_failure(self):
        self.replay(empty=True)

    def test_real_bootstrap_preserves_entry_and_explicit_cwd_unset_and_ready(self):
        head = capture_entry.git_head(ROOT)
        for ready in (False, True):
            with self.subTest(ready=ready), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                images = root / "images"
                images.mkdir()
                image = images / "11726.jpg"
                image.write_bytes(b"CPU identity fixture; not decoded")
                ids = [str(i) for i in range(11726, 11726 + 11738)]
                native = root / "native"
                native.mkdir()
                (native / "predictions.pkl.image_ids.json").write_text(
                    json.dumps({"image_ids": ids}))
                plan = root / "plan.json"
                identity = {
                    "dataset": "DIOR", "domain": "brightness", "seed": 42,
                    "method": "IRG", "role": "ema",
                }
                run_id = "DIOR/brightness/seed_42/IRG/ema"
                plan.write_text(json.dumps({
                    "schema": "iraod-aligned-roi-v3-full-test", "runs": [{
                        **identity, "run_id": run_id, "scope": "full_test",
                        "image_ids": ids, "visualization_image_ids": ids[:16],
                        "checkpoint": "CPU-fixture", "config": "CPU-fixture",
                        "img_prefix": str(images), "out_dir": str(root / "canonical-roi"),
                        "native_prediction": {"eval_dir": str(native)},
                        "export_code_sha": head,
                    }]}))
                resolver = root / "resolver.py"
                resolver.write_text("def map_path(value):\n    return str(value)\n")
                spec = {
                    "scope_id": "cpu-entry-fixture", "consumer_checkout": str(ROOT),
                    "consumer_sha": head, "capture_commit": head, "python": sys.executable,
                    "resolver": {"path": str(resolver), "sha256": capture_entry.sha256(resolver)},
                    "allowed_gpus": [3], "plan": str(plan), "run_id": run_id,
                    "identity": identity, "image_id": "11726", "native_prediction_index": 0,
                    "image_path": str(image), "image_bytes": image.stat().st_size,
                    "image_sha256": capture_entry.sha256(image),
                    "capture_dir": str(root / "must-not-be-created"),
                }
                spec_file = root / "spec.json"
                spec_file.write_text(json.dumps(spec))
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1"}
                env.pop("IRAOD_RUNTIME_READY", None)
                if ready:
                    env["IRAOD_RUNTIME_READY"] = "1"
                command = [
                    sys.executable, str(ROOT / "experiments/comparison/capture_first_roi_image.py"),
                    "--spec", "spec.json", "--physical-gpu", "3", "--cpu-entry-check",
                ]
                result = subprocess.run(command, cwd=root, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                info = json.loads(result.stdout)
                self.assertEqual(info["cwd"], str(ROOT.resolve()))
                self.assertEqual(info["runtime_ready"], "1")
                self.assertEqual(info["consumer_sha"], head)
                self.assertFalse(info["model_forward"])
                self.assertFalse(Path(spec["capture_dir"]).exists())
                self.assertFalse((root / "canonical-roi").exists())


if __name__ == "__main__":
    unittest.main()
