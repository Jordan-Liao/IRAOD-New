"""CPU-only publication contract; synthetic arrays, no detector or model load."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from experiments.comparison import joint_roi, roi_joint_native
from experiments.comparison.report_inputs import CLASSES
from experiments.comparison.result_completion import EXPECTED_TEST_IMAGES, SCHEMA, load_export
from experiments.comparison.roi_joint_native import CAPTURE_ENTRY, capture_command
from experiments.comparison.roi_rollout_bind import bind_plan
from tools.prediction_export import capture_image_records, save_predictions_with_ids


class ArrayFixture:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class FCFixture:
    def register_forward_pre_hook(self, callback):
        self.callback = callback
        return types.SimpleNamespace(remove=lambda: setattr(self, "callback", None))


def synthetic_loop(model, loader, *args, **kwargs):
    predictions, images = [], []
    for data in loader:
        result = model.forward(return_loss=False, rescale=True, **data)
        predictions.extend(result)
        images.extend(capture_image_records(data, len(result)))
    return predictions, images


class LoaderFixture(list):
    batch_size = 1

    def __init__(self, ids):
        super().__init__({
            "img_metas": [types.SimpleNamespace(data=[[{
                "ori_filename": image_id + ".jpg", "filename": "/fixture/" + image_id + ".jpg",
            }]])],
        } for image_id in ids)
        self.dataset = self
        self.CLASSES = CLASSES["DIOR"]


class JointPublicationTest(unittest.TestCase):
    def test_one_forward_and_native_gated_publication(self):
        ids = [str(i) for i in range(11726, 11742)]
        code_sha = subprocess.check_output(
            ["git", "-C", str(joint_roi.ROOT), "rev-parse", "HEAD"], text=True).strip()
        detections = np.asarray([[10, 20, 3, 4, .1, .9], [30, 40, 5, 6, .2, .8]], dtype=np.float32)
        labels = np.zeros(2, dtype=np.int64)
        flat = np.asarray([0, 20], dtype=np.int64)
        features = np.arange(6, dtype=np.float32).reshape(2, 3)
        predictions = [
            [detections.copy(), *[np.empty((0, 6), dtype=np.float32) for _ in range(19)]]
            for _ in ids]
        command = capture_command(["/native/python", "/native/code/test.py", "cfg", "ckpt"],
                                  "/new/case", "DIOR", 1)
        self.assertEqual(Path(command[1]).name, "test.py")
        self.assertEqual(command[1], str(CAPTURE_ENTRY))
        self.assertEqual(command[-3:], ["/native/code/test.py", "cfg", "ckpt"])

        cases = ("success", "native_failure", "AP_failure", "baseline_value_difference", "different_score",
                 "different_order", "nonfinite_feature", "short_loop", "forward_failure")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(EXPECTED_TEST_IMAGES, {"DIOR": len(ids)}):
            bootstrap_root = Path(temporary) / "bootstrap"
            writer = bootstrap_root / "writer"
            writer.mkdir(parents=True)
            (bootstrap_root / "spec.json").write_text(json.dumps({
                "profile": {"writer_code": str(writer), "python": sys.executable},
            }))

            def bootstrap_stop():
                self.assertEqual(Path.cwd(), writer)
                self.assertEqual(sys.argv, [
                    str(Path(roi_joint_native.__file__).resolve()),
                    "--spec", str(bootstrap_root / "spec.json"),
                    "--out", str(bootstrap_root / "candidate"),
                    "--physical-gpu", "1", "--baseline-root", str(bootstrap_root / "baseline")])
                raise RuntimeError("CPU bootstrap seam reached before any execution")

            previous_cwd = Path.cwd()
            try:
                os.chdir(bootstrap_root)
                with patch.dict(os.environ), patch.object(sys, "path", sys.path.copy()), \
                        patch.object(sys, "argv", [
                            roi_joint_native.__file__, "--spec", "spec.json", "--out", "candidate",
                            "--physical-gpu", "1", "--baseline-root", "baseline"]), \
                        patch("iraod_runtime.ensure_iraod_runtime", bootstrap_stop), \
                        self.assertRaisesRegex(RuntimeError, "CPU bootstrap seam"):
                    roi_joint_native.main()
                self.assertFalse((bootstrap_root / "candidate").exists())
            finally:
                os.chdir(previous_cwd)
            for case in cases:
                with self.subTest(case=case):
                    root = Path(temporary) / case
                    root.mkdir()
                    resolver = root / "resolver.py"
                    resolver.write_text("def map_path(path): return path\n"
                                        "def map_data(data): return data\n")
                    checkpoint, config = root / "iter_185_ema.pth", root / "config.py"
                    checkpoint.write_bytes(b"CPU_FIXTURE_NOT_A_MODEL")
                    config.write_text("# CPU fixture only\n")
                    original = {
                        "dataset": "DIOR", "domain": "contrast", "seed": 42,
                        "method": "IRG", "role": "ema", "run_id": "DIOR/contrast/seed_42/IRG/ema",
                        "scope": "full_test", "image_ids": ids, "visualization_image_ids": ids,
                        "checkpoint": str(checkpoint), "config": str(config),
                        "ann_file": "/fixture/test.txt", "img_prefix": "/fixture",
                        "out_dir": "/historical/unchanged", "allowed_gpus": [1],
                        "native_prediction": {"eval_dir": "/historical/native/unchanged"},
                    }
                    original_plan = root / "original_plan.json"
                    original_plan.write_text(json.dumps({"schema": SCHEMA, "runs": [original]}))
                    profile = {"host": 221, "resolver": str(resolver), "allowed_gpus": [1]}
                    original_input = {"run": original, "source_plan": str(original_plan),
                                      "execution": {"source_id": "CPU_FIXTURE_SOURCE"}}
                    cell = {key: original[key] for key in (
                        "dataset", "domain", "seed", "method", "role", "checkpoint",
                        "config", "ann_file", "img_prefix")}
                    cell.update(training_code_sha="CPU_FIXTURE_TRAIN", source_id="CPU_FIXTURE_SOURCE")

                    def native_fixture(destination, outputs, *, joint=False):
                        native_dir = destination / "native"
                        native_dir.mkdir(parents=True)
                        selected = {**cell, "eval_dir": str(native_dir)}
                        if joint:
                            selected["joint_exporter"] = {
                                "code_commit": code_sha, "mode": "single-forward-native-roi"}
                        (native_dir / "execution.json").write_text(json.dumps({
                            **selected, "evaluation_code_sha": code_sha,
                            "command": ["CPU_FIXTURE_NOT_EXECUTED"],
                        }))
                        (native_dir / "eval_status").write_text(
                            f"eval_exit=0 name=IRG domain=contrast seed=42 role=ema checkpoint={checkpoint}\n")
                        (native_dir / "class_ap.txt").write_text(
                            "| class | ap |\n" + "".join(f"| {name} | 0.25 |\n" for name in CLASSES["DIOR"]))
                        save_predictions_with_ids(
                            native_dir / "predictions.pkl", outputs,
                            [capture_image_records(data, 1)[0] for data in LoaderFixture(ids)],
                            {"dataset_size": len(ids), "config": str(config), "checkpoint": str(checkpoint),
                             "training_code_sha": cell["training_code_sha"],
                             "cfg_options": {"data.test.ann_file": cell["ann_file"],
                                             "data.test.img_prefix": cell["img_prefix"]}})
                        (destination / "native_pass.json").write_text(json.dumps({
                            "status": "complete", "kind": "port", "profile": profile,
                            "cell": selected, "original_input": original_input, "physical_gpu": 1,
                            "canonical_key": "DIOR/contrast/42/IRG/ema", "evaluation_code_sha": code_sha,
                        }))

                    baseline = root / "baseline"
                    native_fixture(baseline, predictions)
                    bind_plan(baseline)
                    candidate = root / "candidate"
                    candidate.mkdir()
                    actual_detections = detections.copy()
                    if case == "baseline_value_difference":
                        actual_detections[0, 0] += np.float32(.01)
                    actual_predictions = copy.deepcopy(predictions)
                    for image in actual_predictions:
                        image[0] = actual_detections.copy()
                    fc = FCFixture()
                    nms = types.SimpleNamespace(multiclass_nms_rotated=lambda *a, **kw: (
                        ArrayFixture(actual_detections), ArrayFixture(labels), ArrayFixture(flat)))
                    fake = types.SimpleNamespace(module=types.SimpleNamespace(
                        CLASSES=CLASSES["DIOR"],
                        roi_head=types.SimpleNamespace(
                            bbox_head=types.SimpleNamespace(fc_cls=fc), test_cfg={"fixture": True})))
                    calls = []

                    def forward(*args, **kwargs):
                        self.assertFalse((candidate / "roi/index.json").exists())
                        calls.append(1)
                        if case == "forward_failure" and len(calls) == 2:
                            raise RuntimeError("synthetic forward failure")
                        fc.callback(None, [ArrayFixture(features)])
                        nms.multiclass_nms_rotated(
                            ArrayFixture(np.zeros((2, 5))), ArrayFixture(np.zeros((2, 21))))
                        return [copy.deepcopy(actual_predictions[0])]

                    fake.forward = forward
                    loader = LoaderFixture(ids)

                    def loop(*args, **kwargs):
                        if case == "short_loop":
                            return synthetic_loop(args[0], args[1][:-1], **kwargs)
                        return synthetic_loop(*args, **kwargs)

                    loop_error = case in ("forward_failure", "short_loop")
                    with contextlib.ExitStack() as stack:
                        if loop_error:
                            stack.enter_context(self.assertRaises((ValueError, RuntimeError)))
                        outputs = joint_roi.observe_native_loop(
                            loop, fake, loader, pending=candidate / "roi.unverified",
                            dataset_name="DIOR", nms_module=nms, provenance={"cpu_fixture": True},
                            checkpoint_meta={"iter": 185}, loop_args=(), loop_kwargs={"return_image_ids": True})
                    self.assertIsNone(fc.callback)
                    self.assertFalse((candidate / "roi/index.json").exists())
                    self.assertFalse((candidate / "roi.unverified/index.json").exists())
                    if loop_error:
                        self.assertFalse((candidate / "roi.unverified/capture.json").exists())
                        self.assertEqual(len(calls), 2 if case == "forward_failure" else len(ids) - 1)
                        continue
                    self.assertEqual(len(calls), len(ids))
                    for actual, expected in zip(outputs[0], actual_predictions):
                        joint_roi.exact_baseline_predictions(actual, expected)
                    native_fixture(candidate, outputs[0], joint=True)
                    if case == "native_failure":
                        status = candidate / "native/eval_status"
                        status.write_text(status.read_text().replace("eval_exit=0", "eval_exit=1"))
                    elif case == "AP_failure":
                        (candidate / "native/class_ap.txt").write_text("| class | ap |\n")
                    elif case in ("different_score", "different_order", "nonfinite_feature"):
                        path = candidate / "roi.unverified" / (ids[0] + ".npz")
                        with np.load(path) as saved:
                            arrays = {key: saved[key] for key in saved.files}
                        if case == "different_score":
                            arrays["scores"][0] += np.float32(.01)
                        elif case == "different_order":
                            for key in arrays:
                                if key != "detection_indices":
                                    arrays[key] = arrays[key][::-1]
                        else:
                            arrays["features"][0, 0] = np.nan
                        np.savez_compressed(path, **arrays)
                    if case == "success":
                        with contextlib.redirect_stdout(io.StringIO()):
                            index = joint_roi.publish(candidate, baseline)
                        self.assertEqual(index["n_images"], len(ids))
                        self.assertEqual(index["n_post_nms_detections"], 2 * len(ids))
                        self.assertEqual(index["native_prediction"]["status"], "complete")
                        self.assertFalse((candidate / "roi.unverified").exists())
                        exported_run = json.loads((candidate / "roi_plan.json").read_text())["runs"][0]
                        _, exported = load_export(exported_run)
                        self.assertEqual(sum(len(a["features"]) for _, a in exported), 2 * len(ids))
                        with np.load(candidate / "roi" / (ids[0] + ".npz")) as arrays:
                            np.testing.assert_array_equal(arrays["features"], features)
                    else:
                        with self.assertRaises(ValueError):
                            joint_roi.publish(candidate, baseline)
                        self.assertFalse((candidate / "roi/index.json").exists())
                        self.assertFalse((candidate / "roi.unverified/index.json").exists())
                    self.assertEqual(json.loads(original_plan.read_text())["runs"][0], original)


if __name__ == "__main__":
    unittest.main()
