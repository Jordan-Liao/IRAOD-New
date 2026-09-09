"""CPU-only regression for preserved native eval-status path identities."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.comparison import host_binding as host
from experiments.comparison.report_inputs import CLASSES, EXPECTED_IMAGES, inspect_cell
from experiments.comparison.result_completion import write_json
from tools.prediction_export import save_predictions_with_ids


class ReportEvalPathAliasTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ds = "RSAR"
        self.eval = self.root / "eval_full_clean"
        self.eval.mkdir()
        self.checkpoint = {
            "dataset": self.ds, "domain": "clean", "method": "B", "seed": 42, "role": "ema",
            "path": "/owner/rsar/clean/seed_42/methods/B/work/iter_266_ema.pth",
            "selection": "final", "iteration": 266, "verified": True,
            "source_id": "fixture-source", "gpu_provenance": {"topology": "single"},
        }
        self.cell = {
            "dataset": self.ds, "domain": "clean", "method": "B", "seed": 42, "role": "ema",
            "eval_dir": str(self.eval), "eval_json": str(self.eval / "eval_fixture.json"),
            "prediction_image_ids": str(self.eval / "predictions.pkl.image_ids.json"),
        }
        write_json(self.cell["eval_json"], {
            "config": "/owner/source_inference.py", "metric": {"mAP": .2446555644273758, "AP50": .245}})
        (self.eval / "eval_status").write_text(
            "eval_exit=0 2026-09-06T00:00:00+08:00 name=B domain=clean seed=42 "
            f"ema={self.checkpoint['path']}\n")
        (self.eval / "class_ap.txt").write_text(
            "| class | gts | dets | recall | ap |\n+---+---+---+---+---+\n"
            + "".join(f"| {c} | 1 | 1 | 0.500 | 0.123 |\n" for c in CLASSES[self.ds])
            + "| mAP | | | | 0.245 |\n{'mAP': 0.2446555644273758, 'AP50': 0.245}\n")
        count = EXPECTED_IMAGES[self.ds]
        per_class = [np.zeros((0, 6), dtype=np.float32) for _ in CLASSES[self.ds]]
        per_class[0] = np.array([[2, 3, 4, 5, .1, .15]], dtype=np.float32)
        save_predictions_with_ids(
            self.eval / "predictions.pkl", [per_class] * count,
            [{"image_id": f"test_{i}", "ori_filename": f"test_{i}.png",
              "filename": f"/fixture/test_{i}.png"} for i in range(count)],
            {"dataset_size": count, "config": "/owner/source_inference.py",
             "checkpoint": self.checkpoint["path"], "training_code_sha": "fixture-training"})
        (self.eval / "pred_count.txt").write_text(f"{count} expect={count}\n")
        self.manifest = {
            "schema": "iraod-comparison-report-v1", "roles": ["ema"],
            "cells": [self.cell], "checkpoints": [self.checkpoint],
            "source_ids": {"RSAR": "fixture-source", "DIOR": "fixture-dior-source"},
        }

    def test_preserved_status_paths_match_only_the_selected_host_aliases(self):
        for method, role, field in (("A", "source", "ema"), ("B", "ema", "ema"),
                                    ("B", "student", "student"), ("B", "ema", "checkpoint")):
            with self.subTest(method=method, role=role, field=field):
                filename = "iter_266.pth" if role == "student" else "iter_266_ema.pth"
                original = "/mnt/shared/zechuan/iraod_artifacts/fixture/" + filename
                mapped = original.replace("/mnt/shared/zechuan", "/home/zechuan")
                self.cell.update(method=method, role=role)
                self.checkpoint.update(method=method, role=role, path=mapped,
                                       selection="source" if method == "A" else "final")
                order_path = Path(self.cell["prediction_image_ids"])
                order = json.loads(order_path.read_text())
                order["checkpoint"] = mapped
                write_json(order_path, order)
                status = self.eval / "eval_status"
                status.write_text(f"eval_exit=0 name={method} domain=clean seed=42 "
                                  f"role={role} {field}={original}\n")
                before = status.read_bytes()
                with patch.object(host, "is_target_host", return_value=True):
                    result, _, _ = inspect_cell(self.cell, self.checkpoint, "fixture-source")
                self.assertEqual(result["status"], "complete", result["problems"])
                self.assertEqual(result["n_predictions"], 8538)
                self.assertEqual(status.read_bytes(), before)
                with patch.object(host, "is_target_host", return_value=False):
                    result, _, _ = inspect_cell(self.cell, self.checkpoint, "fixture-source")
                self.assertIn("last_eval_status_failed_or_identity_mismatch", result["problems"])
                status.write_text(before.decode().replace(original, mapped))
                with patch.object(host, "is_target_host", return_value=False):
                    result, _, _ = inspect_cell(self.cell, self.checkpoint, "fixture-source")
                self.assertEqual(result["status"], "complete", result["problems"])

    def test_original_status_metric_and_sidecar_paths_form_one_source_identity(self):
        original_checkpoint = "/mnt/shared/zechuan/iraod_artifacts/source/train/epoch_100.pth"
        mapped_checkpoint = original_checkpoint.replace("/mnt/shared/zechuan", "/home/zechuan")
        original_config = "/mnt/SSD2_8TB/zechuan/IRAOD-New-rc331d213/configs/source.py"
        mapped_config = original_config.replace("/mnt/SSD2_8TB/zechuan", "/home/zechuan")
        self.cell.update(method="A", role="source", config=mapped_config)
        self.checkpoint.update(method="A", role="source", selection="source", path=mapped_checkpoint)
        payload_path = Path(self.cell["eval_json"])
        payload = json.loads(payload_path.read_text())
        payload["config"] = original_config
        write_json(payload_path, payload)
        order_path = Path(self.cell["prediction_image_ids"])
        order = json.loads(order_path.read_text())
        order.update(checkpoint=original_checkpoint, config=mapped_config)
        write_json(order_path, order)
        status_path = self.eval / "eval_status"
        status_path.write_text("eval_exit=0 name=A domain=clean seed=42 "
                               f"ema={original_checkpoint}\n")
        original_bytes = {p: p.read_bytes() for p in (payload_path, order_path, status_path)}
        with patch.object(host, "is_target_host", return_value=True):
            result, _, _ = inspect_cell(self.cell, self.checkpoint, "fixture-source")
        self.assertEqual(result["status"], "complete", result["problems"])
        self.assertEqual(result["n_predictions"], 8538)
        self.assertEqual({p: p.read_bytes() for p in original_bytes}, original_bytes)

        for origin, field, value, problem in (
                (payload_path, "config", original_config + ".wrong", "eval_config_mismatch"),
                (order_path, "config", mapped_config + ".wrong", "prediction_sidecar_checkpoint_or_config_mismatch"),
                (order_path, "checkpoint", original_checkpoint + ".wrong", "prediction_sidecar_checkpoint_or_config_mismatch")):
            with self.subTest(origin=origin.name, field=field):
                for path, content in original_bytes.items():
                    path.write_bytes(content)
                changed = json.loads(origin.read_text())
                changed[field] = value
                write_json(origin, changed)
                with patch.object(host, "is_target_host", return_value=True):
                    result, _, _ = inspect_cell(self.cell, self.checkpoint, "fixture-source")
                self.assertEqual(result["status"], "incomplete")
                self.assertIn(problem, result["problems"])

    def test_status_path_alias_does_not_relax_native_identity(self):
        original = "/mnt/shared/zechuan/iraod_artifacts/fixture/iter_266_ema.pth"
        mapped = original.replace("/mnt/shared/zechuan", "/home/zechuan")
        self.checkpoint["path"] = mapped
        order_path = Path(self.cell["prediction_image_ids"])
        order = json.loads(order_path.read_text())
        order["checkpoint"] = mapped
        write_json(order_path, order)
        good = {"eval_exit": "0", "name": "B", "domain": "clean", "seed": "42",
                "role": "ema", "checkpoint": original}
        for field, value in (("eval_exit", "1"), ("name", "C"), ("domain", "chaff"),
                             ("seed", "43"), ("role", "student"),
                             ("checkpoint", original + ".wrong"), ("checkpoint", "")):
            with self.subTest(field=field, value=value):
                status = self.eval / "eval_status"
                status.write_text(" ".join(f"{k}={v}" for k, v in
                                           {**good, field: value}.items()) + "\n")
                before = status.read_bytes()
                with patch.object(host, "is_target_host", return_value=True):
                    result, _, _ = inspect_cell(self.cell, self.checkpoint, "fixture-source")
                self.assertEqual(result["status"], "incomplete")
                self.assertIn("last_eval_status_failed_or_identity_mismatch", result["problems"])
                self.assertEqual(status.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
