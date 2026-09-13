"""One lightweight CPU check of fresh BF provenance and finite spec generation."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from experiments.comparison import roi_rollout_bind as binder
from experiments.comparison import roi_rollout_specs as generator
from experiments.comparison.result_completion import native_binding_evidence


class RolloutTest(unittest.TestCase):
    def test_new_bf_execution_binding_and_finite_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = {
                "host": 183, "roi_code": "/accepted/860", "python": "/native/python",
                "resolver": "/frozen/host.py", "resolver_sha256": "retained",
                "allowed_gpus": [8, 9],
            }
            cases = []
            for role in ("ema", "student"):
                case = root / role
                native = case / "native"
                native.mkdir(parents=True)
                checkpoint = case / f"iter_185{'_ema' if role == 'ema' else ''}.pth"
                checkpoint.write_bytes(b"fixture-not-model")
                config = case / "config.py"
                config.write_text("fixture config")
                ids = ["11726", "11727"]
                original = {
                    "run_id": f"DIOR/clean/B/{role}", "dataset": "DIOR", "domain": "clean",
                    "seed": 43, "method": "B", "role": role,
                    "checkpoint": str(checkpoint), "config": str(config),
                    "ann_file": "/staged/test.txt", "img_prefix": "/staged/clean",
                    "image_ids": ids, "out_dir": "/old/untouched",
                }
                untouched = {"run_id": "unselected", "value": ["preserve"]}
                plan = case / "old_plan.json"
                plan.write_text(json.dumps({"runs": [original, untouched]}))
                record = {
                    "original_plan": {"path": str(plan), "run_index": 0},
                    "original_report": {"path": "/source/report.json",
                                        "raw_results_index": 130, "code_commit": "historical"},
                    "recorded_checkpoint_proof": {"fields": {
                        "verified": True, "accepted_ema_checkpoint": "/source/iter_185_ema.pth"}},
                    "recorded_source_proof": {"fields": {"record_status": "present"}},
                }
                cell = {key: original[key] for key in (
                    "dataset", "domain", "seed", "method", "role", "checkpoint",
                    "config", "ann_file", "img_prefix")}
                cell.update(eval_dir=str(native), training_code_sha="actual-train", source_id="source-id")
                (native / "execution.json").write_text(
                    json.dumps({**cell, "evaluation_code_sha": "actual-eval"}))
                (native / "eval_status").write_text(
                    f"eval_exit=0 name=B domain=clean seed=43 role={role} checkpoint={checkpoint}\n")
                (native / "predictions.pkl").write_bytes(b"fixture-presence")
                (native / "predictions.pkl.image_ids.json").write_text(json.dumps({
                    "schema": "iraod-prediction-image-order-v1", "origin": "inference_batch_img_metas",
                    "status": "complete", "predictions_file": "predictions.pkl",
                    "checkpoint": str(checkpoint), "config": str(config),
                    "training_code_sha": "actual-train", "evaluation_code_sha": "actual-eval",
                    "n_images": 2, "dataset_size": 2, "image_ids": ids,
                    "records": [{"prediction_index": i, "image_id": image,
                                 "ori_filename": image + ".jpg"} for i, image in enumerate(ids)],
                    "cfg_options": {"data.test.ann_file": "/staged/test.txt",
                                    "data.test.img_prefix": "/staged/clean"},
                }))
                (case / "native_pass.json").write_text(json.dumps({
                    "status": "complete", "kind": "bf", "profile": profile, "cell": cell,
                    "original_input": record, "physical_gpu": 8,
                    "canonical_key": f"DIOR/clean/43/B/{role}", "evaluation_code_sha": "actual-eval",
                }))
                with patch.object(binder, "load_host", return_value=types.SimpleNamespace(
                        map_path=str, map_data=copy.deepcopy)), \
                        patch.object(sys, "argv", ["bind", str(case)]), \
                        contextlib.redirect_stdout(io.StringIO()):
                    binder.main()
                output = json.loads((case / "roi_plan.json").read_text())
                run = output["runs"][0]
                self.assertEqual(output["runs"][1], untouched)
                self.assertEqual(json.loads(plan.read_text())["runs"][0], original)
                self.assertIsNone(run["native_prediction"]["format"])
                self.assertEqual(run["native_prediction"]["record_kind"],
                                 "fresh-successful-destination-native-execution")
                self.assertEqual(run["native_prediction"]["historical_input_provenance"]
                                 ["original_report"], record["original_report"])
                self.assertEqual(native_binding_evidence(run)["status"], "complete")
                self.assertFalse((case / "native_report").exists())
                self.assertFalse(Path(run["out_dir"]).exists())
                cases.append({"method": "B", "seed": 43,
                              "canonical_key": f"DIOR/clean/43/B/{role}"})
            profile_file = root / "profile.json"
            profile_file.write_text(json.dumps(profile))
            bf_file = root / "bf.json"
            bf_file.write_text(json.dumps({"records": cases}))
            ports = root / "ports.json"
            ports.write_text(json.dumps({"runs": [
                {"dataset": "DIOR", "domain": domain, "seed": 42, "method": "IRG",
                 "role": "ema", "run_id": f"DIOR/{domain}/seed_42/IRG/ema"}
                for domain in ("brightness", "cloudy", "contrast")]}))
            output = root / "specs"
            with patch.object(sys, "argv", [
                    "specs", "--profile", str(profile_file), "--port-plan", str(ports),
                    "--bf-input", str(bf_file), "--out", str(output),
                    "--case-root", str(root / "cases")]), \
                    contextlib.redirect_stdout(io.StringIO()):
                generator.main()
            self.assertEqual(len(list(output.glob("*.json"))), 4)
            self.assertFalse(any("brightness" in path.name for path in output.iterdir()))


if __name__ == "__main__":
    unittest.main()
