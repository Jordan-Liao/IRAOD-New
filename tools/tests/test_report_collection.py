"""Filesystem metadata discovery tests; no prediction/feature binaries are loaded."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.comparison.collect_report_manifest import choose_evaluation, collect_manifest
from experiments.comparison.result_completion import write_json


class ReportCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = SimpleNamespace(
            root=lambda ds, domain, seed: str(self.root / ds / domain / str(seed)),
            method_dir=lambda ds, domain, seed, method: str(
                self.root / ds / domain / str(seed) / method),
            ema_path=lambda ds, domain, seed, method: str(
                self.root / ds / domain / str(seed) / method / "iter_266_ema.pth"),
            train_verified=lambda *args: False,
            eval_full_dir=lambda ds, domain, seed, method: str(
                self.root / ds / domain / str(seed) / method / f"eval_full_{domain}"),
        )

    def evaluation(self, suffix, value):
        directory = Path(self.paths.eval_full_dir("RSAR", "chaff", "42", "B") + suffix)
        directory.mkdir(parents=True)
        write_json(directory / "eval_1.json", {"config": "/fixture.py", "metric": {"mAP": value}})
        return directory

    def test_native_directory_is_selected_without_deserializing_predictions(self):
        self.evaluation("", .1)
        native = self.evaluation("_ids_v1", .2)
        (native / "predictions.pkl").write_bytes(b"not a pickle; discovery must not load it")
        directory, metric, _, issue = choose_evaluation(self.paths, {}, "RSAR", "chaff", 42, "B")
        self.assertEqual(directory, native)
        self.assertEqual(json.loads(metric.read_text())["metric"]["mAP"], .2)
        self.assertIsNone(issue)

    def test_ambiguous_json_is_not_chosen_by_mtime(self):
        native = self.evaluation("_ids_v1", .2)
        write_json(native / "eval_2.json", {"config": "/fixture.py", "metric": {"mAP": .3}})
        _, metric, _, issue = choose_evaluation(self.paths, {}, "RSAR", "chaff", 42, "B")
        self.assertIsNone(metric)
        self.assertEqual(issue, "ambiguous_eval_json")

    def test_owner_source_clean_location_and_scoped_manifest(self):
        source = self.root / "source-eval/eval_source.json"
        source.parent.mkdir()
        write_json(source, {"config": "/source.py", "metric": {"mAP": .5}})
        owner = {"A_source_clean": {"RSAR": {"json": str(source)}},
                 "result_completion": {"scope": "legacy_subset"}}
        _, metric, _, issue = choose_evaluation(self.paths, owner, "RSAR", "clean", 42, "A")
        self.assertEqual(metric, source)
        self.assertIsNone(issue)
        roi_plan = self.root / "plan.json"
        write_json(roi_plan, {"schema": "iraod-aligned-roi-v3-full-test", "runs": []})
        manifest = collect_manifest(self.paths, owner, roi_plan, self.root / "collected",
                                    inspect_roi_run_ids=["RSAR/clean/A/source"])
        self.assertEqual(manifest["inspect_roi_run_ids"], ["RSAR/clean/A/source"])
        self.assertEqual(manifest["collection"]["metric_files_found"], 1)
        inventory = json.loads((self.root / "collected/collection_inventory.json").read_text())
        self.assertEqual(len(inventory), 192)
        self.assertTrue(all(r["status"] == "not_inspected" for r in inventory if r["seed"] != 42))
        self.assertEqual(manifest["embeddings"], [])


if __name__ == "__main__":
    unittest.main()
