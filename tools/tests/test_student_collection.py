"""Student collection uses existing exact bindings, never evaluation fallbacks."""

import tempfile
from pathlib import Path
import unittest

from experiments.comparison.collect_student_report import collect, render
from experiments.comparison.result_completion import DOMAINS, read_json, write_json


class StudentCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.bindings, rows = {}, []
        for ds, domains in DOMAINS.items():
            iteration = 266 if ds == "RSAR" else 185
            for domain in domains:
                for method in "BCDEF":
                    for seed in (42, 43, 44):
                        work = self.root / ds / domain / method / str(seed)
                        work.mkdir(parents=True)
                        ckpt = work / f"iter_{iteration}.pth"
                        ckpt.write_bytes(b"student")
                        base = dict(dataset=ds, domain=domain, method=method, seed=seed)
                        self.bindings[f"{ds}/{domain}/{method}/{seed}"] = {
                            **base, "role": "student", "source_id": ds,
                            "checkpoint": str(ckpt), "checkpoint_bytes": 7,
                            "eval_dir": str(work / "student_eval"), "config": "/config.py",
                            "training_code_sha": "actual-training",
                        }
                        rows.append({**base, "role": "ema", "source_id": ds,
                                     "status": "complete",
                                     "checkpoint": str(work / f"iter_{iteration}_ema.pth")})
        self.core = self.root / "core.json"
        write_json(self.core, dict(status="declared_scopes_complete", raw_results=rows,
                                  checkpoints=[], source_ids={"RSAR": "RSAR", "DIOR": "DIOR"},
                                  source_provenance={}))
        self.save_runtime()

    def save_runtime(self):
        write_json(self.queue / "runtime.json", {"student_cells": self.bindings})

    def test_missing_and_ambiguous_artifacts_are_not_success(self):
        binding = next(iter(self.bindings.values()))
        directory = Path(binding["eval_dir"])
        directory.mkdir()
        for name in ("eval_1.json", "eval_2.json"):
            write_json(directory / name, {"metric": {"mAP": .5}})
        manifest = read_json(collect(self.queue, self.core, self.root / "out"))
        self.assertEqual(manifest["roles"], ["student"])
        self.assertEqual(read_json(manifest["cells"]), [])
        rows = read_json(self.root / "out/collection_inventory.json")
        self.assertEqual(len(rows), 180)
        self.assertIn("ambiguous_eval_json", rows[0]["missing"])
        self.assertTrue(all(r["status"] == "incomplete" for r in rows))

    def test_wrong_final_checkpoint_is_rejected_before_output(self):
        next(iter(self.bindings.values()))["checkpoint"] = "/wrong/iter_266.pth"
        self.save_runtime()
        with self.assertRaisesRegex(ValueError, "differs from accepted"):
            collect(self.queue, self.core, self.root / "out")
        self.assertFalse((self.root / "out").exists())

    def test_exact_metric_is_bound_and_runtime_is_preserved(self):
        binding = next(iter(self.bindings.values()))
        directory = Path(binding["eval_dir"])
        directory.mkdir()
        write_json(directory / "eval_1.json", {"metric": {"mAP": .5}})
        manifest = read_json(collect(self.queue, self.core, self.root / "out"))
        self.assertEqual(len(read_json(manifest["cells"])), 1)
        self.assertEqual(read_json(manifest["cells"])[0]["eval_dir"], str(directory))
        self.assertEqual((self.root / "out/student_runtime.json").read_bytes(),
                         (self.queue / "runtime.json").read_bytes())

    def test_student_render_rejects_incomplete_collection(self):
        write_json(self.root / "report.json", {
            "roles": ["student"], "quantitative_complete_cells": 191,
            "raw_results": [],
        })
        with self.assertRaisesRegex(ValueError, "all180 actual Student"):
            render(self.root, "/not-invoked")


if __name__ == "__main__":
    unittest.main()
