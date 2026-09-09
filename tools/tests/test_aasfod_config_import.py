"""Real MMCV stage serialization/import on CPU, not detector training evidence."""

from collections.abc import Mapping
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from mmcv import Config
from torchvision.transforms import ColorJitter

from experiments.comparison import host_binding as host, train_aasfod
from experiments.comparison.aasfod_protocol import TSD_CHOICE, budget
from experiments.comparison.b_regression import config_diff


ROOT = Path(__file__).resolve().parents[2]
NATIVE_FROMFILE = Config.fromfile


def operators(value):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from operators(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from operators(item)
    elif isinstance(value, ColorJitter):
        yield value


class AASFODConfigImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aasfod config import ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cwd, path = Path.cwd(), sys.path[:]
        self.addCleanup(os.chdir, cwd)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), path))

    def fixture(self, dataset):
        root = self.root / dataset
        root.mkdir()
        count = 8467 if dataset == "RSAR" else 5863
        work = root / "work"
        cell = dict(
            dataset=dataset, domain="clean", seed=42,
            training_code=str(ROOT), work_dir=str(work),
            config=str(ROOT / "configs/unbiased_teacher/sfod/extensions"
                       / f"aasfod_{dataset.lower()}.py"),
            source_checkpoint=str(root / "source.pth"),
            target_val=str(root / "val"), unlabeled_epoch_size=count,
            tsd_split=str(root / "tsd.json"), aasfod_budget=budget(count),
            student_checkpoint=str(work / f"iter_{budget(count)['total_updates'] + 1}.pth"),
            checkpoint=str(work / f"iter_{budget(count)['total_updates'] + 1}_ema.pth"))
        names = [f"image{i:05}.png" for i in range(count)]
        split = dict(
            identity={key: cell[key] for key in
                      ("dataset", "domain", "seed", "source_checkpoint", "target_val")},
            choice=TSD_CHOICE, status="complete",
            scores={name: float(i) for i, name in enumerate(names)},
            similar=names[-(count // 5):],
            dissimilar=names[:count - count // 5])
        Path(cell["tsd_split"]).write_text(json.dumps(split))
        return cell

    def entry_scope(self, stack, cell, target=False):
        stack.enter_context(patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}))
        stack.enter_context(patch.object(host, "is_target_host", return_value=target))
        stack.enter_context(patch.object(train_aasfod, "load_cell", return_value=(
            {"python": sys.executable, "training_code": str(ROOT)}, cell)))
        # Only custom detector registration is excluded; Config, recursive
        # Python imports, and torchvision operators are the installed originals.
        def load(filename):
            return NATIVE_FROMFILE(filename, import_custom_modules=False)
        stack.enter_context(patch.object(Config, "fromfile", side_effect=load))

    def test_plain_mmcv_dump_reproduces_exact_nameerror(self):
        cell = self.fixture("RSAR")
        cfg = NATIVE_FROMFILE(cell["config"], import_custom_modules=False)
        self.assertTrue(list(operators(cfg._cfg_dict)))
        generated = self.root / "alignment.py"
        cfg.dump(str(generated))
        with self.assertRaisesRegex(NameError, "name 'ColorJitter' is not defined"):
            NATIVE_FROMFILE(str(generated), import_custom_modules=False)

    def test_actual_stage_entry_roundtrips_both_configs_and_host_modes(self):
        for dataset in ("RSAR", "DIOR"):
            cell = self.fixture(dataset)
            split_bytes = Path(cell["tsd_split"]).read_bytes()
            for target in (False, True):
                with self.subTest(dataset=dataset, target=target), ExitStack() as stack:
                    cell = {**cell, "work_dir": str(self.root / f"{dataset}-{target}")}
                    work = Path(cell["work_dir"])
                    cell.update(student_checkpoint=str(work / "final.pth"),
                                checkpoint=str(work / "final_ema.pth"))
                    self.entry_scope(stack, cell, target)
                    resolved, imported = [], []

                    def load(filename):
                        cfg = NATIVE_FROMFILE(filename, import_custom_modules=False)
                        resolved.append(cfg)
                        return cfg

                    stack.enter_context(patch.object(Config, "fromfile", side_effect=load))

                    def native_import(command, cwd, env, check):
                        self.assertEqual(cwd, ROOT)
                        self.assertEqual(env["PYTHONPATH"], str(ROOT))
                        self.assertTrue(check)
                        index = command.index(str(ROOT / "train.py"))
                        self.assertEqual("native" in command, target)
                        with host.native_config_paths():
                            cfg = NATIVE_FROMFILE(command[index + 1], import_custom_modules=False)
                        self.assertEqual(config_diff(resolved[-1], cfg), [])
                        self.assertNotIn("ColorJitter", cfg)
                        before = list(operators(resolved[-1]._cfg_dict))
                        after = list(operators(cfg._cfg_dict))
                        self.assertGreater(len(after), 0)
                        self.assertEqual(len(before), len(after))
                        for original, reloaded in zip(before, after):
                            self.assertIs(type(original), ColorJitter)
                            self.assertIs(type(reloaded), type(original))
                            for name in ("brightness", "contrast", "saturation", "hue"):
                                self.assertEqual(getattr(reloaded, name), getattr(original, name))
                        imported.append(cfg)
                        directory = Path(cfg.work_dir)
                        directory.mkdir()
                        for suffix in ("", "_ema"):
                            (directory / f"iter_{cfg.runner.max_iters}{suffix}.pth").write_bytes(
                                b"CPU dispatch fixture, not trained weights")

                    stack.enter_context(patch.object(train_aasfod.subprocess, "run",
                                                    side_effect=native_import))
                    train_aasfod.run("/fixture/queue", dataset, "clean", 42)
                    alignment, fns = imported
                    self.assertEqual([cfg.runner.max_iters for cfg in imported],
                                     [cell["aasfod_budget"][stage + "_updates"]
                                      for stage in ("alignment", "fns")])
                    self.assertEqual([cfg.data.samples_per_gpu for cfg in imported], [16, 32])
                    self.assertEqual(alignment.load_from, cell["source_checkpoint"])
                    self.assertEqual(alignment.model.ema_ckpt, alignment.load_from)
                    self.assertEqual(fns.load_from, str(work / "alignment" /
                                     f"iter_{alignment.runner.max_iters}.pth"))
                    self.assertEqual(fns.model.ema_ckpt, fns.load_from)
                    self.assertEqual(alignment.lr_config.warmup_iters, 100)
                    self.assertNotIn("warmup", fns.lr_config)
                    self.assertEqual(Path(cell["tsd_split"]).read_bytes(), split_bytes)

    def test_native_failure_preserves_split_and_partial_stage(self):
        cell = self.fixture("RSAR")
        split = Path(cell["tsd_split"])
        split_bytes = split.read_bytes()
        work = Path(cell["work_dir"])
        with ExitStack() as stack:
            self.entry_scope(stack, cell)
            def fail(command, **kwargs):
                (work / "alignment").mkdir()
                (work / "alignment" / "failure.log").write_text("retained native failure")
                raise subprocess.CalledProcessError(1, command)
            run = stack.enter_context(patch.object(train_aasfod.subprocess, "run", side_effect=fail))
            with self.assertRaises(subprocess.CalledProcessError):
                train_aasfod.run("/fixture/queue", "RSAR", "clean", 42)
            with self.assertRaises(FileExistsError):
                train_aasfod.run("/fixture/queue", "RSAR", "clean", 42)
            self.assertEqual(run.call_count, 1)
        self.assertEqual(split.read_bytes(), split_bytes)
        self.assertEqual((work / "alignment" / "failure.log").read_text(), "retained native failure")
        self.assertEqual(json.loads((work / "stages.json").read_text())["status"],
                         "invoked_not_completion_evidence")
        self.assertFalse(Path(cell["checkpoint"]).exists())
        self.assertFalse(Path(cell["student_checkpoint"]).exists())
        self.assertFalse((work / "fns.py").exists())

    def test_partial_tsd_never_reaches_config_or_native_launch(self):
        cell = self.fixture("RSAR")
        split = Path(cell["tsd_split"])
        payload = json.loads(split.read_text())
        payload["status"] = "partial"
        split.write_text(json.dumps(payload))
        with ExitStack() as stack:
            self.entry_scope(stack, cell)
            run = stack.enter_context(patch.object(train_aasfod.subprocess, "run"))
            with self.assertRaisesRegex(ValueError, "Smoke/partial TSD"):
                train_aasfod.run("/fixture/queue", "RSAR", "clean", 42)
            Config.fromfile.assert_not_called()
            run.assert_not_called()
        self.assertFalse(Path(cell["work_dir"]).exists())


if __name__ == "__main__":
    unittest.main()
