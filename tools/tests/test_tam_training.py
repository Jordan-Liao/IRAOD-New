"""CPU-only TAM CLI/loop contract tests; never call production GPU main."""

import csv
import io
from itertools import islice
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from experiments.comparison import result_completion
from experiments.comparison import train_tam as training
from experiments.comparison.tam_artifacts import load_completed_tam


class FakeTAM(torch.nn.Module):
    """Record the API boundary, not a replacement implementation of TAM losses."""

    def __init__(self, weights_path=None, fail_at=None):
        super().__init__()
        self.decoder = torch.nn.Linear(1, 1)
        self.F1 = torch.nn.Linear(1, 1)
        self.F2 = torch.nn.Linear(1, 1)
        self.fail_at = fail_at
        self.events = []
        self.optimizers = (object(), object())

    def make_optimizers(self):
        self.events.append("optimizers")
        return self.optimizers

    def alternating_step(self, content, style, optimizer_d, optimizer_f, iteration):
        assert (optimizer_d, optimizer_f) == self.optimizers
        self.events.append((iteration, content.clone(), style.clone()))
        if iteration == self.fail_at:
            raise RuntimeError("injected update failure")
        return {"decoder": torch.tensor(2.), "moments": torch.tensor(3.),
                "lr": 1e-4 / (1 + 5e-5 * iteration)}


class SamplingAndLoopTest(unittest.TestCase):
    def test_independent_deterministic_complete_permutations(self):
        size = 19
        first = list(islice(iter(training.InfinitePermutationSampler(size, 42)), size * 3))
        replay = list(islice(iter(training.InfinitePermutationSampler(size, 42)), size * 3))
        style = list(islice(iter(training.InfinitePermutationSampler(size, 43)), size * 3))
        self.assertEqual(first, replay)
        self.assertNotEqual(first, style)
        generator = torch.Generator().manual_seed(42)
        for offset in range(0, size * 3, size):
            permutation = first[offset:offset + size]
            self.assertEqual(sorted(permutation), list(range(size)))
            self.assertEqual(permutation, torch.randperm(size, generator=generator).tolist())
        self.assertNotEqual(first[:size], first[size:size * 2])

    def test_preprocess_bgr_centered_not_unit_range(self):
        for rgb in ((255, 0, 0), (255, 255, 255), (0, 0, 0)):
            with self.subTest(rgb=rgb):
                tensor = training.preprocess_image(Image.new("RGB", (19, 17), rgb))
                self.assertEqual(tensor.shape, (3, 128, 128))
                self.assertEqual(tensor.dtype, torch.float32)
                expected = torch.tensor(rgb[::-1], dtype=torch.float32)
                expected -= torch.tensor(training.BGR_MEAN)
                torch.testing.assert_close(tensor[:, 0, 0], expected)
                torch.testing.assert_close(tensor, expected[:, None, None].expand_as(tensor))
        self.assertGreater(float(tensor.abs().max()), 100)

    def test_resize_crop_flip_order(self):
        pixels = np.arange(15 * 23 * 3, dtype=np.uint8).reshape(15, 23, 3)
        image = Image.fromarray(pixels)
        with patch.object(training.random, "randint", side_effect=[37, 51]) as crop:
            with patch.object(training.random, "random", return_value=0.1):
                actual = training.preprocess_image(image)
        self.assertEqual(crop.call_args_list[0].args, (0, 672))
        self.assertEqual(crop.call_args_list[1].args, (0, 472))
        expected_image = image.resize((800, 600), Image.Resampling.BILINEAR)
        expected_image = expected_image.crop((37, 51, 165, 179))
        expected_image = expected_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        expected = np.asarray(expected_image, dtype=np.float32)[:, :, ::-1].copy()
        expected -= np.asarray(training.BGR_MEAN, dtype=np.float32)
        torch.testing.assert_close(actual, torch.from_numpy(expected.transpose(2, 0, 1)))

    def test_worker_rngs_replay(self):
        with patch.object(torch, "initial_seed", return_value=2**32 + 43):
            training.seed_worker(0)
            first = (random.random(), np.random.random(), torch.rand(1))
            training.seed_worker(0)
            second = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2])

    def test_finite_loop_exact_order_pairing_and_csv(self):
        module = FakeTAM()
        content = [torch.tensor([i]) for i in range(5)]
        style = [torch.tensor([i + 20]) for i in range(5)]
        log = io.StringIO()
        completed = training.train_loop(module, content, style, 3, "cpu", log)
        self.assertEqual(completed, 3)
        self.assertEqual(module.events[0], "optimizers")
        self.assertEqual(len(module.events), 4)
        for i, (iteration, actual_content, actual_style) in enumerate(module.events[1:]):
            self.assertEqual(iteration, i)
            torch.testing.assert_close(actual_content, content[i])
            torch.testing.assert_close(actual_style, style[i])
        rows = list(csv.DictReader(io.StringIO(log.getvalue())))
        self.assertEqual([row["iteration"] for row in rows], ["0", "1", "2"])
        self.assertEqual(float(rows[-1]["lr"]), 1e-4 / (1 + 5e-5 * 2))

    def test_log_periodically_flushed(self):
        class Log(io.StringIO):
            flushes = 0

            def flush(self):
                self.flushes += 1
                super().flush()

        log = Log()
        batches = [torch.zeros(1)] * 201
        training.train_loop(FakeTAM(), batches, batches, 201, "cpu", log)
        self.assertEqual(log.flushes, 4)  # header, iterations 100/200, terminal

    def test_loop_exhaustion_and_invalid_steps_do_not_finish(self):
        with self.assertRaises(StopIteration):
            training.train_loop(FakeTAM(), [], [], 1, "cpu", io.StringIO())
        for steps in (0, -1, 160001):
            with self.assertRaises(ValueError):
                training.train_loop(FakeTAM(), [], [], steps, "cpu", io.StringIO())


class TargetTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tam fixtures ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base.json"
        self.encoder = self.root / "external-vgg.pth"
        self.encoder.write_bytes(b"test-only reference, never loaded by FakeTAM")
        self.out = self.root / "new training"
        runs = []
        self.vals = {}
        for dataset, domains in training.DOMAINS.items():
            ids = ([str(i) for i in range(11726, 11742)] if dataset == "DIOR"
                   else [f"rsar_{i}" for i in range(32)])
            for domain in domains:
                directory = self.root / dataset / domain
                test = directory / ("test/images" if dataset == "RSAR" else "test")
                val = directory / ("val/images" if dataset == "RSAR" else "val")
                test.mkdir(parents=True)
                val.mkdir(parents=True)
                self.vals[dataset, domain] = val
                Image.new("RGB", (20, 20), (255, 0, 0)).save(val / "b.png")
                Image.new("RGB", (20, 20), (0, 0, 255)).save(val / "a.JPG")
                # No annotations exist. Non-images must not become training examples.
                (val / "unrelated.txt").write_text("not annotation syntax")
                Image.new("RGB", (20, 20), (0, 255, 0)).save(test / "test_only.png")
                for method, role in result_completion.ROLES:
                    runs.append({
                        "run_id": f"{dataset}/{domain}/{method}/{role}",
                        "dataset": dataset, "domain": domain,
                        "method": method, "role": role, "seed": 42,
                        "scope": "full_test", "image_ids": ids,
                        "visualization_image_ids": ids,
                        "checkpoint_domain": "source" if method == "A" else domain,
                        "img_prefix": str(test), "ann_file": "/must/not/read/GT",
                    })
        self.plan = {"schema": result_completion.SCHEMA, "adaptation_seed": 42, "runs": runs}
        self.write_plan()
        self.counts = patch.dict(training.TARGET_VAL_SIZE, RSAR=2, DIOR=2)
        self.counts.start()
        self.addCleanup(self.counts.stop)
        self.test_counts = patch.dict(result_completion.EXPECTED_TEST_IMAGES, RSAR=32, DIOR=16)
        self.test_counts.start()
        self.addCleanup(self.test_counts.stop)

    def write_plan(self):
        self.base.write_text(json.dumps(self.plan))

    def run_fit(self, **overrides):
        kwargs = dict(base_plan=self.base, dataset="RSAR", domain="clean", seed=43,
                      vgg_weights=self.encoder, out_dir=self.out, steps=3,
                      workers=0, device="cpu", module_factory=FakeTAM)
        kwargs.update(overrides)
        return training.run_training(**kwargs)

    def test_target_only_all_images_no_gt_filtering_for_both_layouts(self):
        for dataset, domain in (("RSAR", "chaff"), ("DIOR", "cloudy")):
            manifest = training.discover_target_images(self.base, dataset, domain, 44)
            self.assertEqual(manifest["image_ids"], ["a", "b"])
            self.assertEqual(manifest["root"], str(self.vals[dataset, domain]))
            self.assertEqual(manifest["split"], "val")
            self.assertEqual(manifest["seed"], 44)
            self.assertTrue(all(str(self.vals[dataset, domain]) in path
                                for path in manifest["image_paths"]))
            self.assertIn("not bitwise", manifest["reproducibility"]["replay"])

    def test_reject_wrong_identity_count_and_duplicate_stems(self):
        for dataset, domain, seed in (("other", "clean", 42), ("RSAR", "cloudy", 42),
                                      ("DIOR", "chaff", 42), ("DIOR", "clean", 45)):
            with self.assertRaises(ValueError):
                training.discover_target_images(self.base, dataset, domain, seed)
        val = self.vals["RSAR", "clean"]
        (val / "a.JPG").unlink()
        with self.assertRaisesRegex(ValueError, "requires 2"):
            training.discover_target_images(self.base, "RSAR", "clean", 42)
        Image.new("RGB", (20, 20)).save(val / "b.jpg")
        with self.assertRaisesRegex(ValueError, "unique stems"):
            training.discover_target_images(self.base, "RSAR", "clean", 42)

    def test_reject_nonaccepted_base_plan(self):
        self.plan["adaptation_seed"] = 43
        for run in self.plan["runs"]:
            if run["method"] != "A":
                run["seed"] = 43
        self.write_plan()
        with self.assertRaisesRegex(ValueError, "adaptation seed42"):
            training.discover_target_images(self.base, "RSAR", "clean", 42)
        self.plan["schema"] = "not-full-test"
        self.write_plan()
        with self.assertRaises(ValueError):
            training.discover_target_images(self.base, "RSAR", "clean", 42)

    def test_loader_batch_and_independent_rngs(self):
        manifest = training.discover_target_images(self.base, "RSAR", "clean", 43)
        content, style = training.make_loaders(manifest, workers=0)
        self.assertEqual((content.sampler.seed, style.sampler.seed), (43, 44))
        self.assertIsNot(content.generator, style.generator)
        self.assertEqual((content.generator.initial_seed(), style.generator.initial_seed()),
                         (43, 44))
        for loader in (content, style):
            self.assertTrue(loader.drop_last)
            self.assertEqual(next(iter(loader)).shape, (8, 3, 128, 128))

    def test_smoke_outputs_atomic_and_not_formal(self):
        terminal = self.run_fit()
        self.assertEqual(terminal["status"], "smoke_not_formal")
        self.assertEqual(terminal["training_steps"], 3)
        self.assertTrue(terminal["loop_completed"])
        self.assertEqual({path.name for path in self.out.iterdir()},
                         {"training_config.json", "image_manifest.json", "train_log.csv",
                          "tam.pth", "terminal.json"})
        config = json.loads((self.out / "training_config.json").read_text())
        self.assertEqual(config["result_scope"], "NON_RESULT")
        self.assertEqual(config["official_sfyolo_commit"], training.OFFICIAL_SFYOLO_SHA)
        self.assertEqual(config["batch_size"], 8)
        self.assertEqual(config["encoder_bytes"], self.encoder.stat().st_size)
        self.assertEqual(len(config["training_code_sha"]), 40)
        payload = torch.load(self.out / "tam.pth", weights_only=True)
        self.assertEqual(set(payload["components"]), {"decoder", "F1", "F2"})
        self.assertEqual(payload["encoder_weights"], str(self.encoder))
        self.assertEqual(payload["identity"], {"dataset": "RSAR", "domain": "clean", "seed": 43})
        with self.assertRaisesRegex(ValueError, "completed TAM"):
            load_completed_tam(self.out / "tam.pth", payload["identity"], "cpu")
        with (self.out / "train_log.csv").open() as log:
            self.assertEqual(len(list(csv.DictReader(log))), 3)
        with self.assertRaises(FileExistsError):
            self.run_fit()

    def test_failure_terminal_no_final_checkpoint(self):
        fake = FakeTAM(fail_at=1)
        with self.assertRaisesRegex(RuntimeError, "injected update"):
            self.run_fit(module_factory=lambda **_: fake)
        terminal = json.loads((self.out / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["loop_completed"])
        self.assertNotIn("training_steps", terminal)
        self.assertFalse((self.out / "tam.pth").exists())
        self.assertFalse((self.out / "tam.pth.partial").exists())
        with (self.out / "train_log.csv").open() as log:
            self.assertEqual(len(list(csv.DictReader(log))), 1)

    def test_initialization_failure_is_terminal(self):
        with self.assertRaisesRegex(ValueError, "explicit existing"):
            self.run_fit(vgg_weights=self.root / "missing.pth")
        self.assertEqual(json.loads((self.out / "terminal.json").read_text())["status"], "failed")
        self.assertFalse((self.out / "tam.pth").exists())

    def test_checkpoint_write_failure_cleans_partial(self):
        def fail_save(_payload, path):
            path.write_bytes(b"incomplete")
            raise OSError("injected disk failure")

        with patch.object(training.torch, "save", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "injected disk"):
                self.run_fit()
        terminal = json.loads((self.out / "terminal.json").read_text())
        self.assertEqual(terminal["status"], "failed")
        self.assertTrue(terminal["loop_completed"])
        self.assertFalse((self.out / "tam.pth").exists())
        self.assertFalse((self.out / "tam.pth.partial").exists())


class CLIAdmissionTest(unittest.TestCase):
    def args(self, *extra):
        return ["--base-plan", "plan.json", "--dataset", "RSAR", "--domain", "clean",
                "--seed", "42", "--vgg-weights", "external.pth", "--out-dir", "new",
                "--gpu", "4", *extra]

    def test_cli_defaults_and_smoke_limits(self):
        args = training.parse_args(self.args())
        self.assertIsNone(args.smoke_steps)
        self.assertEqual(args.workers, 16)
        self.assertEqual(training.parse_args(self.args("--smoke-steps", "2")).smoke_steps, 2)
        for extra in (("--smoke-steps", "0"), ("--smoke-steps", "160000"),
                      ("--workers", "-1"), ("--domain", "cloudy"), ("--gpu", "3"),
                      ("--epochs", "1"), ("--lr", "0.1"), ("--image-root", "/arbitrary")):
            with self.subTest(extra=extra), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    training.parse_args(self.args(*extra))

    def test_gpu_lock_and_unsupported_gpu_fail_before_cuda(self):
        with patch.object(torch.cuda, "is_available") as available:
            with patch.object(torch.cuda, "_lazy_init") as initialize:
                with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "0"}):
                    with self.assertRaisesRegex(RuntimeError, "GPU lock"):
                        training.configure_gpu(4)
                with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}):
                    for gpu in (-1, 0, 3, 8):
                        with self.assertRaises(ValueError):
                            training.configure_gpu(gpu)
                available.assert_not_called()
                initialize.assert_not_called()

    def test_requires_startup_isolation_and_sets_gpu_order(self):
        with patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1", "PYTHONNOUSERSITE": "1"}):
            with patch.object(training.sys, "flags") as flags:
                flags.no_user_site = 0
                with self.assertRaisesRegex(RuntimeError, "Start the interpreter"):
                    training.configure_gpu(4)
                flags.no_user_site = 1
                training.configure_gpu(7)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")
                self.assertEqual(os.environ["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")


if __name__ == "__main__":
    unittest.main()
