"""CPU-only checks for labeled-TRAIN appendix adapters; no real SARCLIP loads."""

import csv
import importlib
import json
import random
import subprocess
import shutil
import sys
import tempfile
import types
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image
from torch import nn

from experiments.comparison.labels import CLASSES
from tools import oracle_runtime_paths as host_binding
from tools import train_sarclip_lora_rsar as trainer


def preprocess(image):
    assert image.mode == "RGB"
    return torch.tensor(np.asarray(image).copy(), dtype=torch.float32)[..., 0].flatten() / 255


class TinyClip(nn.Module):
    def __init__(self, with_targets=True):
        super().__init__()
        self.visual = (nn.Sequential(OrderedDict(proj=nn.Linear(4, 4)))
                       if with_targets else nn.Identity())
        self.text = nn.Linear(4, 20, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, image):
        return {"image_features": self.visual(image)}


class FakeSarclip:
    def create_model_with_args(self, *args, **kwargs):
        self.model_args = args, kwargs
        self.model = TinyClip()
        self.original = {name: value.detach().clone()
                         for name, value in self.model.named_parameters()}
        return self.model

    def get_tokenizer(self, *args, **kwargs):
        return object()

    def build_zero_shot_classifier(self, model, **kwargs):
        self.classifier_args = kwargs
        self.classifier_grad_enabled = torch.is_grad_enabled()
        self.classifier_model_frozen = not any(p.requires_grad for p in model.parameters())
        return model.text.weight[:len(kwargs["classnames"])].T

    def get_model_preprocess_cfg(self, model):
        return {"size": 2}

    def image_transform(self, *args, **kwargs):
        return preprocess


class OracleLoraTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = self.root / "patch.png"
        # Grayscale inputs must still be decoded as RGB, as in the original recipe.
        Image.fromarray(np.array([[20, 80], [140, 220]], dtype=np.uint8)).save(self.patch)
        self.base = self.root / "base.safetensors"
        self.base.touch()
        self.metadata = self.root / "metadata.csv"

    def argv(self, dataset="RSAR", output=None):
        return [
            "--dataset", dataset, "--metadata", str(self.metadata),
            "--sarclip-pretrained", str(self.base),
            "--output", str(output or self.root / "output"), "--num-workers", "0",
        ]

    def row(self, dataset="RSAR", class_id=0, **changes):
        row = {
            "dataset": dataset, "split": "train", "class_id": class_id,
            "class_name": CLASSES[dataset][class_id], "patch_path": str(self.patch),
            "crop_mode": "aabb", "crop_expand": "0.4",
        }
        row.update(changes)
        return row

    def write_rows(self, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with self.metadata.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_import_does_not_reexec_runtime(self):
        with mock.patch("iraod_runtime.ensure_iraod_runtime") as ensure:
            importlib.reload(trainer)
        ensure.assert_not_called()

    def test_selected_host_maps_copied_patch_rows_without_rewriting_csv(self):
        original_root = "/mnt/shared/zechuan/iraod_artifacts"
        self.write_rows([self.row("DIOR", patch_path=original_root + "/patch.png")])
        before = self.metadata.read_bytes()
        with mock.patch.object(host_binding.socket, "gethostname",
                               return_value=host_binding.TARGET_HOST), \
                mock.patch.object(host_binding, "PREFIXES", ((original_root, str(self.root)),)):
            rows = trainer.load_metadata(self.metadata, "aabb", "DIOR")
        self.assertEqual(rows[0]["patch_path"], str(self.patch))
        self.assertEqual(rows[0]["class_id"], 0)
        self.assertEqual(rows[0]["class_name"], CLASSES["DIOR"][0])
        self.assertEqual(self.metadata.read_bytes(), before)
        with mock.patch.object(host_binding.socket, "gethostname", return_value="7T83-8xA100-67"):
            with self.assertRaisesRegex(FileNotFoundError, "patch file"):
                trainer.load_metadata(self.metadata, "aabb", "DIOR")

    def test_reported_real_dior_crop_path_legacy_failure_and_standalone_fix(self):
        prefix = "/mnt/shared/zechuan/iraod_artifacts"
        relative = ("oracle_dior_patches_36c2053/DIOR/aabb/golffield/"
                    "brightness_train_00001_0_e0.4.png")
        original_path = prefix + "/" + relative
        mapped = self.root / relative
        mapped.parent.mkdir(parents=True)
        shutil.copyfile(self.patch, mapped)
        self.write_rows([self.row("DIOR", CLASSES["DIOR"].index("golffield"),
                                  patch_path=original_path)])
        before = self.metadata.read_bytes()
        # Replay the actual deployed adb6ba4 loader, which has no prefix binding.
        source = subprocess.check_output(
            ["git", "show", "adb6ba4:tools/train_sarclip_lora_rsar.py"],
            cwd=trainer.REPO_ROOT, text=True)
        legacy = types.ModuleType("legacy_adb6ba4_oracle")
        legacy.__file__ = trainer.__file__
        exec(compile(source, "adb6ba4:tools/train_sarclip_lora_rsar.py", "exec"), legacy.__dict__)
        with self.assertRaisesRegex(FileNotFoundError, "brightness_train_00001_0_e0.4.png"):
            legacy.load_metadata(self.metadata, "aabb", "DIOR")
        with mock.patch.object(host_binding.socket, "gethostname",
                               return_value=host_binding.TARGET_HOST), \
                mock.patch.object(host_binding, "PREFIXES", ((prefix, str(self.root)),)):
            rows = trainer.load_metadata(self.metadata, "aabb", "DIOR")
        self.assertEqual(rows[0]["patch_path"], str(mapped))
        self.assertEqual(Path(rows[0]["patch_path"]).relative_to(self.root).as_posix(), relative)
        self.assertEqual(mapped.read_bytes(), self.patch.read_bytes())
        self.assertEqual(self.metadata.read_bytes(), before)

    def test_standalone_path_module_needs_no_detector_host_binding(self):
        result = subprocess.run([
            sys.executable, "-S", "-c",
            "from tools import oracle_runtime_paths; import sys; "
            "assert 'experiments.comparison.host_binding' not in sys.modules; "
            "assert not {'torch','mmcv','mmdet','mmrotate'} & sys.modules.keys()"],
            cwd=trainer.REPO_ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_target_cli_paths_runtime_and_code_cache_preserve_recipe(self):
        with mock.patch.object(host_binding.socket, "gethostname",
                               return_value=host_binding.TARGET_HOST):
            args = trainer.parse_args([
                "--dataset", "DIOR",
                "--metadata", "/mnt/shared/zechuan/iraod_artifacts/oracle_dior_patches_36c2053/metadata.csv",
                "--sarclip-pretrained", "/mnt/shared/zechuan/iraod_weights/sarclip/base.safetensors",
                "--output", "/mnt/shared/zechuan/iraod_artifacts/oracle_dior_adapter",
            ])
            self.assertEqual(args.metadata,
                             "/home/zechuan/iraod_artifacts/oracle_dior_patches_36c2053/metadata.csv")
            self.assertEqual(args.sarclip_pretrained, "/home/zechuan/iraod_weights/sarclip/base.safetensors")
            self.assertEqual(args.sarclip_dir, host_binding.map_path(trainer.REPO_ROOT))
            self.assertEqual(args.sarclip_cache_dir, args.sarclip_dir)
            self.assertEqual((args.epochs, args.batch_size, args.lr, args.weight_decay,
                              args.lora_r, args.lora_alpha, args.lora_dropout),
                             (10, 64, 1e-4, 1e-4, 8, 16.0, 0.0))
            with mock.patch.dict(trainer.os.environ, {}, clear=False), \
                    mock.patch.object(trainer, "ensure_iraod_runtime") as ensure:
                trainer.os.environ.pop("IRAOD_CONDA_PREFIX", None)
                trainer.configure_runtime()
                self.assertEqual(trainer.os.environ["IRAOD_CONDA_PREFIX"],
                                 "/home/zechuan/miniforge3/envs/iraod")
                ensure.assert_called_once()

    def test_optional_legacy_cap_is_seeded_but_formal_default_keeps_all_rows(self):
        rows = [{"class_id": c, "id": i} for c in range(2) for i in range(6)]
        self.assertIs(trainer.cap_metadata(rows, None, 42), rows)
        capped = trainer.cap_metadata(rows, 4, 42)
        self.assertEqual(capped, trainer.cap_metadata(rows, 4, 42))
        self.assertEqual(len(capped), 4)
        self.assertEqual([r["class_id"] for r in capped], [0, 0, 1, 1])

    def test_nonfinite_loss_fails_before_optimizer_step(self):
        model = TinyClip()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        with mock.patch.object(model, "forward",
                               return_value={"image_features": torch.full((2, 4), float("inf"))}), \
                mock.patch.object(optimizer, "step") as step, \
                self.assertRaisesRegex(FloatingPointError, "Nonfinite"):
            trainer.train_one_epoch(model, torch.eye(4),
                                    [(torch.ones(2, 4), torch.zeros(2, dtype=torch.long))],
                                    optimizer, "cpu")
        step.assert_not_called()

    def test_canonical_defaults_and_dataset_prompts(self):
        for dataset, size, prompt in (
            ("RSAR", 6, "A SAR image of a {}"),
            ("DIOR", 20, "an aerial image of a {}"),
        ):
            with self.subTest(dataset=dataset):
                args = trainer.parse_args(self.argv(dataset))
                self.assertEqual(len(CLASSES[dataset]), size)
                self.assertEqual((args.epochs, args.batch_size, args.lr, args.weight_decay),
                                 (10, 64, 1e-4, 1e-4))
                self.assertEqual((args.lora_r, args.lora_alpha, args.lora_dropout), (8, 16, 0))
                self.assertEqual((args.seed, args.precision, args.sarclip_model),
                                 (42, "fp32", "ViT-B-32"))
                self.assertEqual((args.crop_mode, args.templates), ("aabb", prompt))
                self.assertIsNone(args.max_patches)
        self.assertEqual(trainer.parse_args(self.argv()[2:]).dataset, "RSAR")
        args = trainer.parse_args(self.argv("DIOR") + ["--templates", "one {}; two {}"])
        self.assertEqual(trainer.split_templates(args.templates), ["one {}", "two {}"])
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            trainer.parse_args(self.argv() + ["--epochs", "0"])

    def test_legacy_rsar_and_explicit_dior_class_order(self):
        for dataset in CLASSES:
            rows = [self.row(dataset, i) for i in range(len(CLASSES[dataset]))]
            if dataset == "RSAR":
                for row in rows:
                    del row["dataset"]
            self.write_rows(rows)
            loaded = trainer.load_metadata(self.metadata, "aabb", dataset)
            self.assertEqual([r["class_name"] for r in loaded], list(CLASSES[dataset]))
            self.assertEqual([r["class_id"] for r in loaded], list(range(len(rows))))

    def test_reject_nontrain_mapping_and_contamination_even_if_crop_filtered(self):
        bad_rows = [
            self.row(split="test"), self.row(split="val"), self.row(split=""),
            self.row(class_name="airplane"),
            dict(self.row(), class_id=1),
            *[dict(self.row(), class_id=value) for value in (-1, 6, "bad", "")],
            self.row(dataset="DIOR"),
            self.row(split="test", crop_mode="rotated"),
            self.row(class_name="unknown", crop_mode="rotated"),
        ]
        for row in bad_rows:
            with self.subTest(row=row):
                self.write_rows([self.row(), row])
                with self.assertRaises(ValueError):
                    trainer.load_metadata(self.metadata, "aabb")
        for identity in ("", "RSAR"):
            row = self.row("DIOR")
            row["dataset"] = identity
            self.write_rows([row])
            with self.assertRaises(ValueError):
                trainer.load_metadata(self.metadata, "aabb", "DIOR")

    def test_missing_metadata_patch_and_empty_filter_fail(self):
        with self.assertRaises(FileNotFoundError):
            trainer.load_metadata(self.metadata, "aabb")
        for path in ("", str(self.root / "missing.png")):
            self.write_rows([self.row(patch_path=path)])
            with self.assertRaises(FileNotFoundError):
                trainer.load_metadata(self.metadata, "aabb")
        self.write_rows([self.row(crop_mode="rotated")])
        with self.assertRaises(RuntimeError):
            trainer.load_metadata(self.metadata, "aabb")

    def test_seeded_inverse_class_sampler_and_tail_batch(self):
        rows = [self.row(class_id=i) for i in (0, 0, 0, 1, 2)]
        sampler = trainer.build_balanced_sampler(rows, seed=123)
        self.assertTrue(sampler.replacement)
        self.assertEqual(sampler.num_samples, 5)
        torch.testing.assert_close(sampler.weights, torch.tensor(
            [1 / 3, 1 / 3, 1 / 3, 1, 1], dtype=torch.double))
        first = list(sampler)
        self.assertEqual(first, list(trainer.build_balanced_sampler(rows, seed=123)))
        args = trainer.parse_args(self.argv() + ["--batch-size", "2", "--seed", "123"])
        loader = trainer.build_loader(rows, preprocess, args)
        self.assertEqual(loader.generator.initial_seed(), 123)
        self.assertIs(loader.worker_init_fn, trainer.seed_worker)
        self.assertFalse(loader.drop_last)
        self.assertEqual([len(labels) for _, labels in loader], [2, 2, 1])
        trainer.seed_everything(123)
        first = (random.random(), np.random.rand(), torch.rand(2))
        trainer.seed_everything(123)
        self.assertEqual(first[:2], (random.random(), np.random.rand()))
        torch.testing.assert_close(first[2], torch.rand(2))
        with mock.patch.object(torch, "initial_seed", return_value=123):
            trainer.seed_worker(0)
            values = random.random(), np.random.rand()
            trainer.seed_worker(0)
            self.assertEqual(values, (random.random(), np.random.rand()))

    def test_no_silent_fallback_or_logit_scale_only_lora(self):
        args = trainer.parse_args(self.argv())
        with mock.patch.object(trainer, "mark_visual_proj_trainable") as fallback:
            with self.assertRaisesRegex(RuntimeError, "No nn.Linear LoRA targets"):
                trainer.configure_trainable_params(TinyClip(with_targets=False), args)
            with mock.patch.object(trainer, "inject_lora", side_effect=ValueError("injection failed")):
                with self.assertRaisesRegex(ValueError, "injection failed"):
                    trainer.configure_trainable_params(TinyClip(), args)
            with mock.patch.object(trainer, "inject_lora", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "factor pair"):
                    trainer.configure_trainable_params(TinyClip(), args)
            fallback.assert_not_called()

    def test_explicit_visual_projection_is_not_lora(self):
        args = trainer.parse_args(self.argv() + ["--train-visual-proj-only"])
        with mock.patch.object(trainer, "inject_lora") as inject:
            kind, names = trainer.configure_trainable_params(TinyClip(), args)
        inject.assert_not_called()
        self.assertEqual(kind, "visual_proj")
        self.assertIn("visual.proj.weight", names)
        self.assertIn("logit_scale", names)
        self.assertFalse(any("lora_" in name or name.startswith("text.") for name in names))

    def test_classifier_frozen_and_repeated_backward_only_updates_adapters(self):
        for dataset in CLASSES:
            with self.subTest(dataset=dataset):
                args = trainer.parse_args(self.argv(dataset) + ["--batch-size", "2"])
                fake = FakeSarclip()
                with mock.patch.object(trainer, "import_sarclip", return_value=fake):
                    model, classifier, transform = trainer.build_model(args, torch.device("cpu"))
                self.assertEqual(classifier.shape, (4, len(CLASSES[dataset])))
                self.assertEqual(fake.classifier_args["classnames"], list(CLASSES[dataset]))
                self.assertEqual(fake.classifier_args["templates"], [args.templates])
                self.assertFalse(fake.classifier_grad_enabled)
                self.assertTrue(fake.classifier_model_frozen)
                self.assertFalse(classifier.requires_grad)
                self.assertIsNone(classifier.grad_fn)
                kind, names = trainer.configure_trainable_params(model, args)
                self.assertEqual(kind, "lora")
                self.assertEqual(set(names), {
                    "logit_scale", "visual.proj.lora_down.weight", "visual.proj.lora_up.weight",
                })
                self.assertEqual((model.visual.proj.r, model.visual.proj.alpha), (8, 16))
                self.assertIsInstance(model.visual.proj.dropout, nn.Identity)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, weight_decay=args.weight_decay,
                )
                rows = [self.row(dataset, i) for i in range(3)]
                loader = trainer.build_loader(rows, transform, args)
                # Two batches in each epoch: an attached text graph would fail here.
                for _ in range(2):
                    metrics = trainer.train_one_epoch(model, classifier, loader, optimizer,
                                                     torch.device("cpu"))
                    self.assertTrue(np.isfinite(metrics["loss"]))
                torch.testing.assert_close(model.text.weight, fake.original["text.weight"])
                torch.testing.assert_close(model.visual.proj.base.weight,
                                           fake.original["visual.proj.weight"])
                torch.testing.assert_close(model.visual.proj.base.bias,
                                           fake.original["visual.proj.bias"])
                self.assertIsNone(model.text.weight.grad)
                self.assertIsNone(model.visual.proj.base.weight.grad)
                self.assertIsNotNone(model.logit_scale.grad)
                self.assertGreater(model.visual.proj.lora_up.weight.count_nonzero().item(), 0)

    def test_main_final_epoch_provenance_and_atomic_save(self):
        for dataset in CLASSES:
            with self.subTest(dataset=dataset):
                output = self.root / dataset
                output.mkdir()  # An empty precreated output directory is valid.
                rows = [self.row(dataset, i % len(CLASSES[dataset])) for i in range(65)]
                rows.append(self.row(dataset, crop_mode="rotated", crop_expand="0.2"))
                self.write_rows(rows)
                fake = FakeSarclip()
                real_replace = trainer.os.replace
                replacements = []

                def replace(source, destination):
                    self.assertTrue(source.is_file())
                    self.assertFalse(destination.exists())
                    checkpoint = torch.load(source, weights_only=True)
                    self.assertEqual(checkpoint["epochs_completed"], 10)
                    replacements.append((source.name, destination.name))
                    real_replace(source, destination)

                with mock.patch.object(trainer, "import_sarclip", return_value=fake), \
                     mock.patch.object(torch.cuda, "is_available", return_value=False), \
                     mock.patch.object(trainer.os, "replace", side_effect=replace):
                    trainer.main(self.argv(dataset, output))
                filename = f"lora_{dataset.lower()}.pth"
                self.assertEqual(replacements, [(filename + ".partial", filename)])
                self.assertFalse((output / (filename + ".partial")).exists())
                config = json.loads((output / "config.json").read_text())
                checkpoint = torch.load(output / filename, weights_only=True)
                self.assertEqual(config, {k: v for k, v in checkpoint.items() if k != "state_dict"})
                expected = {
                    "format": trainer.ADAPTER_FORMAT, "adapter_type": "lora",
                    "dataset": dataset, "classes": list(CLASSES[dataset]), "seed": 42,
                    "train_only": True, "split": "train", "status": "complete",
                    "epochs": 10, "epochs_requested": 10, "epochs_completed": 10,
                    "selection": "final_epoch", "selected_epoch": 10,
                    "batch_size": 64, "optimizer": "AdamW", "lr": 1e-4,
                    "weight_decay": 1e-4, "precision": "fp32", "force_rgb": True,
                    "metadata": str(self.metadata.resolve()), "metadata_row_count": 65,
                    "crop_modes": {"aabb": 65}, "crop_expansions": {"0.4": 65},
                    "sarclip_pretrained": str(self.base.resolve()),
                    "templates": [trainer.DEFAULT_TEMPLATES[dataset]],
                    "training_git_sha": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=trainer.REPO_ROOT, text=True).strip(),
                    "lora": {"r": 8, "alpha": 16, "dropout": 0,
                             "target_prefixes": ["visual"]},
                    "sampler": {"type": "inverse_class_weighted", "replacement": True,
                                "num_samples": 65, "seed": 42},
                }
                for key, value in expected.items():
                    self.assertEqual(config[key], value, key)
                self.assertEqual(sum(config["class_counts"].values()), 65)
                self.assertEqual(set(config["class_counts"]), set(CLASSES[dataset]))
                state = checkpoint["state_dict"]
                self.assertEqual(set(state), set(config["trainable_names"]))
                self.assertEqual(sum(value.numel() for value in state.values()),
                                 config["trainable_numel"])
                with (output / "train_log.csv").open() as stream:
                    logs = list(csv.DictReader(stream))
                self.assertEqual([int(row["epoch"]) for row in logs], list(range(1, 11)))
                self.assertTrue(all(row["dataset"] == dataset and row["seed"] == "42"
                                    for row in logs))
                with mock.patch.object(trainer, "build_model") as build:
                    with self.assertRaises(FileExistsError):
                        trainer.main(self.argv(dataset, output))
                    build.assert_not_called()

    def test_smoke_cli_rejects_projection_and_nonpositive_steps(self):
        for extra in (["--smoke-steps", "0"], ["--smoke-steps", "-1"],
                      ["--smoke-steps", "1", "--train-visual-proj-only"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                trainer.parse_args(self.argv() + extra)

    def test_smoke_is_bounded_real_lora_non_result_for_both_datasets(self):
        for dataset in CLASSES:
            with self.subTest(dataset=dataset):
                output = self.root / dataset
                self.write_rows([self.row(dataset, i % len(CLASSES[dataset]))
                                 for i in range(65)])
                fake = FakeSarclip()
                with mock.patch.object(trainer, "import_sarclip", return_value=fake), \
                     mock.patch.object(torch.cuda, "is_available", return_value=False), \
                     mock.patch.object(trainer, "save_final_adapter") as save, \
                     mock.patch.object(trainer.PatchDataset, "__getitem__",
                                       autospec=True, wraps=None,
                                       side_effect=trainer.PatchDataset.__getitem__) as decode:
                    report = trainer.main(self.argv(dataset, output) + [
                        "--smoke-steps", "2", "--batch-size", "2",
                    ])
                save.assert_not_called()
                self.assertEqual(decode.call_count, 4)
                self.assertEqual(report["result_status"], "NON_RESULT")
                self.assertEqual(report["status"], "complete")
                self.assertEqual(report["adapter_type"], "lora")
                self.assertEqual(report["optimizer_steps_completed"], 2)
                self.assertEqual(report["optimizer_steps_requested"], 2)
                self.assertEqual(report["sampler"]["num_samples"], 4)
                self.assertEqual(report["metadata_row_count"], 65)
                self.assertEqual(report["epochs_completed"], 0)
                self.assertEqual(report["epochs_requested"], 0)
                self.assertEqual(report["selection"], "none")
                self.assertIsNone(report["selected_epoch"])
                evidence = report["evidence"]
                self.assertEqual(evidence["module_types"],
                                 {"visual.proj": "sarclip_adapter.LoRALinear"})
                self.assertEqual(evidence["factor_counts"], {"lora_down": 1, "lora_up": 1})
                self.assertEqual(evidence["factor_tensor_count"], 2)
                self.assertEqual(evidence["factor_numel"], 64)
                self.assertTrue(evidence["base_unchanged"])
                self.assertFalse(evidence["base_requires_grad"])
                self.assertGreater(evidence["factor_max_abs_delta"][
                    "visual.proj.lora_up.weight"], 0)
                self.assertEqual(json.loads((output / "smoke.json").read_text()), report)
                self.assertFalse(list(output.glob("*.pth*")))
                self.assertFalse((output / "train_log.csv").exists())
                with self.assertRaisesRegex(RuntimeError, "NON_RESULT"):
                    trainer.save_final_adapter(output, fake.model, report, 10)

    def test_smoke_sampler_bounds_draws_even_when_dataset_exceeds_budget(self):
        args = trainer.parse_args(self.argv() + ["--smoke-steps", "2", "--batch-size", "3"])
        rows = [self.row(class_id=0)] * 1000 + [self.row(class_id=1)]
        loader = trainer.build_loader(rows, preprocess, args)
        self.assertEqual(len(loader), 2)
        self.assertEqual(loader.sampler.num_samples, 6)
        self.assertEqual(len(list(loader.sampler)), 6)
        self.assertAlmostEqual(loader.sampler.weights[0].item(), 1 / 1000)
        self.assertEqual(loader.sampler.weights[-1].item(), 1)

    def test_smoke_rejects_no_factor_delta_and_changed_base(self):
        self.write_rows([self.row()])
        for failure in ("no_delta", "changed_base", "no_steps", "nonfinite"):
            with self.subTest(failure=failure):
                output = self.root / failure
                original_epoch = trainer.train_one_epoch

                def faulty_epoch(model, classifier, loader, optimizer, device, **kwargs):
                    if failure == "no_steps":
                        return {}
                    if failure == "no_delta":
                        for group in optimizer.param_groups:
                            group["lr"] = 0
                    metrics = original_epoch(model, classifier, loader, optimizer, device, **kwargs)
                    with torch.no_grad():
                        if failure == "changed_base":
                            model.visual.proj.base.weight.add_(1)
                        if failure == "nonfinite":
                            model.visual.proj.lora_up.weight.fill_(float("nan"))
                    return metrics

                with mock.patch.object(trainer, "import_sarclip", return_value=FakeSarclip()), \
                     mock.patch.object(torch.cuda, "is_available", return_value=False), \
                     mock.patch.object(trainer, "train_one_epoch", side_effect=faulty_epoch), \
                     self.assertRaisesRegex(RuntimeError, "Smoke failed"):
                    trainer.main(self.argv(output=output) + ["--smoke-steps", "1"])
                self.assertFalse((output / "smoke.json").exists())
                self.assertFalse(list(output.glob("*.pth*")))
                self.assertEqual(json.loads((output / "config.json").read_text())["status"], "running")

    def test_output_refuses_each_existing_artifact(self):
        for name in ("config.json", "train_log.csv", "lora_rsar.pth",
                     "lora_dior.pth", "lora_dior.pth.partial", "visual_proj_rsar.pth",
                     "smoke.json", "smoke.json.partial"):
            with self.subTest(name=name):
                output = self.root / name.replace(".", "_")
                output.mkdir()
                artifact = output / name
                artifact.write_bytes(b"existing")
                with self.assertRaises(FileExistsError):
                    trainer.prepare_output(output)
                self.assertEqual(artifact.read_bytes(), b"existing")

    def test_failed_or_interrupted_epochs_never_publish_final(self):
        self.write_rows([self.row()])
        metrics = {"loss": 1., "top1_acc": 0., "mean_prob_gt": .1, "mean_entropy": 1.}
        for failure in (RuntimeError("batch failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                output = self.root / type(failure).__name__
                with mock.patch.object(trainer, "import_sarclip", return_value=FakeSarclip()), \
                     mock.patch.object(torch.cuda, "is_available", return_value=False), \
                     mock.patch.object(trainer, "train_one_epoch", side_effect=[metrics, failure]):
                    with self.assertRaises(type(failure)):
                        trainer.main(self.argv(output=output))
                config = json.loads((output / "config.json").read_text())
                self.assertNotEqual(config["status"], "complete")
                self.assertIsNone(config["selected_epoch"])
                self.assertFalse(list(output.glob("*.pth*")))

    def test_save_failure_and_incomplete_epoch_never_publish_final(self):
        output = trainer.prepare_output(self.root / "output")
        args = trainer.parse_args(self.argv())
        model = TinyClip()
        kind, names = trainer.configure_trainable_params(model, args)
        config = trainer.build_config(args, [self.row()], model, kind, names)
        (output / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(RuntimeError, "fully completed"):
            trainer.save_final_adapter(output, model, config, 9)

        def fail_save(checkpoint, path):
            path.write_bytes(b"incomplete checkpoint")
            raise OSError("disk full")

        with mock.patch.object(torch, "save", side_effect=fail_save):
            with self.assertRaisesRegex(OSError, "disk full"):
                trainer.save_final_adapter(output, model, config, 10)
        self.assertFalse((output / "lora_rsar.pth").exists())
        self.assertTrue((output / "lora_rsar.pth.partial").exists())
        self.assertEqual(json.loads((output / "config.json").read_text())["status"], "running")


if __name__ == "__main__":
    unittest.main()
