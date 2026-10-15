"""CPU-only selected-host paths, frozen config views, locks and native routing."""

from contextlib import ExitStack, nullcontext
import copy
import json
import os
import pickle
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.comparison import host_binding as host
from experiments.comparison import extension_training as training
from experiments.comparison import finite_resumer as finite
from experiments.comparison import mixed_queue
from experiments.comparison import smoke_frozen_training as smoke


FIXTURE = Path(__file__).parent / "fixtures/b_reg_0e5f8f7_dior_clean_42.json"
PYTHON = "/home/zechuan/miniforge3/envs/iraod/bin/python"
ART = "/mnt/shared/zechuan/iraod_artifacts"


def target(stack):
    stack.enter_context(patch.object(host.socket, "gethostname", return_value=host.TARGET_HOST))
    for module, name in ((training, "ALLOWED_GPUS"), (finite, "APPROVED"),
                         (mixed_queue, "ALLOWED_GPUS")):
        stack.enter_context(patch.object(module, name, host.approved_gpus()))
    for module in (training, finite, mixed_queue):
        stack.enter_context(patch.object(module, "PAIR_PORTS", host.pair_ports()))


class HostBindingTest(unittest.TestCase):
    def test_nested_rsar_ema_config_resolves_original_relative_base(self):
        from mmcv import Config

        teacher_name = "configs/baseline/ema_config/baseline_oriented_rcnn_ema_rsar_cga_orthonet.py"
        teacher = training.ROOT / teacher_name
        base = teacher.with_name("baseline_oriented_rcnn_ema_rsar_cga.py")
        before = {path: path.read_bytes() for path in (teacher, base)}
        expected = Config.fromfile(str(teacher), import_custom_modules=False).to_dict()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            previous_cwd = Path.cwd()
            stack.callback(os.chdir, previous_cwd)
            os.chdir(training.ROOT)
            config = Path(directory) / "b_reg_rsar.py"
            config.write_text(f"model=dict(ema_config={teacher_name!r})\n")
            original_loader = Config._file2dict
            with host.native_config_paths():
                student = Config.fromfile(str(config), import_custom_modules=False)
                actual = Config.fromfile(
                    student.model.ema_config, import_custom_modules=False)
            self.assertEqual(actual.to_dict(), expected)
            self.assertEqual(actual.model.backbone.type, "OrthoNet")
            self.assertEqual(actual.model.roi_head.bbox_head.num_classes, 6)
            self.assertIs(Config._file2dict, original_loader)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_ddp_launcher_rank_before_native_is_forwarded(self):
        for option, rank in (("--local-rank=0", 0), ("--local_rank=1", 1)):
            with self.subTest(option=option), ExitStack() as stack:
                old_path = list(sys.path)
                stack.callback(lambda: sys.path.__setitem__(slice(None), old_path))
                entry = "/bound/frozen/train.py"
                arguments = ["config.py", "--launcher", "pytorch",
                             "--cfg-options", "data.samples_per_gpu=16"]
                stack.enter_context(patch.object(sys, "argv", [
                    host.__file__, option, "native", entry, *arguments]))
                stack.enter_context(patch.object(host, "is_target_host", return_value=False))
                stack.enter_context(patch.object(host, "native_config_paths", return_value=nullcontext()))
                stack.enter_context(patch.object(host, "native_integrity", return_value=nullcontext()))

                def native(path, run_name):
                    self.assertEqual(path, entry)
                    self.assertEqual(run_name, "__main__")
                    self.assertEqual(sys.argv, [entry, f"--local-rank={rank}", *arguments])

                run = stack.enter_context(patch.object(host.runpy, "run_path", side_effect=native))
                host.main()
                run.assert_called_once()

    def test_actual_two_process_cpu_launcher_forwards_both_local_ranks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "rank_probe.py"
            probe.write_text(
                "import argparse,json,os\nfrom pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "p.add_argument('--local-rank','--local_rank',type=int,required=True)\n"
                "p.add_argument('--out-dir',required=True)\n"
                "a=p.parse_args()\n"
                "assert a.local_rank == int(os.environ['LOCAL_RANK'])\n"
                "(Path(a.out_dir)/f'rank_{a.local_rank}.json').write_text("
                "json.dumps({'argument_rank':a.local_rank,'environment_rank':int(os.environ['LOCAL_RANK'])}))\n")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command = [
                sys.executable, "-m", "torch.distributed.launch", "--nproc_per_node=2",
                "--master_addr=127.0.0.1", f"--master_port={port}",
                str(Path(host.__file__).absolute()), "native", str(probe), "--out-dir", str(root),
            ]
            result = subprocess.run(
                command, cwd=training.ROOT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                     "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                text=True, capture_output=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            for rank in (0, 1):
                self.assertEqual(json.loads((root / f"rank_{rank}.json").read_text()),
                                 {"argument_rank": rank, "environment_rank": rank})

    def test_stdlib_only_import(self):
        result = subprocess.run([
            sys.executable, "-S", "-c",
            "from experiments.comparison import host_binding; import sys; "
            "assert not {'torch', 'mmcv', 'numpy'} & sys.modules.keys()"],
            cwd=training.ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cold_native_entry_retains_frozen_imports_and_maps_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "frozen"
            (frozen / "sfod").mkdir(parents=True)
            (frozen / "sfod/__init__.py").write_text("SOURCE='frozen'\n")
            (frozen / "iraod_runtime.py").write_text(
                "import os\n"
                "def ensure_iraod_runtime():\n"
                "    assert os.environ.get('IRAOD_RUNTIME_READY') == '1', 'would re-exec away overlay'\n"
                f"    assert os.environ['CONDA_PREFIX'] == {host.PYTHON_PREFIX!r}\n")
            config = frozen / "config.py"
            config.write_text(
                "model=dict(source='/mnt/shared/zechuan/iraod_artifacts/source.pth')\n"
                "data=dict(samples_per_gpu=16)\noptimizer=dict(lr=.02)\n")
            before = config.read_bytes()
            entry = frozen / "train.py"
            entry.write_text(
                "import sys\nfrom iraod_runtime import ensure_iraod_runtime\n"
                "ensure_iraod_runtime()\nfrom mmcv import Config\nimport sfod\n"
                "assert sfod.SOURCE == 'frozen'\n"
                "cfg=Config.fromfile(sys.argv[1])\n"
                "assert cfg.model.source == '/home/zechuan/iraod_artifacts/source.pth'\n"
                "assert cfg.data.samples_per_gpu == 16 and cfg.optimizer.lr == .02\n"
                "print('NATIVE_PATH_OVERLAY_OK')\n")
            bootstrap = (
                "import runpy,socket,sys; "
                f"socket.gethostname=lambda:{host.TARGET_HOST!r}; "
                f"sys.argv={[str(Path(host.__file__).absolute()), 'native', str(entry), str(config)]!r}; "
                f"runpy.run_path({str(Path(host.__file__).absolute())!r},run_name='__main__')")
            result = subprocess.run(
                [sys.executable, "-c", bootstrap], cwd=root,
                env={**os.environ, "PYTHONPATH": str(training.ROOT),
                     "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
                     "IRAOD_RUNTIME_READY": "0", "CONDA_PREFIX": "/not-the-selected-env"},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("NATIVE_PATH_OVERLAY_OK", result.stdout)
            self.assertEqual(config.read_bytes(), before)

    def test_exact_host_prefixes_and_scientific_values(self):
        raw = json.loads(FIXTURE.read_text())
        original = copy.deepcopy(raw)
        with patch.object(host.socket, "gethostname", return_value=host.TARGET_HOST):
            self.assertTrue(host.is_target_host())
            mapped = host.map_data(raw)
            self.assertEqual(mapped["training_code"], "/home/zechuan/IRAOD-New-strict-af")
            for key in ("source_id", "training_code_sha", "unlabeled_epoch_size", "seed"):
                self.assertEqual(mapped[key], raw[key])
            self.assertEqual(host.map_path(Path(ART) / "source.pth"),
                             "/home/zechuan/iraod_artifacts/source.pth")
            self.assertEqual(host.map_path("/mnt/HDD_14TB/zechuan/iraod_artifacts/source.pth"),
                             "/home/zechuan/iraod_artifacts/source.pth")
            self.assertEqual(host.map_path("/mnt/shared/zechuan/iraod_data/DIOR"),
                             "/home/zechuan/iraod_data/DIOR")
            self.assertEqual(host.map_path("/mnt/shared/zechuan/iraod_weights/sarclip/model"),
                             "/home/zechuan/iraod_weights/sarclip/model")
            self.assertEqual(host.map_path(PYTHON), PYTHON)
            data = {"lr": .02, "batch": 32, "world_sizes": (1, 2), "budgets": [185, 266, 160000],
                    "notes": "source was " + ART, "other": ART + "_other",
                    ART: [ART + "/weights", (False, None)]}
            view = host.map_data(data)
            self.assertEqual(view[ART][0], "/home/zechuan/iraod_artifacts/weights")
            for key in ("lr", "batch", "world_sizes", "budgets", "notes", "other"):
                self.assertEqual(view[key], data[key])
        self.assertEqual(raw, original)
        for name in ("old.67", host.TARGET_HOST.lower(), host.TARGET_HOST + ".domain"):
            with patch.object(host.socket, "gethostname", return_value=name):
                self.assertFalse(host.is_target_host())
                self.assertEqual(host.map_data(raw), raw)
                self.assertEqual(host.approved_gpus(), (4, 5, 6, 7))

    def test_selected_host_path_identity_handles_actual_symlink_without_changing_python(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            root = Path(directory)
            storage = root / "storage"
            storage.mkdir()
            alias = root / "home_alias"
            alias.symlink_to(storage, target_is_directory=True)
            (storage / "config.py").write_text("model={}\n")
            self.assertTrue(host.same_path(str(alias / "config.py"), str(storage / "config.py")))
            self.assertEqual(host.map_path(PYTHON), PYTHON)

    def test_native_evaluation_uses_target_zero_and_recognizes_home_hdd_identity(self):
        import numpy as np
        from experiments.comparison import extension_manifest
        from experiments.comparison.report_inputs import CLASSES
        from experiments.comparison.result_completion import write_json

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            root = Path(directory)
            out = root / "eval"
            checkpoint = ART + "/comparison/RSAR/iter_266.pth"
            config = ART + "/config.py"
            cell = {
                "dataset": "RSAR", "domain": "clean", "method": "IRG", "seed": 42,
                "role": "student", "checkpoint": checkpoint, "config": config,
                "eval_dir": str(out), "training_code_sha": "actual-training",
                "ann_file": "/mnt/shared/zechuan/iraod_data/RSAR/test/annfiles",
                "img_prefix": "/mnt/shared/zechuan/iraod_data/RSAR/test/images",
            }
            original = copy.deepcopy(cell)
            runtime = {"python": PYTHON, "evaluation_code": "/mnt/SSD2_8TB/zechuan/IRAOD-New-rc331d213",
                       "evaluation_code_sha": "native331"}
            stack.enter_context(patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}))
            stack.enter_context(patch.dict(extension_manifest.EXPECTED_IMAGES, {"RSAR": 1}))

            def native(command, **kwargs):
                self.assertEqual(command[0], PYTHON)
                self.assertIn(str(Path(host.__file__).absolute()), command)
                self.assertIn("/home/zechuan/IRAOD-New-rc331d213/test.py", command)
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(kwargs["env"]["RSAR_ROOT"], "/home/zechuan/iraod_data/RSAR")
                self.assertEqual(kwargs["cwd"], Path("/home/zechuan/IRAOD-New-rc331d213"))
                kwargs["stdout"].write("| class | ap |\n" + "".join(
                    f"| {name} | 0.2 |\n" for name in CLASSES["RSAR"]) + "| mAP | 0.2 |\n")
                hdd = "/mnt/HDD_14TB/zechuan/iraod_artifacts"
                write_json(out / "predictions.pkl.image_ids.json", {
                    "schema": "iraod-prediction-image-order-v1", "origin": "inference_batch_img_metas",
                    "status": "complete", "predictions_file": "predictions.pkl",
                    "checkpoint": checkpoint.replace(ART, hdd), "config": config.replace(ART, hdd),
                    "evaluation_code_sha": "native331", "training_code_sha": "actual-training",
                    "n_images": 1, "dataset_size": 1, "image_ids": ["test"],
                    "records": [{"prediction_index": 0, "image_id": "test", "ori_filename": "test.png"}],
                })
                write_json(out / "eval_fixture.json",
                           {"config": host.map_path(config), "metric": {"mAP": .2}})
                with (out / "predictions.pkl").open("wb") as stream:
                    pickle.dump([[np.zeros((0, 6)) for _ in CLASSES["RSAR"]]], stream)
                return SimpleNamespace(returncode=0)

            stack.enter_context(patch.object(extension_manifest.subprocess, "run", side_effect=native))
            extension_manifest.evaluate_binding(cell, runtime, 0)
            paths = SimpleNamespace(
                eval_student_dir=lambda *args: str(out), student_path=lambda *args: host.map_path(checkpoint),
                EXPECT_PRED={"RSAR": 1}, EVALUATION_CODE_SHA="native331")
            self.assertEqual(finite.eval_state(root, paths, finite.Cell(
                "RSAR", "clean", 42, "IRG", "student")), "complete")
            self.assertEqual(cell, original)
            with self.assertRaises(ValueError):
                extension_manifest.evaluate_binding(cell, runtime, 5)
        with patch.object(host.socket, "gethostname", return_value="7T83-8xA100-67"):
            with self.assertRaises(ValueError):
                extension_manifest.evaluate_binding(cell, runtime, 0)

    def test_old_and_new_gpu_domains_pairs_and_ports(self):
        with patch.object(training, "_train", return_value="routed"):
            for gpu in range(4):
                with self.assertRaises(ValueError):
                    training.train("q", gpu, "DIOR", "clean", 42, "B_REG")
            self.assertEqual(finite.allowed_gpus(SimpleNamespace(
                DATA={"allowed_gpus": list(range(8))})), (4, 5, 6, 7))
            self.assertEqual(host.pair_ports(), {(4, 5): 29804, (6, 7): 29806})
            with ExitStack() as stack:
                target(stack)
                self.assertEqual(host.approved_gpus(), (0, 1, 2, 3, 4))
                self.assertEqual(host.pair_ports(), {(0, 1): 29804, (2, 3): 29806})
                self.assertEqual(finite.allowed_gpus(SimpleNamespace(
                    DATA={"allowed_gpus": [4, 5, 6, 7]})), (0, 1, 2, 3, 4))
                for gpu in (5, 6, 7):
                    with self.assertRaises(ValueError):
                        training.train("q", gpu, "DIOR", "clean", 42, "B_REG")
                for pair, port in (("0,1", 29804), ("2,3", 29806)):
                    self.assertEqual(training.train_ddp(
                        "q", pair, port, "DIOR", "clean", 42, "F_text_only"), "routed")
                for pair, port in (("4,5", 29804), ("2,3", 29804), ("0,2", 29804)):
                    with self.assertRaises(ValueError):
                        training.train_ddp("q", pair, port, "DIOR", "clean", 42, "F_text_only")

    def test_copied_symlink_resolver_and_original_runtime_are_read_only(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            root = Path(directory)
            stack.enter_context(patch.object(host, "PREFIXES", (
                ("/old-owner", str(root)), *host.PREFIXES)))
            original = root / "original"
            original.mkdir()
            raw = {"cells": {"DIOR/clean/42/B_REG": json.loads(FIXTURE.read_text())},
                   "python": PYTHON, "training_code": "/mnt/SSD2_8TB/zechuan/IRAOD-New-strict-af"}
            text = json.dumps(raw)
            (original / "runtime.json").write_text(text)
            (original / "paths.py").write_text(
                "import json\nfrom pathlib import Path\n"
                "DATA=json.loads(Path('/old-owner/original/runtime.json').read_text())\n"
                "def binding(ds,domain,seed,method):\n"
                "    return DATA['cells'][f'{ds}/{domain}/{seed}/{method}']\n"
                "def ema_path(*args): return binding(*args)['checkpoint']\n")
            queue = root / "release"
            queue.mkdir()
            (queue / "paths.py").symlink_to("/old-owner/original/paths.py")
            (queue / "with_gpu_lock.sh").symlink_to("/old-owner/original/with_gpu_lock.sh")
            (queue / "runtime.json").write_text(json.dumps({
                "cells": raw["cells"], "source_queues": {
                    "DIOR/clean/42/B_REG": "/old-owner/original"}}))
            paths = finite.load_paths(queue)
            runtime, cell = training.load_cell(queue, "DIOR", "clean", 42, "B_REG")
            self.assertEqual(runtime["python"], PYTHON)
            self.assertEqual(paths.ema_path("DIOR", "clean", "42", "B_REG"), cell["checkpoint"])
            self.assertEqual(cell["training_code"], "/home/zechuan/IRAOD-New-strict-af")
            self.assertEqual((original / "runtime.json").read_text(), text)
            self.assertFalse((original / "__pycache__").exists())
            self.assertEqual(finite.lock_root(queue), Path(host.SHARED_LOCK_ROOT))

    def test_native_config_base_environment_and_numeric_values_unchanged(self):
        from mmcv import Config
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            stack.enter_context(patch.dict(os.environ))
            root = Path(directory)
            stack.enter_context(patch.object(host, "PREFIXES", (
                ("/old-owner", str(root)), *host.PREFIXES)))
            base = root / "baseline.py"
            base.write_text(
                "optimizer=dict(lr=.02, momentum=.9)\n"
                "data=dict(samples_per_gpu=32)\n"
                "model=dict(cfg=dict(use_bbox_reg=False), "
                "ema_ckpt='/mnt/shared/zechuan/iraod_artifacts/source.pth')\n"
                "runner=dict(max_epochs=1)\n")
            # Exact legacy B_REG overlay shape: old absolute _base_ + one boolean.
            overlay = root / "b_reg_dior.py"
            overlay.write_text("_base_='/old-owner/baseline.py'\n"
                               "model=dict(cfg=dict(use_bbox_reg=True))\n")
            f_config = root / "f.py"
            f_config.write_text(
                "_base_='baseline.py'\n"
                "data=dict(samples_per_gpu=16)\n"
                "import os as _f_os\n"
                "_f_os.environ['SARCLIP_PRETRAINED']="
                "'/mnt/shared/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors'\n"
                "_f_os.environ['CGA_BLEND_DET_WEIGHT']='0.7'\n"
                "del _f_os\n")
            contents = {p: p.read_bytes() for p in (base, overlay, f_config)}
            original_loader = Config._file2dict
            with host.native_config_paths():
                cfg = Config.fromfile(str(overlay))
                f_cfg = Config.fromfile(str(f_config))
            self.assertIs(Config._file2dict, original_loader)
            self.assertTrue(cfg.model.cfg.use_bbox_reg)
            self.assertEqual(cfg.optimizer.lr, .02)
            self.assertEqual(cfg.data.samples_per_gpu, 32)
            self.assertEqual(f_cfg.data.samples_per_gpu, 16)
            self.assertEqual(cfg.runner.max_epochs, 1)
            self.assertEqual(cfg.model.ema_ckpt, "/home/zechuan/iraod_artifacts/source.pth")
            self.assertEqual(os.environ["SARCLIP_PRETRAINED"],
                             "/home/zechuan/iraod_weights/sarclip/ViT-B-32/vit_b_32_model.safetensors")
            self.assertEqual(os.environ["CGA_BLEND_DET_WEIGHT"], "0.7")
            for path, content in contents.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertEqual(set(root.iterdir()), set(contents))

    def test_actual_legacy_breg_smoke_selflocks_and_routes_mapped_native(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            root = Path(directory)
            stack.enter_context(patch.object(host, "SHARED_LOCK_ROOT", str(root / "gpu_locks")))
            raw = json.loads(FIXTURE.read_text())
            queue = root / "copied_Q"
            queue.mkdir()
            contents = json.dumps({"cells": {"DIOR/clean/42/B_REG": raw},
                                   "python": PYTHON, "training_code": raw["training_code"]})
            (queue / "runtime.json").write_text(contents)
            out = root / "NON_RESULT"
            stack.enter_context(patch.object(training, "require_file", side_effect=Path))
            stack.enter_context(patch.object(training, "code_sha", return_value=raw["training_code_sha"]))
            stack.enter_context(patch.object(finite, "idle_devices", return_value={4}))
            def launched(command, **kwargs):
                self.assertEqual(command[0], PYTHON)
                self.assertEqual(kwargs["cwd"], Path("/home/zechuan/IRAOD-New-strict-af"))
                self.assertIn(str(Path(host.__file__).absolute()), command)
                self.assertIn("--native", command)
                self.assertIn("--frozen-train", command)
                self.assertIn("data.samples_per_gpu=32", command)
                self.assertIn("optimizer.lr=0.02", command)
                self.assertIn("model.cfg.use_bbox_reg=True", command)
                self.assertEqual(kwargs["env"]["CONDA_PREFIX"], str(Path(PYTHON).parent.parent))
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "4")
                self.assertIsNone(finite.take_lock(root / "gpu_locks/gpu4.lock"))
                (out / "rank_0.json").write_text(json.dumps({
                    "bounded_pass": True, "optimizer_updates": 2}))
            native = stack.enter_context(patch.object(smoke.subprocess, "run", side_effect=launched))
            smoke.run(queue, "DIOR", "clean", 42, "B_REG", "4", out)
            native.assert_called_once()
            self.assertEqual((queue / "runtime.json").read_text(), contents)
            self.assertEqual(set(queue.iterdir()), {queue / "runtime.json"})
            # IRAOD_GPU_LOCKED never bypasses smoke's real lock.
            outer = finite.take_lock(root / "gpu_locks/gpu4.lock")
            stack.callback(outer.close)
            stack.enter_context(patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}))
            with self.assertRaisesRegex(RuntimeError, "shared lock busy"):
                smoke.run(queue, "DIOR", "clean", 42, "B_REG", "4", root / "blocked")
            native.assert_called_once()

    def test_outer_lock_and_finite_command_share_target_contract(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            root = Path(directory)
            stack.enter_context(patch.object(host, "SHARED_LOCK_ROOT", str(root / "gpu_locks")))
            stack.enter_context(patch.object(finite, "idle_devices", return_value={0, 1}))
            def launch(command, env):
                self.assertEqual(command, ["true"])
                self.assertEqual(env["IRAOD_GPU_LOCKED"], "1")
                for gpu in (0, 1):
                    self.assertIsNone(finite.take_lock(finite.lock_root("/copied_Q") / f"gpu{gpu}.lock"))
                return 0
            stack.enter_context(patch("subprocess.call", side_effect=launch))
            self.assertEqual(host.locked_command((0, 1), ["true"]), 0)
            stack.enter_context(patch.object(training, "load_cell", return_value=(
                {"python": PYTHON}, {})))
            command = finite.runner_command(ART + "/Q", finite.Cell("DIOR", "clean", 42, "F_text_only"),
                                            "train", (0, 1))
            self.assertEqual(command[:3], [PYTHON, str(finite.SCRIPT.with_name("extension_training.py")),
                                           "train-ddp"])
            self.assertIn("/home/zechuan/iraod_artifacts/Q", command)
            self.assertEqual(command[-4:], ["--gpu-pair", "0,1", "--port", "29804"])

    def test_aasfod_helper_is_current_but_model_checkout_stays_bound(self):
        raw = json.loads(FIXTURE.read_text())
        raw.update(method="AASFOD", training_code=ART + "/code_a39", training_code_sha="a39")
        with ExitStack() as stack:
            target(stack)
            stack.enter_context(patch.object(training, "require_file", side_effect=Path))
            stack.enter_context(patch.object(training, "code_sha", return_value="a39"))
            code, sha, command, env = training.training_invocation(
                ART + "/Q", {"python": PYTHON, "training_code": raw["training_code"]},
                raw, (4,), None, ART + "/work")
            self.assertEqual(command[:2], [
                PYTHON, str(training.ROOT / "experiments/comparison/train_aasfod.py")])
            self.assertEqual(command[command.index("--queue") + 1],
                             "/home/zechuan/iraod_artifacts/Q")
            self.assertEqual(str(code), "/home/zechuan/iraod_artifacts/code_a39")
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [str(training.ROOT), str(code)])
            self.assertEqual(sha, "a39")

    def test_formal_aasfod_stages_execute_bound_model_code(self):
        from experiments.comparison import train_aasfod
        from mmcv import Config
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            target(stack)
            root = Path(directory)
            code = root / "bound_code"
            code.mkdir()
            work = root / "work"
            config = root / "config.py"
            config.write_text("model=dict(cfg={})\ndata=dict(train={})\n")
            cell = dict(
                aasfod_budget={"alignment_updates": 2, "fns_updates": 3, "ema_interval": 1},
                training_code=str(code), work_dir=str(work), config=str(config),
                tsd_split=str(root / "tsd.json"), target_val=str(root / "val"),
                source_checkpoint=str(root / "source.pth"), unlabeled_epoch_size=5,
                student_checkpoint=str(work / "final.pth"), checkpoint=str(work / "final_ema.pth"))
            (root / "tsd.json").write_text("{}")
            stack.enter_context(patch.dict(os.environ, {"IRAOD_GPU_LOCKED": "1"}))
            stack.enter_context(patch.object(train_aasfod, "load_cell", return_value=(
                {"python": PYTHON, "training_code": str(code)}, cell)))
            stack.enter_context(patch.object(train_aasfod, "validate_split"))
            old_cwd, old_path = Path.cwd(), list(sys.path)
            stack.callback(os.chdir, old_cwd)
            stack.callback(lambda: sys.path.__setitem__(slice(None), old_path))
            calls = []
            def launched(command, cwd, env, check):
                self.assertEqual(cwd, code)
                self.assertEqual(env["PYTHONPATH"], str(code))
                index = command.index(str(code / "train.py"))
                self.assertEqual(command[index - 1], "native")
                stage = Config.fromfile(command[index + 1])
                self.assertEqual(stage.optimizer.lr, .02)
                out = Path(stage.work_dir)
                out.mkdir()
                for suffix in ("", "_ema"):
                    (out / f"iter_{stage.runner.max_iters}{suffix}.pth").write_bytes(b"checkpoint")
                calls.append((stage.runner.max_iters, stage.data.samples_per_gpu))
            stack.enter_context(patch.object(train_aasfod.subprocess, "run", side_effect=launched))
            train_aasfod.run("/mapped/Q", "DIOR", "clean", 42)
            self.assertEqual(calls, [(2, 16), (3, 32)])
            self.assertEqual((work / "final.pth").read_bytes(), b"checkpoint")

    def test_tam_completed_payload_maps_external_encoder_without_tensor_changes(self):
        from experiments.comparison import tam_artifacts
        from types import ModuleType
        import torch
        identity = {"dataset": "DIOR", "domain": "clean", "seed": 42}
        weights = torch.tensor([1., 2.])
        payload = {
            "schema": tam_artifacts.SCHEMA, "status": "complete", "training_steps": 160000,
            "identity": identity, "normalization": tam_artifacts.NORMALIZATION,
            "encoder_weights": ART + "/third_party/vgg16_oxford/encoder.pt",
            "components": {name: {"weights": weights} for name in ("decoder", "F1", "F2")}}
        seen = []
        class Component:
            def load_state_dict(self, state, strict):
                self.asserted = strict
                seen.append(state["weights"])
        class NativeTAM:
            def __init__(self, weights_path):
                seen.append(weights_path)
                self.decoder, self.F1, self.F2 = Component(), Component(), Component()
            def to(self, device):
                return self
            def freeze_for_inference(self):
                return self
        module = ModuleType("sfod.extensions.tam")
        module.TargetAugmentationModule = NativeTAM
        with ExitStack() as stack:
            target(stack)
            stack.enter_context(patch.dict(sys.modules, {"sfod.extensions.tam": module}))
            load = stack.enter_context(patch.object(torch, "load", return_value=payload))
            tam_artifacts.load_completed_tam(ART + "/tam.pth", identity, "cpu")
            self.assertEqual(load.call_args.args[0], "/home/zechuan/iraod_artifacts/tam.pth")
            self.assertEqual(seen[0],
                             "/home/zechuan/iraod_artifacts/third_party/vgg16_oxford/encoder.pt")
            self.assertTrue(all(item is weights for item in seen[1:]))
            self.assertEqual(payload["encoder_weights"], ART + "/third_party/vgg16_oxford/encoder.pt")


if __name__ == "__main__":
    unittest.main()
