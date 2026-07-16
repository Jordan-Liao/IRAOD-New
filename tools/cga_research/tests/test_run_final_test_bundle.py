from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from tools.cga_research import build_data_manifest
from tools.cga_research import final_test_manifest
from tools.cga_research import run_final_test_bundle as runner
from tools.cga_research.gpu_scheduler import ComputeApp, GPUInfo


class FakeProcess:
    def __init__(self, returncode: int, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = returncode
        self.terminated = False

    def wait(self, timeout=None):
        del timeout
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.terminated = True


class ControllableProcess:
    def __init__(self, pid: int, released: threading.Event) -> None:
        self.pid = pid
        self.released = released
        self.returncode = 0

    def poll(self):
        return self.returncode if self.released.is_set() else None

    def wait(self, timeout=None):
        self.released.wait(timeout)
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15
        self.released.set()

    def kill(self) -> None:
        self.returncode = -9
        self.released.set()


class SimulatedRunnerCrash(RuntimeError):
    pass


class UnkillableProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    @staticmethod
    def poll():
        return None

    @staticmethod
    def wait(timeout=None):
        raise subprocess.TimeoutExpired("synthetic-unkillable", timeout)

    @staticmethod
    def terminate() -> None:
        return None

    @staticmethod
    def kill() -> None:
        return None


class FinalTestBundleRunnerTests(unittest.TestCase):
    METHODS = ("no_cga", "legacy")
    SEED = 41

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.compute_apps: list[ComputeApp] = []
        self.annotation_root = self.root / "synthetic" / "test" / "annfiles"
        self.image_root = self.root / "synthetic" / "test" / "images"
        self.annotation_root.mkdir(parents=True)
        self.image_root.mkdir(parents=True)
        (self.annotation_root / "sample.txt").write_text(
            "1 2 3 4 ship 0\n", encoding="utf-8"
        )
        (self.image_root / "sample.png").write_bytes(b"synthetic-image")

        self.data_manifest = self.root / "state" / "data_manifest.json"
        build_data_manifest.build_manifest(
            ann_root=self.annotation_root,
            image_root=self.image_root,
            split="test",
            corruption="chaff",
            class_order=(
                "ship",
                "aircraft",
                "car",
                "tank",
                "bridge",
                "harbor",
            ),
            output=self.data_manifest,
        )
        self.config = self._write("configs/final.py", b"model = dict()\n")
        self.evidence = self._write("reports/selection.json", b'{"selected":true}\n')
        self.payload = self._payload()
        draft = self.root / "draft.json"
        draft.write_text(json.dumps(self.payload), encoding="utf-8")
        self.manifest = self.root / "frozen_manifest.json"
        final_test_manifest.create_manifest(draft, self.manifest)

    def _write(self, relative: str, content: bytes) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _artifact(self, path: Path) -> dict[str, str]:
        try:
            rendered = str(path.relative_to(self.root))
        except ValueError:
            rendered = str(path)
        return {
            "path": rendered,
            "sha256": final_test_manifest.sha256_file(path),
        }

    def _checkpoint(self, method: str) -> dict[str, object]:
        path = (
            self.root
            / "training"
            / method
            / f"seed_{self.SEED}"
            / runner.CHECKPOINT_BASENAME
        )
        path.parent.mkdir(parents=True)
        torch.save(
            {
                "meta": {
                    "iter": 4235,
                    "epoch": 1,
                    "seed": self.SEED,
                    "synthetic_method": method,
                },
                "state_dict": {},
            },
            path,
        )
        return {
            "path": str(path.relative_to(self.root)),
            "size_bytes": path.stat().st_size,
            "sha256": final_test_manifest.sha256_file(path),
            "meta": {"iter": 4235, "epoch": 1, "seed": self.SEED},
        }

    def _arm(self, method: str) -> dict[str, object]:
        return {
            "method": method,
            "method_environment": (
                {"CGA_SCORER": "none"}
                if method == "no_cga"
                else {
                    "CGA_SCORER": "sarclip",
                    "CGA_FILTER_MODE": "legacy",
                }
            ),
            "hyperparameters": {
                "corrupt": "chaff",
                "model.cfg.score_thr": 0.9,
            },
            "runs": [
                {
                    "seed": self.SEED,
                    "training_fingerprint": hashlib.sha256(
                        f"{method}:{self.SEED}".encode("utf-8")
                    ).hexdigest(),
                    "checkpoint": self._checkpoint(method),
                    "output_dir": (
                        f"{runner.OUTPUT_ROOT_NAME}/{method}/" f"seed_{self.SEED}"
                    ),
                }
            ],
        }

    def _payload(self) -> dict[str, object]:
        runtime_files = [self._artifact(path) for path in runner.REQUIRED_RUNTIME_FILES]
        return {
            "schema_version": 2,
            "state": final_test_manifest.FROZEN_STATE,
            "project_root": str(self.root),
            "arms": [self._arm(method) for method in self.METHODS],
            "aggregation": {
                "required_seed_set": [self.SEED],
                "paired_comparator": "no_cga",
                "exclude_failed_seed": False,
                "mean": True,
                "sample_std_ddof": 1,
                "bootstrap_samples": 10000,
                "bootstrap_seed": 20260715,
                "paired_t_test": True,
                "exact_sign_flip_permutation": True,
                "cohens_dz": True,
                "holm_correction": True,
            },
            "selection": {
                "evidence": [self._artifact(self.evidence)],
                "config": self._artifact(self.config),
                "runtime_files": runtime_files,
                "data_manifest": self._artifact(self.data_manifest),
            },
        }

    @staticmethod
    def _gpu(index: int = 0) -> GPUInfo:
        return GPUInfo(
            index=index,
            uuid=f"GPU-synthetic-{index}",
            name="Synthetic GPU",
            memory_used_mib=0,
            memory_free_mib=24000,
            memory_total_mib=24000,
            utilization_percent=0,
        )

    def _bundle(self, verify_data: bool = True) -> runner.FrozenBundle:
        return runner.load_frozen_bundle(self.manifest, verify_data=verify_data)

    def _success_factory(self, calls: list[tuple[list[str], dict]]):
        def factory(command, **kwargs):
            calls.append((command, kwargs))
            output = Path(command[command.index("--out") + 1])
            output.write_bytes(b"synthetic-predictions")
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "eval_20260715_123456.json").write_text(
                '{"mAP": 0.5}\n', encoding="utf-8"
            )
            kwargs["stdout"].write(b"synthetic online evaluation output\n")
            gpu_uuid = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            gpu_index = int(gpu_uuid.rsplit("-", 1)[1])
            pid = 4321 + gpu_index
            self.compute_apps.append(ComputeApp(pid, self._gpu(gpu_index).uuid, 1024))
            return FakeProcess(0, pid=pid)

        return factory

    def _registry(self) -> runner.AppendOnlyRegistry:
        return runner.AppendOnlyRegistry(
            runner.registry_path_for_manifest(self.manifest)
        )

    def _process_kwargs(self) -> dict[str, object]:
        return {
            "compute_apps_probe": lambda: list(self.compute_apps),
            "process_start_probe": lambda pid: pid + 100000,
            "gpu_verify_timeout": 1.0,
            "monitor_interval": 0.01,
            "run_timeout": 5.0,
            "stall_timeout": 5.0,
            "terminate_grace": 0.01,
        }

    def _create_completed_run(self):
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        calls: list[tuple[list[str], dict]] = []
        result = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=self._success_factory(calls),
            **self._process_kwargs(),
        )
        self.assertEqual(result.status, "completed")
        return bundle, run, registry, calls

    def _create_unpublished_success(self):
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        calls: list[tuple[list[str], dict]] = []

        def crash_after_terminal(_paths, record):
            self.assertEqual(record["event"], "completed")
            raise SimulatedRunnerCrash("synthetic pre-publication crash")

        with self.assertRaisesRegex(SimulatedRunnerCrash, "pre-publication crash"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=self._success_factory(calls),
                after_attempt_terminal=crash_after_terminal,
                **self._process_kwargs(),
            )
        return bundle, run, registry, calls

    @staticmethod
    def _rewrite_registry(
        registry: runner.AppendOnlyRegistry,
        records: list[dict[str, object]],
    ) -> None:
        registry.path.write_bytes(
            b"".join(runner._canonical_json_bytes(record) for record in records)
        )

    def test_dry_run_does_not_verify_dataset_query_gpu_or_launch(self) -> None:
        output = io.StringIO()
        with (
            mock.patch.object(
                runner.data_manifest_tool,
                "verify_manifest",
                side_effect=AssertionError("dataset bytes were read"),
            ),
            mock.patch.object(
                runner.gpu_scheduler,
                "query_gpu_inventory",
                side_effect=AssertionError("GPU was queried"),
            ),
            mock.patch.object(
                runner.subprocess,
                "Popen",
                side_effect=AssertionError("model was launched"),
            ),
            contextlib.redirect_stdout(output),
        ):
            returncode = runner.main(
                [
                    "--manifest",
                    str(self.manifest),
                    "--python",
                    sys.executable,
                    "--dry-run",
                ]
            )
        self.assertEqual(returncode, 0)
        plan = json.loads(output.getvalue())
        self.assertFalse(plan["dataset_bytes_verified"])
        self.assertFalse(plan["gpu_queried"])
        self.assertEqual(len(plan["runs"]), 2)
        self.assertEqual(
            set(plan["runs"][0]["outputs"]),
            {
                "framework_eval",
                "online_eval_metadata",
                "predictions",
                "stdout",
            },
        )

    def test_manifest_verification_precedes_dataset_verification(self) -> None:
        order: list[str] = []
        manifest_verify = runner.manifest_tool.verify_manifest
        dataset_verify = runner.data_manifest_tool.verify_manifest

        def verify_manifest(path):
            order.append("manifest")
            return manifest_verify(path)

        def verify_dataset(path):
            order.append("dataset")
            return dataset_verify(path)

        with (
            mock.patch.object(
                runner.manifest_tool,
                "verify_manifest",
                side_effect=verify_manifest,
            ),
            mock.patch.object(
                runner.data_manifest_tool,
                "verify_manifest",
                side_effect=verify_dataset,
            ),
        ):
            self._bundle(verify_data=True)
        self.assertEqual(order[:2], ["manifest", "dataset"])

    def test_command_has_exact_safe_overrides_and_argument_list(self) -> None:
        bundle = self._bundle(verify_data=False)
        run = bundle.runs[0]
        command = runner.build_test_command(bundle, run, Path(sys.executable))
        self.assertIsInstance(command, list)
        self.assertEqual(command[1], str(runner.REQUIRED_RUNTIME_FILES[1]))
        self.assertEqual(command[3], str(run.checkpoint_path))
        self.assertEqual(command[command.index("--out") + 1], str(run.predictions_path))
        self.assertEqual(command[command.index("--work-dir") + 1], str(run.output_dir))
        self.assertIn("corrupt=chaff", command)
        self.assertIn("model.ema_config=None", command)
        self.assertIn("model.ema_ckpt=None", command)
        self.assertIn(f"data.test.ann_file={self.annotation_root}", command)
        self.assertIn(f"data.test.img_prefix={self.image_root}", command)
        self.assertEqual(command[command.index("--eval") + 1], "mAP")
        self.assertEqual(command[-2:], ["iou_thr=0.5", "nproc=4"])

    def test_environment_removes_inherited_cga_and_sarclip_values(
        self,
    ) -> None:
        environment = runner.build_run_environment(
            {
                "PATH": "/bin",
                "CGA_SCORER": "sarclip",
                "CGA_EXTRA_FUTURE_KEY": "unsafe",
                "SARCLIP_LORA": "/secret/model.pth",
                "PYTHONPATH": "/untrusted/python",
                "PYTHONHOME": "/untrusted/home",
                "LD_PRELOAD": "/untrusted/preload.so",
                "LD_LIBRARY_PATH": "/untrusted/lib",
            },
            gpu_uuid=self._gpu(3).uuid,
            python=Path(sys.executable),
        )
        self.assertEqual(
            {
                key: value
                for key, value in environment.items()
                if key.startswith("CGA_")
            },
            runner.DISABLED_CGA_ENV,
        )
        self.assertNotIn("SARCLIP_LORA", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], self._gpu(3).uuid)
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        prefix = Path(sys.executable).resolve().parent.parent
        self.assertEqual(environment["IRAOD_CONDA_PREFIX"], str(prefix))
        self.assertEqual(environment["LD_LIBRARY_PATH"], str(prefix / "lib"))
        for key in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD"):
            self.assertNotIn(key, environment)

    def test_success_writes_fixed_outputs_registry_and_verified_skip(
        self,
    ) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        calls: list[tuple[list[str], dict]] = []
        with (
            mock.patch.object(
                runner.manifest_tool,
                "verify_manifest",
                wraps=runner.manifest_tool.verify_manifest,
            ) as manifest_verify,
            mock.patch.object(
                runner.data_manifest_tool,
                "verify_manifest",
                wraps=runner.data_manifest_tool.verify_manifest,
            ) as dataset_verify,
        ):
            result = runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={"CGA_SCORER": "sarclip"},
                popen_factory=self._success_factory(calls),
                **self._process_kwargs(),
            )
        self.assertGreaterEqual(manifest_verify.call_count, 1)
        self.assertGreaterEqual(dataset_verify.call_count, 1)
        self.assertEqual(result.status, "completed")
        self.assertEqual(
            {path.name for path in run.output_dir.iterdir()},
            runner.CORE_OUTPUT_NAMES | {runner.ATTEMPTS_DIRNAME},
        )
        records = registry.read()
        self.assertEqual(
            [record["event"] for record in records], ["started", "completed"]
        )
        self.assertEqual(records[-1]["returncode"], 0)
        self.assertEqual(records[-1]["gpu_uuid"], self._gpu().uuid)
        self.assertEqual(
            records[0]["environment"],
            runner.relevant_environment(calls[0][1]["env"]),
        )
        self.assertEqual(
            records[0]["environment"]["CUDA_VISIBLE_DEVICES"],
            self._gpu().uuid,
        )
        self.assertIsInstance(calls[0][0], list)
        self.assertNotIn("shell", calls[0][1])
        self.assertEqual(
            {
                key: value
                for key, value in calls[0][1]["env"].items()
                if key.startswith("CGA_")
            },
            runner.DISABLED_CGA_ENV,
        )

        registry_before = registry.path.read_bytes()
        skipped = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(1),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=mock.Mock(
                side_effect=AssertionError("verified success was rerun")
            ),
            **self._process_kwargs(),
        )
        self.assertEqual(skipped.status, "skipped_verified")
        self.assertEqual(registry.path.read_bytes(), registry_before)

        (run.output_dir / runner.POSTPROCESS_BASENAME).write_text(
            '{"ship": 0.5}\n', encoding="utf-8"
        )
        postprocess_skip = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(1),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=mock.Mock(
                side_effect=AssertionError("postprocessed run was rerun")
            ),
            **self._process_kwargs(),
        )
        self.assertEqual(postprocess_skip.status, "skipped_verified")

        inventory = mock.Mock(
            side_effect=AssertionError("verified success queried a GPU")
        )
        single_run_bundle = runner.dataclasses.replace(bundle, runs=(run,))
        results = runner.execute_bundle(
            bundle=single_run_bundle,
            registry=registry,
            python=Path(sys.executable),
            max_workers=1,
            max_attempts=3,
            required_free_mib=8192,
            utilization_limit=30,
            gpu_poll_interval=0.01,
            max_gpu_wait=0,
            base_environment={},
            inventory_probe=inventory,
            **self._process_kwargs(),
        )
        self.assertEqual(results[0].status, "skipped_verified")
        inventory.assert_not_called()

    def test_nonzero_process_retries_at_most_three_times(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        calls = []

        def failing_factory(command, **kwargs):
            calls.append((command, kwargs))
            kwargs["stdout"].write(b"synthetic failure\n")
            pid = 5000 + len(calls)
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            return FakeProcess(7, pid=pid)

        result = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=failing_factory,
            **self._process_kwargs(),
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(calls), 3)
        records = registry.read()
        self.assertEqual(
            [record["event"] for record in records],
            ["started", "failed"] * 3,
        )
        with self.assertRaisesRegex(runner.FinalTestBundleError, "attempt limit"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=mock.Mock(),
                **self._process_kwargs(),
            )

    def test_concurrent_runners_share_one_exclusive_run_lease(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        released = threading.Event()
        launched = threading.Event()
        calls: list[list[str]] = []

        def blocking_factory(command, **kwargs):
            calls.append(command)
            output = Path(command[command.index("--out") + 1])
            output.write_bytes(b"synthetic-predictions")
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "eval_20260715_123456.json").write_text(
                '{"mAP": 0.5}\n', encoding="utf-8"
            )
            kwargs["stdout"].write(b"blocking synthetic process\n")
            pid = 6101
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            launched.set()
            return ControllableProcess(pid, released)

        outcomes: list[object] = []

        def first_runner() -> None:
            try:
                outcomes.append(
                    runner.run_one_frozen_run(
                        initial_bundle=bundle,
                        run=run,
                        gpu=self._gpu(),
                        registry=registry,
                        python=Path(sys.executable),
                        max_attempts=1,
                        base_environment={},
                        popen_factory=blocking_factory,
                        **self._process_kwargs(),
                    )
                )
            except BaseException as error:  # surfaced in the test thread
                outcomes.append(error)

        thread = threading.Thread(target=first_runner, daemon=True)
        thread.start()
        self.assertTrue(launched.wait(timeout=2.0))
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "lease is already held"
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(1),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=mock.Mock(
                    side_effect=AssertionError("duplicate child was launched")
                ),
                **self._process_kwargs(),
            )
        released.set()
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outcomes), 1)
        self.assertIsInstance(outcomes[0], runner.RunResult)
        self.assertEqual(outcomes[0].status, "completed")
        self.assertEqual(
            [record["event"] for record in registry.read()],
            ["started", "completed"],
        )

    def test_terminal_success_recovers_after_runner_crash(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        calls: list[tuple[list[str], dict]] = []

        def crash_after_terminal(paths, record):
            self.assertTrue(paths.terminal.is_file())
            self.assertEqual(record["event"], "completed")
            raise SimulatedRunnerCrash("crash before registry completion")

        with self.assertRaisesRegex(SimulatedRunnerCrash, "before registry completion"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=self._success_factory(calls),
                after_attempt_terminal=crash_after_terminal,
                **self._process_kwargs(),
            )
        self.assertEqual([record["event"] for record in registry.read()], ["started"])
        self.assertFalse(run.predictions_path.exists())

        recovered = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(1),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=mock.Mock(
                side_effect=AssertionError("completed attempt was rerun")
            ),
            **self._process_kwargs(),
        )
        self.assertEqual(recovered.status, "skipped_verified")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [record["event"] for record in registry.read()],
            ["started", "completed"],
        )
        verified = runner.verify_completed_run(self.manifest, run.method, run.seed)
        self.assertEqual(verified["event"], "completed")

    def test_orphan_started_without_terminal_never_reruns(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        released = threading.Event()

        def running_factory(command, **kwargs):
            Path(command[command.index("--out") + 1]).write_bytes(
                b"untrusted-partial-predictions"
            )
            kwargs["stdout"].write(b"runner will crash\n")
            pid = 6201
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            return ControllableProcess(pid, released)

        def crashing_sleep(_seconds):
            raise SimulatedRunnerCrash("monitor crashed")

        process_kwargs = self._process_kwargs()
        process_kwargs["sleeper"] = crashing_sleep
        with self.assertRaisesRegex(SimulatedRunnerCrash, "monitor crashed"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=running_factory,
                **process_kwargs,
            )
        attempt = runner._attempt_paths(run, 1)
        self.assertFalse(attempt.terminal.exists())
        self.assertEqual([record["event"] for record in registry.read()], ["started"])
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "automatic rerun is forbidden"
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=mock.Mock(side_effect=AssertionError("orphan was rerun")),
                **self._process_kwargs(),
            )

    def test_failed_attempt_is_preserved_before_new_retry_directory(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()
        calls = 0

        def retry_factory(command, **kwargs):
            nonlocal calls
            calls += 1
            pid = 6300 + calls
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            kwargs["stdout"].write(f"attempt {calls}\n".encode("utf-8"))
            if calls == 1:
                return FakeProcess(7, pid=pid)
            Path(command[command.index("--out") + 1]).write_bytes(
                b"successful-retry-predictions"
            )
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "eval_20260715_123457.json").write_text(
                '{"mAP": 0.6}\n', encoding="utf-8"
            )
            return FakeProcess(0, pid=pid)

        result = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=retry_factory,
            **self._process_kwargs(),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.attempts, 2)
        attempt_one = runner._attempt_paths(run, 1)
        attempt_two = runner._attempt_paths(run, 2)
        first_stdout = attempt_one.stdout.read_bytes()
        self.assertIn(b"final-test attempt 1", first_stdout)
        self.assertTrue(first_stdout.endswith(b"attempt 1\n"))
        self.assertNotIn(b"attempt 2", first_stdout)
        first_terminal = json.loads(attempt_one.terminal.read_text(encoding="utf-8"))
        self.assertEqual(first_terminal["event"], "failed")
        self.assertTrue(attempt_two.terminal.is_file())
        self.assertTrue(run.framework_eval_path.is_file())
        self.assertEqual(
            [record["event"] for record in registry.read()],
            ["started", "failed", "started", "completed"],
        )

    def test_completed_history_rejects_records_appended_after_completion(
        self,
    ) -> None:
        _bundle, run, registry, _calls = self._create_completed_run()
        forged = copy.deepcopy(registry.read()[-1])
        forged.update(
            {
                "event": "failed",
                "failure": "forged post-completion event",
                "returncode": 9,
            }
        )
        registry.append(forged)
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "records after completion"
        ):
            runner.verify_completed_run(self.manifest, run.method, run.seed)

    def test_completed_registry_provenance_must_match_terminal(self) -> None:
        _bundle, run, registry, _calls = self._create_completed_run()
        original = registry.read()
        attacks = {
            "gpu": lambda record: (
                record.__setitem__("gpu_uuid", "GPU-tampered"),
                record["environment"].__setitem__(
                    "CUDA_VISIBLE_DEVICES", "GPU-tampered"
                ),
            ),
            "pid": lambda record: (
                record.__setitem__("pid", record["pid"] + 1),
                record.__setitem__("pid_start_ticks", record["pid_start_ticks"] + 1),
            ),
            "command": lambda record: record["command"].append("--tampered"),
        }
        for name, mutate in attacks.items():
            with self.subTest(name=name):
                records = copy.deepcopy(original)
                mutate(records[-1])
                self._rewrite_registry(registry, records)
                with self.assertRaises(runner.FinalTestBundleError):
                    runner.verify_completed_run(self.manifest, run.method, run.seed)
                self._rewrite_registry(registry, original)

    def test_completed_history_rejects_unregistered_attempt_directory(
        self,
    ) -> None:
        _bundle, run, _registry, _calls = self._create_completed_run()
        orphan = run.output_dir / runner.ATTEMPTS_DIRNAME / "attempt_002"
        orphan.mkdir()
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "attempt directory set differs"
        ):
            runner.verify_completed_run(self.manifest, run.method, run.seed)

    def test_registry_is_closed_world_for_current_manifest(self) -> None:
        _bundle, run, registry, _calls = self._create_completed_run()
        rogue = copy.deepcopy(registry.read()[-1])
        rogue.update(
            {
                "event": "failed",
                "failure": "rogue undeclared run",
                "method": "rogue",
                "seed": 999,
                "run_identity": hashlib.sha256(b"rogue-run").hexdigest(),
                "returncode": 9,
            }
        )
        registry.append(rogue)
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "exactly one frozen run"
        ):
            runner.verify_completed_run(self.manifest, run.method, run.seed)

    def test_canonical_copy_is_inode_independent_and_preserves_attempt(
        self,
    ) -> None:
        _bundle, run, _registry, _calls = self._create_completed_run()
        paths = runner._attempt_paths(run, 1)
        terminal = json.loads(paths.terminal.read_text(encoding="utf-8"))
        for name in runner.CORE_OUTPUT_NAMES:
            source = Path(terminal["artifacts"][name]["path"])
            canonical = run.output_dir / name
            self.assertNotEqual(source.stat().st_ino, canonical.stat().st_ino)
        original_attempt = paths.predictions.read_bytes()
        os.chmod(run.predictions_path, 0o600)
        run.predictions_path.write_bytes(b"tampered canonical only")
        self.assertEqual(paths.predictions.read_bytes(), original_attempt)
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "artifact identity mismatch"
        ):
            runner.verify_completed_run(self.manifest, run.method, run.seed)

    def test_anonymous_copy_crash_recovers_without_staging_residue(
        self,
    ) -> None:
        bundle, run, registry, calls = self._create_unpublished_success()

        def crash_mid_copy(source_descriptor, staging_descriptor, **_kwargs):
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            chunk = os.read(source_descriptor, 4)
            self.assertTrue(chunk)
            os.write(staging_descriptor, chunk)
            raise SimulatedRunnerCrash("anonymous copy interrupted")

        with (
            mock.patch.object(
                runner,
                "_copy_to_publication_staging",
                side_effect=crash_mid_copy,
            ),
            self.assertRaisesRegex(SimulatedRunnerCrash, "anonymous copy interrupted"),
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(1),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=mock.Mock(
                    side_effect=AssertionError("completed attempt was rerun")
                ),
                **self._process_kwargs(),
            )
        self.assertEqual(set(run.output_dir.iterdir()), {run.output_dir / "attempts"})
        self.assertEqual([record["event"] for record in registry.read()], ["started"])

        recovered = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(1),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=mock.Mock(
                side_effect=AssertionError("completed attempt was rerun")
            ),
            **self._process_kwargs(),
        )
        self.assertEqual(recovered.status, "skipped_verified")
        self.assertEqual(len(calls), 1)

    def test_crash_immediately_after_atomic_link_recovers(self) -> None:
        bundle, run, registry, calls = self._create_unpublished_success()
        real_link = runner._link_anonymous_file_noreplace

        def link_then_crash(*args, **kwargs):
            real_link(*args, **kwargs)
            raise SimulatedRunnerCrash("crash after atomic publication")

        with (
            mock.patch.object(
                runner,
                "_link_anonymous_file_noreplace",
                side_effect=link_then_crash,
            ),
            self.assertRaisesRegex(SimulatedRunnerCrash, "after atomic publication"),
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(1),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=mock.Mock(
                    side_effect=AssertionError("completed attempt was rerun")
                ),
                **self._process_kwargs(),
            )
        published = [
            path
            for path in run.output_dir.iterdir()
            if path.name in runner.CORE_OUTPUT_NAMES
        ]
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].stat().st_nlink, 1)

        recovered = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(1),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=3,
            base_environment={},
            popen_factory=mock.Mock(
                side_effect=AssertionError("completed attempt was rerun")
            ),
            **self._process_kwargs(),
        )
        self.assertEqual(recovered.status, "skipped_verified")
        self.assertEqual(len(calls), 1)

    def test_preplanted_staging_shaped_file_is_closed_world_rejected(
        self,
    ) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        run.output_dir.mkdir(parents=True)
        planted = (
            run.output_dir
            / ".publish-predictions.pkl-0123456789abcdef0123456789abcdef.tmp"
        )
        planted.write_bytes(b"attacker-controlled")
        with self.assertRaisesRegex(runner.FinalTestBundleError, "unknown entry"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=self._registry(),
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=mock.Mock(
                    side_effect=AssertionError("unsafe run was launched")
                ),
                **self._process_kwargs(),
            )

    def test_multiply_linked_attempt_artifact_is_rejected(self) -> None:
        _bundle, run, _registry, _calls = self._create_completed_run()
        paths = runner._attempt_paths(run, 1)
        os.link(paths.predictions, self.root / "external-attempt-link.pkl")
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "exactly one link|single-link"
        ):
            runner.verify_completed_run(self.manifest, run.method, run.seed)

    def test_preplanted_registry_and_run_lease_hardlinks_are_rejected(
        self,
    ) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        external_registry = self.root / "external-registry.txt"
        external_registry.write_text("external registry\n", encoding="utf-8")
        os.link(external_registry, runner.registry_path_for_manifest(self.manifest))
        with self.assertRaisesRegex(runner.FinalTestBundleError, "exactly one link"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=self._registry(),
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=mock.Mock(),
                **self._process_kwargs(),
            )
        self.assertEqual(
            external_registry.read_text(encoding="utf-8"),
            "external registry\n",
        )

        runner.registry_path_for_manifest(self.manifest).unlink()
        run = bundle.runs[1]
        lock_root = self.root / runner.LOCK_ROOT_NAME / "runs"
        lock_root.mkdir(parents=True, exist_ok=True)
        lease_path = lock_root / f"{runner._run_identity(bundle, run)}.lock"
        external_lease = self.root / "external-lease.txt"
        external_lease.write_text("external lease\n", encoding="utf-8")
        os.link(external_lease, lease_path)
        with self.assertRaisesRegex(runner.FinalTestBundleError, "exactly one link"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=self._registry(),
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=mock.Mock(),
                **self._process_kwargs(),
            )
        self.assertEqual(external_lease.read_text(encoding="utf-8"), "external lease\n")

    def test_preplanted_gpu_lease_hardlink_is_rejected(self) -> None:
        lock_root = self.root / runner.LOCK_ROOT_NAME / "gpus"
        lock_root.mkdir(parents=True)
        gpu = self._gpu()
        lock_name = hashlib.sha256(gpu.uuid.encode("utf-8")).hexdigest() + ".lock"
        external = self.root / "external-gpu-lease.txt"
        external.write_text("external GPU lease\n", encoding="utf-8")
        os.link(external, lock_root / lock_name)
        with self.assertRaisesRegex(runner.FinalTestBundleError, "exactly one link"):
            runner.GPUExecutionLease(lock_root, gpu.uuid)
        self.assertEqual(external.read_text(encoding="utf-8"), "external GPU lease\n")

    def test_registered_failed_pid_without_started_never_retries(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()

        def failed_factory(_command, **kwargs):
            kwargs["stdout"].write(b"synthetic failed child\n")
            pid = 6701
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            return FakeProcess(7, pid=pid)

        result = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=1,
            base_environment={},
            popen_factory=failed_factory,
            **self._process_kwargs(),
        )
        self.assertEqual(result.status, "failed")
        failed = registry.read()[-1]
        self.assertEqual(failed["event"], "failed")
        self._rewrite_registry(registry, [failed])
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "invalid started/terminal pairing"
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=2,
                base_environment={},
                popen_factory=mock.Mock(
                    side_effect=AssertionError("ambiguous failure was retried")
                ),
                **self._process_kwargs(),
            )

    def test_completed_terminal_without_started_is_rejected(self) -> None:
        _bundle, run, registry, _calls = self._create_completed_run()
        completed = registry.read()[-1]
        self._rewrite_registry(registry, [completed])
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "invalid started/terminal pairing"
        ):
            runner.verify_completed_run(self.manifest, run.method, run.seed)

    def test_unproven_child_termination_leaves_orphan_without_retry(
        self,
    ) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()

        def unkillable_factory(_command, **kwargs):
            kwargs["stdout"].write(b"unkillable synthetic child\n")
            pid = 6601
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            return UnkillableProcess(pid)

        times = iter((0.0, 0.0, 2.0))
        process_kwargs = self._process_kwargs()
        process_kwargs.update(
            {
                "clock": lambda: next(times),
                "run_timeout": 1.0,
                "stall_timeout": 10.0,
                "sleeper": lambda _seconds: None,
            }
        )
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "termination could not be proven"
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=unkillable_factory,
                **process_kwargs,
            )
        self.assertEqual([record["event"] for record in registry.read()], ["started"])
        self.assertFalse(runner._attempt_paths(run, 1).terminal.exists())

    def test_scheduler_runs_only_manifest_runs_on_distinct_safe_gpus(
        self,
    ) -> None:
        bundle = self._bundle()
        registry = self._registry()
        calls: list[tuple[list[str], dict]] = []
        results = runner.execute_bundle(
            bundle=bundle,
            registry=registry,
            python=Path(sys.executable),
            max_workers=2,
            max_attempts=1,
            required_free_mib=8192,
            utilization_limit=30,
            gpu_poll_interval=0.01,
            max_gpu_wait=0,
            base_environment={},
            inventory_probe=lambda: [self._gpu(0), self._gpu(1)],
            compute_apps_probe=lambda: list(self.compute_apps),
            popen_factory=self._success_factory(calls),
            process_start_probe=lambda pid: pid + 100000,
            gpu_verify_timeout=1.0,
            monitor_interval=0.01,
            run_timeout=5.0,
            stall_timeout=5.0,
            terminate_grace=0.01,
        )
        self.assertEqual(
            {(result.method, result.seed) for result in results},
            {(method, self.SEED) for method in self.METHODS},
        )
        self.assertTrue(all(result.status == "completed" for result in results))
        self.assertEqual(len(calls), len(bundle.runs))
        for _command, kwargs in calls:
            self.assertEqual(len(kwargs["pass_fds"]), 2)
            self.assertEqual(len(set(kwargs["pass_fds"])), 2)
        self.assertEqual(
            {record["gpu_uuid"] for record in registry.read()},
            {self._gpu(0).uuid, self._gpu(1).uuid},
        )

    def test_gpu_inventory_allows_real_driver_reserved_memory(self) -> None:
        observed = GPUInfo(
            index=0,
            uuid="GPU-real-shape",
            name="Observed NVIDIA GPU",
            memory_used_mib=12043,
            memory_free_mib=12173,
            memory_total_mib=24564,
            utilization_percent=0,
        )
        runner._validate_gpu_inventory([observed])

        invalid = (
            GPUInfo(0, "GPU-a", "GPU", -1, 1, 1, 0),
            GPUInfo(0, "GPU-a", "GPU", 2, 0, 1, 0),
            GPUInfo(0, "GPU-a", "GPU", 1, 1, 1, 0),
            GPUInfo(0, "GPU-a", "GPU", 0, 1, 1, -1),
            GPUInfo(0, "GPU-a", "GPU", 0, 1, 1, 101),
        )
        for gpu in invalid:
            with self.subTest(gpu=gpu):
                with self.assertRaises(runner.FinalTestBundleError):
                    runner._validate_gpu_inventory([gpu])

    def test_invalid_gpu_inventory_blocks_launch(self) -> None:
        bundle = self._bundle()
        inventory = mock.Mock(return_value=[GPUInfo(0, "GPU-bad", "GPU", 0, 1, 1, 101)])
        popen = mock.Mock(side_effect=AssertionError("invalid GPU launched"))
        with self.assertRaisesRegex(runner.FinalTestBundleError, "utilization"):
            runner.execute_bundle(
                bundle=bundle,
                registry=self._registry(),
                python=Path(sys.executable),
                max_workers=1,
                max_attempts=1,
                required_free_mib=8192,
                utilization_limit=30,
                gpu_poll_interval=0.01,
                max_gpu_wait=0,
                base_environment={},
                inventory_probe=inventory,
                popen_factory=popen,
                **self._process_kwargs(),
            )
        inventory.assert_called_once()
        popen.assert_not_called()

    def test_nonfinite_intervals_fail_before_gpu_query_or_launch(self) -> None:
        bundle = self._bundle()
        inventory = mock.Mock(side_effect=AssertionError("GPU was queried"))
        popen = mock.Mock(side_effect=AssertionError("child was launched"))
        base = {
            "bundle": bundle,
            "registry": self._registry(),
            "python": Path(sys.executable),
            "max_workers": 1,
            "max_attempts": 1,
            "required_free_mib": 8192,
            "utilization_limit": 30,
            "gpu_poll_interval": 0.01,
            "max_gpu_wait": 0.0,
            "base_environment": {},
            "gpu_verify_timeout": 1.0,
            "monitor_interval": 0.01,
            "run_timeout": 5.0,
            "stall_timeout": 5.0,
            "terminate_grace": 0.01,
            "inventory_probe": inventory,
            "compute_apps_probe": lambda: [],
            "popen_factory": popen,
        }
        attacks = {
            "gpu_poll_interval": math.nan,
            "max_gpu_wait": math.inf,
            "gpu_verify_timeout": math.nan,
            "monitor_interval": math.inf,
            "run_timeout": math.nan,
            "stall_timeout": math.inf,
            "terminate_grace": math.nan,
        }
        for role, value in attacks.items():
            with self.subTest(role=role):
                arguments = dict(base)
                arguments[role] = value
                with self.assertRaisesRegex(runner.FinalTestBundleError, "finite"):
                    runner.execute_bundle(**arguments)
        inventory.assert_not_called()
        popen.assert_not_called()

    def test_cli_rejects_nonfinite_timeout_before_gpu_or_launch(self) -> None:
        inventory = mock.Mock(side_effect=AssertionError("GPU was queried"))
        popen = mock.Mock(side_effect=AssertionError("child was launched"))
        stderr = io.StringIO()
        with (
            mock.patch.object(runner.gpu_scheduler, "query_gpu_inventory", inventory),
            mock.patch.object(runner.subprocess, "Popen", popen),
            contextlib.redirect_stderr(stderr),
        ):
            returncode = runner.main(
                [
                    "--manifest",
                    str(self.manifest),
                    "--python",
                    sys.executable,
                    "--run-timeout",
                    "nan",
                ]
            )
        self.assertEqual(returncode, 2)
        self.assertIn("finite", stderr.getvalue())
        inventory.assert_not_called()
        popen.assert_not_called()

    def test_unsupported_anonymous_publication_fails_before_launch(
        self,
    ) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        popen = mock.Mock(side_effect=AssertionError("child was launched"))
        with (
            mock.patch.object(
                runner,
                "_open_anonymous_publication_staging",
                side_effect=runner.FinalTestBundleError(
                    "synthetic O_TMPFILE unsupported"
                ),
            ),
            self.assertRaisesRegex(
                runner.FinalTestBundleError, "O_TMPFILE unsupported"
            ),
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=self._registry(),
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=popen,
                **self._process_kwargs(),
            )
        popen.assert_not_called()
        self.assertFalse((run.output_dir / runner.ATTEMPTS_DIRNAME).exists())

    def test_wrong_gpu_uuid_binding_fails_without_canonical_outputs(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()

        def wrong_gpu_factory(command, **kwargs):
            Path(command[command.index("--out") + 1]).write_bytes(
                b"wrong-gpu-predictions"
            )
            work_dir = Path(command[command.index("--work-dir") + 1])
            (work_dir / "eval_20260715_123456.json").write_text(
                '{"mAP": 0.5}\n', encoding="utf-8"
            )
            kwargs["stdout"].write(b"wrong GPU\n")
            pid = 6401
            self.compute_apps.append(ComputeApp(pid, self._gpu(1).uuid, 1024))
            return FakeProcess(0, pid=pid)

        result = runner.run_one_frozen_run(
            initial_bundle=bundle,
            run=run,
            gpu=self._gpu(0),
            registry=registry,
            python=Path(sys.executable),
            max_attempts=1,
            base_environment={},
            popen_factory=wrong_gpu_factory,
            **self._process_kwargs(),
        )
        self.assertEqual(result.status, "failed")
        self.assertFalse(run.predictions_path.exists())
        records = registry.read()
        self.assertEqual([record["event"] for record in records], ["started", "failed"])
        self.assertIn("unexpected GPU UUID", records[-1]["failure"])

    def test_success_without_framework_eval_fails_closed(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        registry = self._registry()

        def missing_eval_factory(command, **kwargs):
            Path(command[command.index("--out") + 1]).write_bytes(
                b"predictions-without-eval"
            )
            kwargs["stdout"].write(b"no framework JSON\n")
            pid = 6501
            self.compute_apps.append(ComputeApp(pid, self._gpu().uuid, 1024))
            return FakeProcess(0, pid=pid)

        with self.assertRaisesRegex(runner.FinalTestBundleError, "exactly one eval_"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=registry,
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=missing_eval_factory,
                **self._process_kwargs(),
            )
        self.assertFalse(run.predictions_path.exists())
        self.assertEqual([record["event"] for record in registry.read()], ["started"])

    def test_arbitrary_registry_path_is_rejected(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "fixed manifest-local path"
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=runner.AppendOnlyRegistry(self.root / "alternate.jsonl"),
                python=Path(sys.executable),
                max_attempts=1,
                base_environment={},
                popen_factory=mock.Mock(
                    side_effect=AssertionError("unsafe registry launched")
                ),
                **self._process_kwargs(),
            )

    def test_unknown_output_entry_is_rejected_without_launch(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        run.output_dir.mkdir(parents=True)
        (run.output_dir / "unexpected.txt").write_text("unsafe", encoding="utf-8")
        with self.assertRaisesRegex(runner.FinalTestBundleError, "unknown entry"):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=self._registry(),
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=mock.Mock(),
                **self._process_kwargs(),
            )

    def test_symbolic_link_output_is_rejected(self) -> None:
        bundle = self._bundle()
        run = bundle.runs[0]
        run.output_dir.mkdir(parents=True)
        run.predictions_path.symlink_to(self.config)
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "symbolic-link|non-symlink|single-link"
        ):
            runner.run_one_frozen_run(
                initial_bundle=bundle,
                run=run,
                gpu=self._gpu(),
                registry=self._registry(),
                python=Path(sys.executable),
                max_attempts=3,
                base_environment={},
                popen_factory=mock.Mock(),
                **self._process_kwargs(),
            )

    def test_registry_rejects_duplicate_json_keys(self) -> None:
        registry = self._registry()
        registry.path.write_text(
            '{"event":"failed","event":"failed"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(runner.FinalTestBundleError, "duplicate key"):
            registry.read()

    def test_worker_and_attempt_bounds_fail_before_gpu_queries(self) -> None:
        bundle = self._bundle()
        registry = self._registry()
        inventory = mock.Mock(side_effect=AssertionError("GPU query occurred"))
        with self.assertRaisesRegex(runner.FinalTestBundleError, "max_workers"):
            runner.execute_bundle(
                bundle=bundle,
                registry=registry,
                python=Path(sys.executable),
                max_workers=5,
                max_attempts=3,
                required_free_mib=8192,
                utilization_limit=30,
                gpu_poll_interval=0.01,
                max_gpu_wait=0,
                base_environment={},
                inventory_probe=inventory,
                **self._process_kwargs(),
            )
        inventory.assert_not_called()

    def test_manifest_runtime_file_drift_fails_closed(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["selection"]["runtime_files"] = payload["selection"]["runtime_files"][
            1:
        ]
        draft = self.root / "missing_runner_draft.json"
        draft.write_text(json.dumps(payload), encoding="utf-8")
        frozen = self.root / "missing_runner.json"
        final_test_manifest.create_manifest(draft, frozen)
        with self.assertRaisesRegex(
            runner.FinalTestBundleError, "required runtime files"
        ):
            runner.load_frozen_bundle(frozen, verify_data=False)


if __name__ == "__main__":
    unittest.main()
