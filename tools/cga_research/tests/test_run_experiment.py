from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.cga_research.run_experiment import (
    ComputeApp,
    ExperimentSpec,
    build_environment,
    build_train_command,
    classify_failure,
    environment_mismatches,
    parse_actual_seed,
    parse_final_ema_map,
    parse_progress,
    read_proc_environment,
    run_experiment,
)
from tools.cga_research import run_experiment as runner_module


class FakeProcess:
    def __init__(self, poll_values, pid=4321):
        self.pid = pid
        self._poll_values = list(poll_values)
        self.returncode = None

    def poll(self):
        if self._poll_values:
            value = self._poll_values.pop(0)
            if value is not None:
                self.returncode = value
            return value
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class RunExperimentTests(unittest.TestCase):
    def make_spec(self, root: Path, **changes):
        values = dict(
            python="/opt/iraod/bin/python",
            project_root=root,
            config=root / "config.py",
            work_dir=root / "work dir",
            seed=41,
            method="legacy",
            gpu_index=2,
            gpu_uuid="GPU-test-uuid",
            method_env={
                "CGA_SCORER": "sarclip",
                "CGA_FILTER_MODE": "legacy",
                "SARCLIP_LORA": "/tmp/lora file.pth",
            },
            cfg_options=["corrupt=chaff", "model.cfg.score_thr=0.9"],
            monitor_interval_seconds=0.001,
            stall_timeout_seconds=10.0,
            gpu_verify_timeout_seconds=10.0,
            terminate_grace_seconds=0.01,
        )
        values.update(changes)
        return ExperimentSpec(**values)

    def test_command_is_argument_list_and_dangerous_text_is_literal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "must_not_exist"
            option = f"note=$(touch {marker})"
            spec = self.make_spec(root, cfg_options=[option])
            command = build_train_command(spec)
            self.assertIsInstance(command, list)
            self.assertIn("--work-dir", command)
            self.assertIn("--seed", command)
            self.assertIn("--deterministic", command)
            self.assertIn(option, command)
            self.assertFalse(marker.exists())

    def test_environment_scrubs_inherited_method_values(self):
        environment = build_environment(
            {
                "PATH": "/bin",
                "CGA_FILTER_MODE": "stale",
                "SARCLIP_LORA": "/stale.pth",
            },
            {"CGA_SCORER": "none"},
            gpu_index=3,
            python="/iraod/python",
        )
        self.assertEqual(environment["CGA_SCORER"], "none")
        self.assertNotIn("CGA_FILTER_MODE", environment)
        self.assertNotIn("SARCLIP_LORA", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_method_environment_rejects_unknown_cga_key(self):
        with self.assertRaises(ValueError):
            build_environment({}, {"CGA_TYPO_DOES_NOT_EXIST": "1"}, 0, "/python")

    def test_progress_seed_and_failure_parsers(self):
        text = (
            "Set random seed to 43, deterministic: True\n"
            "Epoch [1][10/999]\tloss: 1.0\n"
        )
        self.assertEqual(parse_actual_seed(text), (43, True))
        progress = parse_progress(text)
        self.assertEqual(
            (progress.epoch, progress.iteration, progress.total), (1, 10, 999)
        )
        self.assertEqual(classify_failure("CUDA out of memory"), "cuda_oom")
        self.assertEqual(classify_failure("ModuleNotFoundError: x"), "import_error")
        self.assertEqual(
            classify_failure("CGA initialization failed"), "cga_initialization_failed"
        )
        self.assertEqual(classify_failure("loss: nan"), "nan_detected")
        self.assertEqual(classify_failure("loss_bbox: inf"), "nan_detected")
        self.assertEqual(classify_failure("grad_norm=-Infinity"), "nan_detected")

    def test_proc_environment_reader_and_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            proc_root = Path(directory)
            process_dir = proc_root / "123"
            process_dir.mkdir()
            (process_dir / "environ").write_bytes(
                b"CUDA_VISIBLE_DEVICES=2\0CGA_SCORER=none\0"
            )
            actual = read_proc_environment(123, proc_root)
            self.assertEqual(actual["CGA_SCORER"], "none")
            mismatches = environment_mismatches(
                {**actual, "CGA_STALE": "1"},
                {"CUDA_VISIBLE_DEVICES": "3", "CGA_SCORER": "none"},
            )
            self.assertEqual(len(mismatches), 2)
            self.assertTrue(any("CGA_STALE" in item for item in mismatches))

    def test_final_map_uses_last_timestamp_log_and_excludes_run_log(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            (work_dir / "run_train.log").write_text(
                "Epoch(val) [1][1] mAP: 0.9999\n", encoding="utf-8"
            )
            first = work_dir / "20260101_000000.log"
            second = work_dir / "20260101_000001.log"
            first.write_text("Epoch(val) [1][1] mAP: 0.6000\n", encoding="utf-8")
            second.write_text(
                "Epoch(val) [1][1] mAP: 0.6100\n" "Epoch(val) [1][1] mAP: 0.6200\n",
                encoding="utf-8",
            )
            first.touch()
            second.touch()
            self.assertEqual(parse_final_ema_map(work_dir), 0.62)

    def test_dry_run_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.make_spec(Path(directory))
            popen = mock.Mock(side_effect=AssertionError("must not spawn"))
            outcome = run_experiment(spec, dry_run=True, popen_factory=popen)
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.status, "dry_run")
            popen.assert_not_called()
            payload = json.loads(
                (spec.work_dir / "run_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "dry_run")
            self.assertRegex(payload["experiment_fingerprint"], r"^[0-9a-f]{64}$")
            self.assertIn("config_sha256", payload["fingerprint_components"])

    def test_fingerprint_changes_with_config_options_environment_seed_and_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.py").write_text("value = 1\n", encoding="utf-8")
            (root / "train.py").write_text("print('v1')\n", encoding="utf-8")
            (root / "sfod").mkdir()
            (root / "sfod" / "rotated_unbiased_teacher.py").write_text(
                "DYNAMIC_THRESHOLD = 1\n", encoding="utf-8"
            )
            lora = root / "lora.pth"
            lora.write_bytes(b"lora-v1")
            method_env = {
                "CGA_SCORER": "sarclip",
                "CGA_FILTER_MODE": "legacy",
                "SARCLIP_LORA": str(lora),
            }
            spec = self.make_spec(root, method_env=method_env)
            first = runner_module.compute_experiment_fingerprint(spec)
            self.assertEqual(first, runner_module.compute_experiment_fingerprint(spec))
            self.assertIn("sfod/rotated_unbiased_teacher.py", first["code_files"])
            changed_option = self.make_spec(
                root,
                cfg_options=["corrupt=chaff", "model.cfg.score_thr=0.8"],
                method_env=method_env,
            )
            self.assertNotEqual(
                first["sha256"],
                runner_module.compute_experiment_fingerprint(changed_option)["sha256"],
            )
            changed_env = self.make_spec(
                root,
                method_env={"CGA_SCORER": "none"},
            )
            self.assertNotEqual(
                first["sha256"],
                runner_module.compute_experiment_fingerprint(changed_env)["sha256"],
            )
            changed_seed = self.make_spec(root, seed=42, method_env=method_env)
            self.assertNotEqual(
                first["sha256"],
                runner_module.compute_experiment_fingerprint(changed_seed)["sha256"],
            )
            changed_python = self.make_spec(
                root, python="/different/python", method_env=method_env
            )
            self.assertNotEqual(
                first["sha256"],
                runner_module.compute_experiment_fingerprint(changed_python)["sha256"],
            )
            lora.write_bytes(b"lora-v2")
            self.assertNotEqual(
                first["sha256"],
                runner_module.compute_experiment_fingerprint(spec)["sha256"],
            )
            lora.write_bytes(b"lora-v1")
            (root / "train.py").write_text("print('v2')\n", encoding="utf-8")
            self.assertNotEqual(
                first["sha256"],
                runner_module.compute_experiment_fingerprint(spec)["sha256"],
            )

    def test_successful_fake_process_requires_gpu_seed_and_final_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            process = FakeProcess([None, 0])
            expected_environment = build_environment(
                {}, spec.method_env, spec.gpu_index, spec.python
            )

            def factory(command, **kwargs):
                self.assertIsInstance(command, list)
                self.assertNotIn("shell", kwargs)
                kwargs["stdout"].write(
                    "Set random seed to 41, deterministic: True\n"
                    "Epoch [1][10/10] loss: 0.1\n"
                )
                kwargs["stdout"].flush()
                (spec.work_dir / "20990101_000000.log").write_text(
                    "Set random seed to 41, deterministic: True\n"
                    "Epoch(val) [1][10] mAP: 0.6543\n",
                    encoding="utf-8",
                )
                return process

            outcome = run_experiment(
                spec,
                popen_factory=factory,
                environment_reader=lambda pid: expected_environment,
                compute_apps_probe=lambda: [
                    ComputeApp(process.pid, spec.gpu_uuid, 6400)
                ],
                snapshotter=lambda *_: None,
                sleeper=lambda _: None,
            )
            self.assertTrue(outcome.success)
            self.assertEqual(outcome.final_map, 0.6543)
            self.assertEqual(outcome.final_val_epoch, 1)
            self.assertEqual(outcome.final_val_iteration, 10)
            self.assertEqual(outcome.actual_seed, 41)
            self.assertTrue(outcome.gpu_seen)

    def test_monitor_detects_oom_and_terminates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            process = FakeProcess([None, None])
            expected_environment = build_environment(
                {}, spec.method_env, spec.gpu_index, spec.python
            )
            terminated = []

            def factory(command, **kwargs):
                kwargs["stdout"].write("CUDA out of memory\n")
                kwargs["stdout"].flush()
                return process

            def terminate(fake_process, grace):
                terminated.append((fake_process.pid, grace))
                fake_process.returncode = -15
                return -15

            outcome = run_experiment(
                spec,
                popen_factory=factory,
                environment_reader=lambda pid: expected_environment,
                compute_apps_probe=lambda: [
                    ComputeApp(process.pid, spec.gpu_uuid, 6400)
                ],
                snapshotter=lambda *_: None,
                terminator=terminate,
                sleeper=lambda _: None,
            )
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.failure_kind, "cuda_oom")
            self.assertTrue(terminated)

    def test_compute_app_probe_failure_is_fail_closed_and_monitored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            process = FakeProcess([None, None])
            expected_environment = build_environment(
                {}, spec.method_env, spec.gpu_index, spec.python
            )
            terminated = []

            def factory(command, **kwargs):
                kwargs["stdout"].write("Set random seed to 41, deterministic: True\n")
                kwargs["stdout"].flush()
                return process

            def terminate(fake_process, grace):
                terminated.append(fake_process.pid)
                fake_process.returncode = -15
                return -15

            outcome = run_experiment(
                spec,
                popen_factory=factory,
                environment_reader=lambda pid: expected_environment,
                compute_apps_probe=mock.Mock(side_effect=RuntimeError("probe failed")),
                snapshotter=lambda *_: None,
                terminator=terminate,
                sleeper=lambda _: None,
            )
            self.assertEqual(outcome.failure_kind, "compute_app_probe_failed")
            self.assertTrue(terminated)
            events = [
                json.loads(line)
                for line in (spec.work_dir / "monitor.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertFalse(events[-1]["compute_probe_ok"])

    def test_unexpected_post_spawn_exception_terminates_and_persists_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            process = FakeProcess([None, None])
            expected_environment = build_environment(
                {}, spec.method_env, spec.gpu_index, spec.python
            )
            terminated = []

            def factory(command, **kwargs):
                return process

            def terminate(fake_process, grace):
                terminated.append(fake_process.pid)
                fake_process.returncode = -15
                return -15

            outcome = run_experiment(
                spec,
                popen_factory=factory,
                environment_reader=lambda pid: expected_environment,
                compute_apps_probe=lambda: [],
                snapshotter=mock.Mock(side_effect=RuntimeError("snapshot exploded")),
                terminator=terminate,
                sleeper=lambda _: None,
            )
            self.assertFalse(outcome.success)
            self.assertEqual(outcome.failure_kind, "runner_exception")
            self.assertTrue(terminated)
            self.assertTrue((spec.work_dir / "run_result.json").is_file())

    def test_old_log_is_not_accepted_for_current_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.make_spec(root)
            spec.work_dir.mkdir(parents=True)
            old_log = spec.work_dir / "20000101_000000.log"
            old_log.write_text(
                "Set random seed to 41, deterministic: True\n"
                "Epoch [1][10/10] loss: 0.1\n"
                "Epoch(val) [1][10] mAP: 0.9999\n",
                encoding="utf-8",
            )
            os.utime(old_log, (1, 1))
            process = FakeProcess([None, 0])
            expected_environment = build_environment(
                {}, spec.method_env, spec.gpu_index, spec.python
            )

            def factory(command, **kwargs):
                kwargs["stdout"].write(
                    "Set random seed to 41, deterministic: True\n"
                    "Epoch [1][10/10] loss: 0.1\n"
                )
                kwargs["stdout"].flush()
                return process

            outcome = run_experiment(
                spec,
                popen_factory=factory,
                environment_reader=lambda pid: expected_environment,
                compute_apps_probe=lambda: [
                    ComputeApp(process.pid, spec.gpu_uuid, 6400)
                ],
                snapshotter=lambda *_: None,
                sleeper=lambda _: None,
            )
            self.assertEqual(outcome.failure_kind, "missing_final_ema_eval")
            self.assertIsNone(outcome.final_map)

    def test_completion_requires_epoch_one_and_val_iteration_equal_total(self):
        cases = [
            ("Epoch(val) [1][9] mAP: 0.5\n", "incomplete_training"),
            ("Epoch(val) [2][10] mAP: 0.5\n", "unexpected_val_epoch"),
        ]
        for val_line, expected_failure in cases:
            with self.subTest(expected_failure=expected_failure):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    spec = self.make_spec(root)
                    process = FakeProcess([None, 0])
                    expected_environment = build_environment(
                        {}, spec.method_env, spec.gpu_index, spec.python
                    )

                    def factory(command, **kwargs):
                        kwargs["stdout"].write(
                            "Set random seed to 41, deterministic: True\n"
                            "Epoch [1][9/10] loss: 0.1\n"
                        )
                        kwargs["stdout"].flush()
                        (spec.work_dir / "20990101_000000.log").write_text(
                            "Set random seed to 41, deterministic: True\n" + val_line,
                            encoding="utf-8",
                        )
                        return process

                    outcome = run_experiment(
                        spec,
                        popen_factory=factory,
                        environment_reader=lambda pid: expected_environment,
                        compute_apps_probe=lambda: [
                            ComputeApp(process.pid, spec.gpu_uuid, 6400)
                        ],
                        snapshotter=lambda *_: None,
                        sleeper=lambda _: None,
                    )
                    self.assertEqual(outcome.failure_kind, expected_failure)


if __name__ == "__main__":
    unittest.main()
