from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tools.cga_research import gpu_scheduler as scheduler
from tools.cga_research.gpu_scheduler import (
    GPUInfo,
    GPULock,
    MethodSpec,
    Registry,
    SchedulerConfig,
    eligible_gpu,
    parse_gpu_inventory_csv,
    randomized_method_order,
    run_scheduler,
    run_seed_block,
)
from tools.cga_research.run_experiment import RunOutcome


def make_gpu(index: int) -> GPUInfo:
    return GPUInfo(
        index=index,
        uuid=f"GPU-uuid-{index}",
        name=f"GPU {index}",
        memory_used_mib=100,
        memory_free_mib=20000,
        memory_total_mib=24000,
        utilization_percent=5,
    )


def make_config(root: Path, seeds=None, max_workers=4) -> SchedulerConfig:
    return SchedulerConfig(
        project_root=root,
        research_root=root / "research",
        python="/iraod/python",
        config=root / "config.py",
        seeds=list(seeds or [41]),
        common_cfg_options=["corrupt=chaff", "model.cfg.score_thr=0.9"],
        run_id="test-run",
        max_workers=max_workers,
        max_attempts=3,
        max_gpu_hours=24.0,
        max_full_runs=20,
        gpu_poll_interval_seconds=0.001,
        max_gpu_wait_seconds=0.1,
        gpu_release_poll_seconds=0.001,
        gpu_release_timeout_seconds=0.01,
        monitor_interval_seconds=0.001,
    )


class SchedulerTests(unittest.TestCase):
    @staticmethod
    def completion_payload(
        seed: int,
        method: str,
        fingerprint: str,
        gpu_uuid: str,
        final_map: float = 0.5,
    ):
        return {
            "status": "completed",
            "success": True,
            "seed": seed,
            "actual_seed": seed,
            "method": method,
            "final_map": final_map,
            "experiment_fingerprint": fingerprint,
            "gpu_uuid": gpu_uuid,
            "deterministic": True,
            "gpu_seen": True,
            "full_run": True,
            "exit_code": 0,
            "failure_kind": None,
            "progress": {"epoch": 1, "iteration": 9, "total": 10},
            "final_val_epoch": 1,
            "final_val_iteration": 10,
        }

    def test_inventory_parser_and_query_use_argument_list(self):
        text = '0, GPU-a, "Model, A", 100, 9000, 10000, 29\n'
        gpu = parse_gpu_inventory_csv(text)[0]
        self.assertEqual(gpu.name, "Model, A")
        self.assertEqual(gpu.memory_free_mib, 9000)

        completed = mock.Mock(returncode=0, stdout=text, stderr="")
        with mock.patch.object(
            scheduler.subprocess, "run", return_value=completed
        ) as run:
            scheduler.query_gpu_inventory()
        argv = run.call_args.args[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[0], "nvidia-smi")
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_gpu_eligibility_boundaries(self):
        gpu = make_gpu(0)
        self.assertTrue(eligible_gpu(gpu, 20000, 30, {gpu.uuid}))
        low_memory = scheduler.dataclasses.replace(gpu, memory_free_mib=19999)
        self.assertFalse(eligible_gpu(low_memory, 20000, 30, set()))
        high_util = scheduler.dataclasses.replace(gpu, utilization_percent=30)
        self.assertFalse(eligible_gpu(high_util, 20000, 30, {gpu.uuid}))
        self.assertTrue(eligible_gpu(high_util, 20000, 30, set()))

    def test_method_order_is_reproducible_per_seed(self):
        names = ["a", "b", "c", "d", "e"]
        first = randomized_method_order(names, seed=41, order_seed=7)
        second = randomized_method_order(names, seed=41, order_seed=7)
        self.assertEqual(first, second)
        self.assertCountEqual(first, names)
        self.assertNotEqual(first, randomized_method_order(names, 42, 7))

    def test_gpu_lock_excludes_second_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = GPULock(root, "GPU-a")
            second = GPULock(root, "GPU-a")
            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())
            first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_registry_reconciles_stale_and_valid_completed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry(root / "registry.csv", "run")
            stale_dir = root / "stale"
            row = {
                "row_id": "41:a:1",
                "run_id": "run",
                "seed": 41,
                "requested_seed": 41,
                "method": "a",
                "status": "running",
                "attempt": 1,
                "work_dir": stale_dir,
                "pid": 99999,
                "started_at": time.time() - 10,
            }
            registry.try_reserve(row, 24.0, 20)
            registry.reconcile(process_alive=lambda pid: False)
            stale_row = registry.rows()[0]
            self.assertEqual(stale_row["failure_kind"], "stale_process")
            self.assertGreater(float(stale_row["gpu_seconds"]), 0.0)

            good_dir = root / "good"
            good_dir.mkdir()
            payload = self.completion_payload(42, "b", "b" * 64, "GPU-good")
            (good_dir / "run_result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            registry.try_reserve(
                {
                    "row_id": "42:b:1",
                    "run_id": "run",
                    "seed": 42,
                    "requested_seed": 42,
                    "method": "b",
                    "status": "running",
                    "attempt": 1,
                    "work_dir": good_dir,
                    "pid": 88888,
                    "gpu_uuid": "GPU-good",
                    "experiment_fingerprint": "b" * 64,
                    "started_at": time.time() - 1,
                },
                24.0,
                20,
            )
            registry.reconcile(process_alive=lambda pid: False)
            rows = {row["row_id"]: row for row in registry.rows()}
            self.assertEqual(rows["42:b:1"]["status"], "completed")

    def test_malformed_completion_sentinel_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            (work_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "success": True,
                        "seed": 41,
                        "actual_seed": None,
                        "method": "a",
                        "final_map": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(scheduler.valid_completed_result(work_dir, 41, "a"))

    def test_active_reservation_enforces_full_run_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory) / "registry.csv", "run")
            row = {
                "row_id": "1:a:1",
                "seed": 1,
                "method": "a",
                "status": "starting",
                "attempt": 1,
                "started_at": time.time(),
            }
            reserved, reason = registry.try_reserve(row, 24.0, 1)
            self.assertIsNotNone(reserved)
            second = dict(row, row_id="2:a:1", seed=2)
            reserved, reason = registry.try_reserve(second, 24.0, 1)
            self.assertIsNone(reserved)
            self.assertEqual(reason, "full_run_budget")

    def test_seed_block_retries_and_keeps_one_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            registry = Registry(config.registry_path, config.run_id)
            methods = [
                MethodSpec("a", {"CGA_SCORER": "none"}),
                MethodSpec("b", {"CGA_SCORER": "none"}),
            ]
            gpu = make_gpu(3)
            calls = []

            def fake_run(spec, **kwargs):
                calls.append(spec)
                success = len(calls) != 1
                return RunOutcome(
                    status="completed" if success else "failed",
                    success=success,
                    method=spec.method,
                    seed=spec.seed,
                    gpu_index=spec.gpu_index,
                    gpu_uuid=spec.gpu_uuid,
                    work_dir=str(spec.work_dir),
                    command=["python", "train.py"],
                    environment={"CUDA_VISIBLE_DEVICES": str(spec.gpu_index)},
                    failure_kind=None if success else "nonzero_exit",
                    exit_code=0 if success else 1,
                    actual_seed=spec.seed if success else None,
                    final_map=0.5 if success else None,
                )

            outcomes = run_seed_block(
                config,
                41,
                gpu,
                methods,
                registry,
                run_one=fake_run,
                compute_apps_probe=lambda: [],
                sleeper=lambda _: None,
            )
            self.assertEqual(len(outcomes), 3)
            self.assertTrue(all(spec.gpu_uuid == gpu.uuid for spec in calls))
            self.assertTrue(all(spec.gpu_index == gpu.index for spec in calls))
            self.assertEqual({spec.method for spec in calls}, {"a", "b"})

    def test_resume_skips_valid_completed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            method = MethodSpec("a", {"CGA_SCORER": "none"})
            registry = Registry(config.registry_path, config.run_id)
            seed_dir = scheduler._seed_method_dir(config, "a", 41)
            attempt_dir = seed_dir / "attempt_1_gpu_previous"
            attempt_dir.mkdir(parents=True)
            fingerprint = scheduler._method_fingerprint(config, 41, method, make_gpu(0))
            payload = self.completion_payload(
                41, "a", fingerprint, "GPU-uuid-0", final_map=0.55
            )
            payload["gpu_index"] = 0
            (attempt_dir / "run_result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )

            outcomes = run_seed_block(
                config,
                41,
                make_gpu(0),
                [method],
                registry,
                run_one=mock.Mock(side_effect=AssertionError("must skip")),
            )
            self.assertEqual(outcomes, [])
            self.assertTrue(registry.completed(41, "a"))

    def test_orphan_attempt_directory_is_not_reused_or_free_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            method = MethodSpec("a", {"CGA_SCORER": "none"})
            registry = Registry(config.registry_path, config.run_id)
            seed_dir = scheduler._seed_method_dir(config, "a", 41)
            orphan = seed_dir / "attempt_1_fp_old_gpu_uuid0"
            orphan.mkdir(parents=True)
            (orphan / "run_result.json").write_text("{}\n", encoding="utf-8")
            calls = []

            def fail(spec, **kwargs):
                calls.append(spec.work_dir.name)
                return RunOutcome(
                    status="failed",
                    success=False,
                    method=spec.method,
                    seed=spec.seed,
                    gpu_index=spec.gpu_index,
                    gpu_uuid=spec.gpu_uuid,
                    work_dir=str(spec.work_dir),
                    command=[],
                    environment={},
                    failure_kind="nonzero_exit",
                    exit_code=1,
                )

            outcomes = run_seed_block(
                config,
                41,
                make_gpu(0),
                [method],
                registry,
                run_one=fail,
            )
            self.assertEqual(len(outcomes), 2)
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0].startswith("attempt_2_"))
            self.assertTrue(calls[1].startswith("attempt_3_"))

    def test_completed_result_requires_matching_fingerprint_and_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            payload = self.completion_payload(41, "a", "a" * 64, "GPU-a")
            (work_dir / "run_result.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            self.assertTrue(
                scheduler.valid_completed_result(
                    work_dir,
                    41,
                    "a",
                    expected_fingerprint="a" * 64,
                    expected_gpu_uuid="GPU-a",
                )
            )
            self.assertFalse(
                scheduler.valid_completed_result(
                    work_dir,
                    41,
                    "a",
                    expected_fingerprint="b" * 64,
                    expected_gpu_uuid="GPU-a",
                )
            )
            self.assertFalse(
                scheduler.valid_completed_result(
                    work_dir,
                    41,
                    "a",
                    expected_fingerprint="a" * 64,
                    expected_gpu_uuid="GPU-b",
                )
            )

    def test_gpu_release_timeout_stops_block_and_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            registry = Registry(config.registry_path, config.run_id)
            methods = [
                MethodSpec("a", {"CGA_SCORER": "none"}),
                MethodSpec("b", {"CGA_SCORER": "none"}),
            ]
            calls = []

            def fake_run(spec, **kwargs):
                calls.append(spec.method)
                (spec.work_dir / "run_result.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "success": True,
                            "seed": spec.seed,
                            "actual_seed": spec.seed,
                            "method": spec.method,
                            "final_map": 0.5,
                            "experiment_fingerprint": scheduler.compute_experiment_fingerprint(
                                spec
                            )[
                                "sha256"
                            ],
                            "gpu_uuid": spec.gpu_uuid,
                        }
                    ),
                    encoding="utf-8",
                )
                return RunOutcome(
                    status="completed",
                    success=True,
                    method=spec.method,
                    seed=spec.seed,
                    gpu_index=spec.gpu_index,
                    gpu_uuid=spec.gpu_uuid,
                    work_dir=str(spec.work_dir),
                    command=[],
                    environment={},
                    pid=123,
                    actual_seed=spec.seed,
                    final_map=0.5,
                )

            run_seed_block(
                config,
                41,
                make_gpu(0),
                methods,
                registry,
                run_one=fake_run,
                wait_release=lambda *args, **kwargs: False,
            )
            self.assertEqual(len(calls), 1)
            row = registry.rows()[-1]
            self.assertEqual(row["status"], "partial")
            self.assertEqual(row["failure_kind"], "gpu_release_timeout")
            self.assertEqual(registry.active_gpu_uuids(), {"GPU-uuid-0"})

    def test_external_running_uuid_is_reserved_and_dead_external_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry(root / "registry.csv", "run")
            registry.try_reserve(
                {
                    "row_id": "41:a:1",
                    "run_id": "run",
                    "seed": 41,
                    "method": "a",
                    "status": "external_running",
                    "attempt": 1,
                    "work_dir": root / "work",
                    "pid": 999,
                    "gpu_uuid": "GPU-reserved",
                    "started_at": time.time() - 10,
                    "experiment_fingerprint": "a" * 64,
                },
                24.0,
                20,
            )
            self.assertEqual(registry.active_gpu_uuids(), {"GPU-reserved"})
            registry.reconcile(process_alive=lambda pid: False)
            row = registry.rows()[0]
            self.assertEqual(row["status"], "failed_terminal")
            self.assertEqual(row["failure_kind"], "external_result_unverified")

    def test_scheduler_probe_failure_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            config.max_gpu_wait_seconds = 0.0
            methods = [MethodSpec("a", {"CGA_SCORER": "none"})]
            with mock.patch.object(scheduler, "_run_locked_seed_block") as run_block:
                result = run_scheduler(
                    config,
                    methods,
                    inventory_probe=lambda: [make_gpu(0)],
                    compute_apps_probe=mock.Mock(
                        side_effect=RuntimeError("compute probe failed")
                    ),
                    sleeper=lambda _: None,
                )
            run_block.assert_not_called()
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["blocked_reason"], "compute_app_probe_failed")

    def test_future_exception_is_persisted_and_returns_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            config.max_gpu_wait_seconds = 0.0
            methods = [MethodSpec("a", {"CGA_SCORER": "none"})]
            inventory_calls = 0

            def inventory():
                nonlocal inventory_calls
                inventory_calls += 1
                return [make_gpu(0)] if inventory_calls == 1 else []

            with mock.patch.object(
                scheduler,
                "_run_locked_seed_block",
                side_effect=RuntimeError("future exploded"),
            ):
                result = run_scheduler(
                    config,
                    methods,
                    inventory_probe=inventory,
                    compute_apps_probe=lambda: [],
                    sleeper=lambda _: None,
                )
            registry = Registry(config.registry_path, config.run_id)
            failures = [
                row
                for row in registry.rows()
                if row["failure_kind"] == "seed_block_future_exception"
            ]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["status"], "failed_terminal")
            self.assertEqual(result["status"], "partial")

    def test_retry_exhaustion_returns_partial_not_finished(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            methods = [MethodSpec("a", {"CGA_SCORER": "none"})]

            def exhausted_block(lock, cfg, seed, gpu, method_specs, registry):
                try:
                    for attempt in range(1, 4):
                        row_id = f"{seed}:a:{attempt}"
                        registry.try_reserve(
                            {
                                "row_id": row_id,
                                "run_id": cfg.run_id,
                                "seed": seed,
                                "method": "a",
                                "status": "failed",
                                "attempt": attempt,
                                "work_dir": root / f"attempt_{attempt}",
                                "gpu_uuid": gpu.uuid,
                                "started_at": time.time(),
                                "failure_kind": "nonzero_exit",
                            },
                            24.0,
                            20,
                        )
                    return []
                finally:
                    lock.release()

            with mock.patch.object(
                scheduler, "_run_locked_seed_block", side_effect=exhausted_block
            ):
                result = run_scheduler(
                    config,
                    methods,
                    inventory_probe=lambda: [make_gpu(0)],
                    compute_apps_probe=lambda: [],
                    sleeper=lambda _: None,
                )
            self.assertEqual(result["status"], "partial")
            self.assertGreater(result["failed_jobs"], 0)

    def test_required_memory_has_hard_floor_and_smoke_peak_is_required(self):
        self.assertEqual(scheduler.required_free_memory_mib(1, None), 8192)
        self.assertEqual(scheduler.required_free_memory_mib(8192, 7000), 8536)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(scheduler, "run_scheduler") as run:
                with self.assertRaises(SystemExit):
                    scheduler.main(
                        [
                            "--project-root",
                            str(root),
                            "--research-root",
                            str(root / "research"),
                            "--python",
                            "/python",
                            "--seed",
                            "41",
                            "--method",
                            "no_cga",
                        ]
                    )
            run.assert_not_called()

    def test_method_specs_support_seed_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "methods.json"
            path.write_text(
                json.dumps(
                    {
                        "methods": [
                            {
                                "name": "causal_control",
                                "environment": {"CGA_SCORER": "none"},
                                "seeds": [41, 43],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            methods = scheduler.load_method_specs(
                path, ["causal_control"], root / "unused-lora.pth"
            )
            self.assertEqual(methods[0].seeds, (41, 43))
            self.assertTrue(methods[0].applies_to(41))
            self.assertFalse(methods[0].applies_to(42))

    def test_seed_block_runs_only_applicable_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            registry = Registry(config.registry_path, config.run_id)
            methods = [
                MethodSpec("seed42_only", {"CGA_SCORER": "none"}, seeds=(42,)),
                MethodSpec("seed41_only", {"CGA_SCORER": "none"}, seeds=(41,)),
            ]
            calls = []

            def fake_run(spec, **kwargs):
                calls.append(spec.method)
                return RunOutcome(
                    status="completed",
                    success=True,
                    method=spec.method,
                    seed=spec.seed,
                    gpu_index=spec.gpu_index,
                    gpu_uuid=spec.gpu_uuid,
                    work_dir=str(spec.work_dir),
                    command=[],
                    environment={},
                    actual_seed=spec.seed,
                    final_map=0.5,
                )

            run_seed_block(
                config,
                41,
                make_gpu(0),
                methods,
                registry,
                run_one=fake_run,
            )
            self.assertEqual(calls, ["seed41_only"])

    def test_no_applicable_method_for_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, seeds=[42])
            methods = [MethodSpec("seed41_only", {"CGA_SCORER": "none"}, seeds=(41,))]
            with self.assertRaisesRegex(
                ValueError, "no applicable methods for seed 42"
            ):
                run_scheduler(config, methods)
            with self.assertRaisesRegex(
                ValueError, "no applicable methods for seed 42"
            ):
                scheduler.build_dry_run_plan(config, methods, 0, "GPU-dry")

    def test_dry_run_exposes_seed_allowlist_and_filters_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, seeds=[41, 42])
            methods = [
                MethodSpec("baseline", {"CGA_SCORER": "none"}),
                MethodSpec("causal", {"CGA_SCORER": "none"}, seeds=(41,)),
            ]
            plan = scheduler.build_dry_run_plan(config, methods, 0, "GPU-dry")
            jobs = {(job["seed"], job["method"]): job for job in plan["jobs"]}
            self.assertEqual(
                set(jobs), {(41, "baseline"), (41, "causal"), (42, "baseline")}
            )
            self.assertEqual(jobs[(41, "causal")]["seed_allowlist"], [41])
            self.assertIsNone(jobs[(42, "baseline")]["seed_allowlist"])
            self.assertIn("experiment_fingerprint", jobs[(41, "causal")])

    def test_existing_seed_gpu_uuid_is_a_hard_assignment_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            registry = Registry(config.registry_path, config.run_id)
            registry.try_reserve(
                {
                    "row_id": "41:prior:1",
                    "run_id": config.run_id,
                    "seed": 41,
                    "method": "prior",
                    "status": "failed",
                    "attempt": 1,
                    "gpu_uuid": "GPU-uuid-0",
                },
                24.0,
                20,
            )
            selected = []
            finished = set()

            def fake_locked(lock, cfg, seed, gpu, method_specs, current_registry):
                try:
                    selected.append(gpu.uuid)
                    finished.add(seed)
                    return []
                finally:
                    lock.release()

            with mock.patch.object(
                scheduler, "_run_locked_seed_block", side_effect=fake_locked
            ), mock.patch.object(
                scheduler,
                "_seed_terminal",
                side_effect=lambda cfg, reg, seed, specs: seed in finished,
            ):
                result = run_scheduler(
                    config,
                    [MethodSpec("a", {"CGA_SCORER": "none"})],
                    inventory_probe=lambda: [
                        make_gpu(1),
                        scheduler.dataclasses.replace(
                            make_gpu(0), memory_free_mib=10000
                        ),
                    ],
                    compute_apps_probe=lambda: [],
                    sleeper=lambda _: None,
                )
            self.assertEqual(selected, ["GPU-uuid-0"])
            self.assertEqual(result["status"], "finished")

    def test_conflicting_registry_gpu_uuids_for_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            registry = Registry(config.registry_path, config.run_id)
            for attempt, gpu_uuid in enumerate(("GPU-a", "GPU-b"), start=1):
                registry.try_reserve(
                    {
                        "row_id": f"41:a:{attempt}",
                        "run_id": config.run_id,
                        "seed": 41,
                        "method": "a",
                        "status": "failed",
                        "attempt": attempt,
                        "gpu_uuid": gpu_uuid,
                    },
                    24.0,
                    20,
                )
            with self.assertRaisesRegex(ValueError, "conflicting registry GPU UUIDs"):
                run_scheduler(
                    config,
                    [MethodSpec("a", {"CGA_SCORER": "none"})],
                )

    def test_scheduler_runs_different_seeds_in_parallel_and_caps_at_four(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, seeds=list(range(8)), max_workers=10)
            methods = [MethodSpec("a", {"CGA_SCORER": "none"})]
            gpus = [make_gpu(index) for index in range(4)]
            finished = set()
            state_lock = threading.Lock()
            active = 0
            max_active = 0

            def fake_locked(lock, cfg, seed, gpu, method_specs, registry):
                nonlocal active, max_active
                try:
                    with state_lock:
                        active += 1
                        max_active = max(max_active, active)
                    time.sleep(0.02)
                    with state_lock:
                        finished.add(seed)
                        active -= 1
                    return []
                finally:
                    lock.release()

            def fake_terminal(cfg, registry, seed, method_specs):
                with state_lock:
                    return seed in finished

            with mock.patch.object(
                scheduler, "_run_locked_seed_block", side_effect=fake_locked
            ), mock.patch.object(
                scheduler, "_seed_terminal", side_effect=fake_terminal
            ):
                result = run_scheduler(
                    config,
                    methods,
                    inventory_probe=lambda: gpus,
                    compute_apps_probe=lambda: [],
                    sleeper=lambda _: None,
                )
            self.assertEqual(result["status"], "finished")
            self.assertGreater(max_active, 1)
            self.assertLessEqual(max_active, 4)
            self.assertEqual(finished, set(config.seeds))


if __name__ == "__main__":
    unittest.main()
