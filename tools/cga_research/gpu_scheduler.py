#!/usr/bin/env python3
"""Schedule reproducible IRAOD seed blocks across single-GPU workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import fcntl
import hashlib
import io
import json
import math
import os
import random
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

try:
    from .run_experiment import (
        ComputeApp,
        ExperimentSpec,
        RunOutcome,
        build_environment,
        build_train_command,
        compute_experiment_fingerprint,
        query_compute_apps,
        relevant_environment,
        run_experiment,
        validate_method_environment,
    )
except ImportError:  # Support direct execution as a script.
    from run_experiment import (  # type: ignore
        ComputeApp,
        ExperimentSpec,
        RunOutcome,
        build_environment,
        build_train_command,
        compute_experiment_fingerprint,
        query_compute_apps,
        relevant_environment,
        run_experiment,
        validate_method_environment,
    )


GPU_QUERY = "index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu"
METHOD_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
ACTIVE_STATUSES = {"starting", "running", "validating", "external_running"}
TERMINAL_FAILURE_STATUSES = {"failed_terminal", "partial"}
GPU_QUARANTINE_FAILURES = {
    "gpu_release_timeout",
    "external_result_unverified",
    "seed_block_future_exception",
}

REGISTRY_FIELDS = [
    "row_id",
    "run_id",
    "seed",
    "requested_seed",
    "actual_seed",
    "method",
    "method_order",
    "experiment_fingerprint",
    "status",
    "attempt",
    "work_dir",
    "pid",
    "exit_code",
    "failure_kind",
    "failure_detail",
    "physical_gpu_index",
    "gpu_uuid",
    "gpu_name",
    "free_memory_mib_at_start",
    "started_at",
    "ended_at",
    "gpu_seconds",
    "full_run",
    "final_map",
    "config_path",
    "cfg_options_json",
    "environment_json",
    "command_json",
    "result_json_path",
]


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    index: int
    uuid: str
    name: str
    memory_used_mib: int
    memory_free_mib: int
    memory_total_mib: int
    utilization_percent: int


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    name: str
    environment: dict[str, str]
    cfg_options: tuple[str, ...] = ()
    seeds: Optional[tuple[int, ...]] = None

    def applies_to(self, seed: int) -> bool:
        return self.seeds is None or seed in self.seeds


@dataclasses.dataclass
class SchedulerConfig:
    project_root: Path
    research_root: Path
    python: str
    config: Path
    seeds: list[int]
    common_cfg_options: list[str]
    run_id: str
    required_free_mib: int = 8192
    utilization_limit: int = 30
    max_workers: int = 4
    max_attempts: int = 3
    max_gpu_hours: float = 24.0
    max_full_runs: int = 20
    gpu_poll_interval_seconds: float = 60.0
    max_gpu_wait_seconds: float = 7200.0
    gpu_release_poll_seconds: float = 5.0
    gpu_release_timeout_seconds: float = 180.0
    monitor_interval_seconds: float = 60.0
    stall_timeout_seconds: float = 900.0
    gpu_verify_timeout_seconds: float = 300.0
    terminate_grace_seconds: float = 30.0
    order_seed: int = 20260714

    def __post_init__(self) -> None:
        self.required_free_mib = max(8192, int(self.required_free_mib))

    @property
    def registry_path(self) -> Path:
        return self.research_root / "gpu_job_registry.csv"

    @property
    def lock_root(self) -> Path:
        return self.research_root / "gpu_locks"


def required_free_memory_mib(requested_mib: int, smoke_peak_mib: Optional[int]) -> int:
    candidates = [8192, int(requested_mib)]
    if smoke_peak_mib is not None:
        candidates.append(int(smoke_peak_mib) + 1536)
    return max(candidates)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_gpu_inventory_csv(text: str) -> list[GPUInfo]:
    reader = csv.reader(io.StringIO(text), skipinitialspace=True)
    gpus: list[GPUInfo] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for line_number, columns in enumerate(reader, start=1):
        if not columns or all(not column.strip() for column in columns):
            continue
        if len(columns) != 7:
            raise ValueError(
                f"malformed GPU row {line_number}: expected 7 columns, got "
                f"{len(columns)}"
            )
        values = [column.strip() for column in columns]
        gpu = GPUInfo(
            index=int(values[0]),
            uuid=values[1],
            name=values[2],
            memory_used_mib=int(values[3]),
            memory_free_mib=int(values[4]),
            memory_total_mib=int(values[5]),
            utilization_percent=int(values[6]),
        )
        if gpu.index in seen_indices or gpu.uuid in seen_uuids:
            raise ValueError(f"duplicate GPU in inventory: {gpu}")
        seen_indices.add(gpu.index)
        seen_uuids.add(gpu.uuid)
        gpus.append(gpu)
    return gpus


def query_gpu_inventory() -> list[GPUInfo]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    return parse_gpu_inventory_csv(completed.stdout)


def eligible_gpu(
    gpu: GPUInfo,
    required_free_mib: int,
    utilization_limit: int,
    busy_gpu_uuids: set[str],
) -> bool:
    """Apply the requested free-memory and utilization/compute-app rule."""

    if gpu.memory_free_mib < required_free_mib:
        return False
    low_utilization = gpu.utilization_percent < utilization_limit
    no_compute_app = gpu.uuid not in busy_gpu_uuids
    return low_utilization or no_compute_app


def randomized_method_order(
    method_names: Sequence[str], seed: int, order_seed: int
) -> list[str]:
    material = f"{order_seed}:{seed}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    ordered = list(method_names)
    random.Random(stable_seed).shuffle(ordered)
    return ordered


class GPULock:
    """A non-blocking advisory lock tied to a GPU UUID."""

    def __init__(self, lock_root: Path, gpu_uuid: str) -> None:
        safe_uuid = re.sub(r"[^A-Za-z0-9_.-]+", "_", gpu_uuid)
        self.path = Path(lock_root) / f"{safe_uuid}.lock"
        self.gpu_uuid = gpu_uuid
        self._handle: Optional[Any] = None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"pid": os.getpid(), "gpu_uuid": self.gpu_uuid, "time": time.time()},
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "GPULock":
        if not self.acquire():
            raise BlockingIOError(f"GPU lock is busy: {self.gpu_uuid}")
        return self

    def __exit__(self, *_: Any) -> None:
        self.release()


def _as_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_true(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes"}


def valid_completed_result(
    work_dir: Path,
    seed: int,
    method: str,
    *,
    expected_fingerprint: Optional[str] = None,
    expected_gpu_uuid: Optional[str] = None,
) -> bool:
    result_path = Path(work_dir) / "run_result.json"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    try:
        payload_seed = int(payload.get("seed"))
        actual_seed = int(payload.get("actual_seed"))
    except (TypeError, ValueError):
        return False
    fingerprint = payload.get("experiment_fingerprint")
    gpu_uuid = payload.get("gpu_uuid")
    progress = payload.get("progress")
    try:
        final_map = float(payload.get("final_map"))
        progress_epoch = int(progress.get("epoch"))
        progress_iteration = int(progress.get("iteration"))
        progress_total = int(progress.get("total"))
        final_val_epoch = int(payload.get("final_val_epoch"))
        final_val_iteration = int(payload.get("final_val_iteration"))
    except (AttributeError, TypeError, ValueError):
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("success") is True
        and payload_seed == int(seed)
        and actual_seed == int(seed)
        and payload.get("method") == method
        and isinstance(fingerprint, str)
        and re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None
        and (expected_fingerprint is None or fingerprint == expected_fingerprint)
        and isinstance(gpu_uuid, str)
        and bool(gpu_uuid)
        and (expected_gpu_uuid is None or gpu_uuid == expected_gpu_uuid)
        and math.isfinite(final_map)
        and payload.get("deterministic") is True
        and payload.get("gpu_seen") is True
        and payload.get("full_run") is True
        and payload.get("exit_code") == 0
        and payload.get("failure_kind") is None
        and progress_epoch == 1
        and 0 < progress_iteration <= progress_total
        and progress_total > 0
        and final_val_epoch == 1
        and final_val_iteration == progress_total
    )


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Registry:
    """Thread-safe, atomically rewritten CSV registry."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._lock = threading.RLock()
        self._rows: list[dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                self._rows.append(
                    {field: str(row.get(field, "") or "") for field in REGISTRY_FIELDS}
                )

    def _write_locked(self) -> None:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(self._rows)
        _atomic_write_text(self.path, buffer.getvalue())

    def rows(self) -> list[dict[str, str]]:
        with self._lock:
            return [row.copy() for row in self._rows]

    def _find_locked(self, row_id: str) -> dict[str, str]:
        for row in self._rows:
            if row["row_id"] == row_id:
                return row
        raise KeyError(row_id)

    def update(self, row_id: str, **changes: Any) -> None:
        with self._lock:
            row = self._find_locked(row_id)
            for key, value in changes.items():
                if key not in REGISTRY_FIELDS:
                    raise KeyError(key)
                if isinstance(value, bool):
                    row[key] = "1" if value else "0"
                elif value is None:
                    row[key] = ""
                else:
                    row[key] = str(value)
            self._write_locked()

    def _budget_usage_locked(self, now: float) -> tuple[float, int, int]:
        gpu_seconds = 0.0
        full_runs = 0
        active_reservations = 0
        for row in self._rows:
            status = row["status"]
            if status in ACTIVE_STATUSES:
                started_at = _as_float(row["started_at"])
                if started_at > 0:
                    gpu_seconds += max(0.0, now - started_at)
                active_reservations += 1
            else:
                gpu_seconds += max(0.0, _as_float(row["gpu_seconds"]))
            if _is_true(row["full_run"]):
                full_runs += 1
        return gpu_seconds, full_runs, active_reservations

    def budget_usage(self, now: Optional[float] = None) -> dict[str, float]:
        with self._lock:
            gpu_seconds, full_runs, active = self._budget_usage_locked(
                time.time() if now is None else now
            )
        return {
            "gpu_seconds": gpu_seconds,
            "gpu_hours": gpu_seconds / 3600.0,
            "full_runs": float(full_runs),
            "active_reservations": float(active),
        }

    def try_reserve(
        self,
        row: Mapping[str, Any],
        max_gpu_hours: float,
        max_full_runs: int,
        now: Optional[float] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        with self._lock:
            current_time = time.time() if now is None else now
            gpu_seconds, full_runs, active = self._budget_usage_locked(current_time)
            if gpu_seconds >= max_gpu_hours * 3600.0:
                return None, "gpu_hour_budget"
            if full_runs + active >= max_full_runs:
                return None, "full_run_budget"
            row_id = str(row["row_id"])
            if any(existing["row_id"] == row_id for existing in self._rows):
                raise ValueError(f"duplicate registry row: {row_id}")
            normalized = {
                field: (
                    "1"
                    if row.get(field) is True
                    else "0"
                    if row.get(field) is False
                    else str(row.get(field, "") or "")
                )
                for field in REGISTRY_FIELDS
            }
            self._rows.append(normalized)
            self._write_locked()
            return row_id, None

    def gpu_budget_stop_reason(self, max_gpu_hours: float) -> Optional[str]:
        usage = self.budget_usage()
        if usage["gpu_hours"] >= max_gpu_hours:
            return (
                f"aggregate GPU time {usage['gpu_hours']:.6f}h reached "
                f"{max_gpu_hours:.6f}h"
            )
        return None

    def attempt_count(self, seed: int, method: str) -> int:
        with self._lock:
            return sum(
                1
                for row in self._rows
                if _as_int(row["seed"], -1) == seed and row["method"] == method
            )

    def max_attempt_number(self, seed: int, method: str) -> int:
        with self._lock:
            return max(
                (
                    _as_int(row["attempt"])
                    for row in self._rows
                    if _as_int(row["seed"], -1) == seed and row["method"] == method
                ),
                default=0,
            )

    def completed(
        self,
        seed: int,
        method: str,
        *,
        expected_fingerprint: Optional[str] = None,
        expected_gpu_uuid: Optional[str] = None,
    ) -> bool:
        with self._lock:
            candidates = [
                row.copy()
                for row in self._rows
                if _as_int(row["seed"], -1) == seed
                and row["method"] == method
                and row["status"] == "completed"
            ]
        return any(
            (
                expected_fingerprint is None
                or row["experiment_fingerprint"] == expected_fingerprint
            )
            and (expected_gpu_uuid is None or row["gpu_uuid"] == expected_gpu_uuid)
            and valid_completed_result(
                Path(row["work_dir"]),
                seed,
                method,
                expected_fingerprint=(
                    expected_fingerprint or row["experiment_fingerprint"]
                ),
                expected_gpu_uuid=(expected_gpu_uuid or row["gpu_uuid"]),
            )
            for row in candidates
        )

    def active_gpu_uuids(self) -> set[str]:
        with self._lock:
            return {
                row["gpu_uuid"]
                for row in self._rows
                if row["gpu_uuid"]
                and (
                    row["status"] in ACTIVE_STATUSES
                    or row["failure_kind"] in GPU_QUARANTINE_FAILURES
                )
            }

    def seed_gpu_uuids(self, seed: int) -> set[str]:
        with self._lock:
            return {
                row["gpu_uuid"]
                for row in self._rows
                if _as_int(row["seed"], -1) == seed and row["gpu_uuid"]
            }

    def method_has_terminal_failure(self, seed: int, method: str) -> bool:
        with self._lock:
            return any(
                _as_int(row["seed"], -1) == seed
                and row["method"] == method
                and row["status"] in TERMINAL_FAILURE_STATUSES
                for row in self._rows
            )

    def seed_has_terminal_failure(self, seed: int) -> bool:
        with self._lock:
            return any(
                _as_int(row["seed"], -1) == seed
                and row["status"] in TERMINAL_FAILURE_STATUSES
                for row in self._rows
            )

    def preferred_gpu_uuid(self, seed: int) -> Optional[str]:
        with self._lock:
            for row in self._rows:
                if _as_int(row["seed"], -1) == seed and row["gpu_uuid"]:
                    return row["gpu_uuid"]
        return None

    def seed_has_active_process(self, seed: int) -> bool:
        with self._lock:
            return any(
                _as_int(row["seed"], -1) == seed and row["status"] in ACTIVE_STATUSES
                for row in self._rows
            )

    def reconcile(self, process_alive: Callable[[int], bool] = pid_alive) -> None:
        changed = False
        with self._lock:
            for row in self._rows:
                status = row["status"]
                if status == "completed":
                    if not valid_completed_result(
                        Path(row["work_dir"]),
                        _as_int(row["seed"]),
                        row["method"],
                        expected_fingerprint=row["experiment_fingerprint"],
                        expected_gpu_uuid=row["gpu_uuid"],
                    ):
                        row["status"] = "failed"
                        row["failure_kind"] = "invalid_completion_sentinel"
                        changed = True
                    continue
                if status not in ACTIVE_STATUSES:
                    continue
                pid = _as_int(row["pid"])
                now = time.time()
                started_at = _as_float(row["started_at"])
                if (
                    status == "starting"
                    and pid <= 0
                    and started_at > 0
                    and now - started_at < 300.0
                ):
                    continue
                if process_alive(pid):
                    if status != "external_running":
                        row["status"] = "external_running"
                        changed = True
                    continue
                if valid_completed_result(
                    Path(row["work_dir"]),
                    _as_int(row["seed"]),
                    row["method"],
                    expected_fingerprint=row["experiment_fingerprint"],
                    expected_gpu_uuid=row["gpu_uuid"],
                ):
                    row["status"] = "completed"
                    payload = _load_result_payload(
                        Path(row["work_dir"]) / "run_result.json"
                    )
                    if payload is not None:
                        row["actual_seed"] = str(payload.get("actual_seed", ""))
                        row["pid"] = str(payload.get("pid", "") or "")
                        row["exit_code"] = str(payload.get("exit_code", ""))
                        row["failure_kind"] = str(payload.get("failure_kind", "") or "")
                        row["failure_detail"] = str(
                            payload.get("failure_detail", "") or ""
                        )
                        row["started_at"] = str(payload.get("started_at", "") or "")
                        row["ended_at"] = str(payload.get("ended_at", "") or "")
                        row["gpu_seconds"] = str(payload.get("gpu_seconds", 0.0))
                        row["full_run"] = "1" if payload.get("full_run") else "0"
                        row["final_map"] = str(payload.get("final_map", ""))
                        row["environment_json"] = json.dumps(
                            payload.get("environment", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        row["command_json"] = json.dumps(
                            payload.get("command", []), ensure_ascii=False
                        )
                        row["result_json_path"] = str(
                            Path(row["work_dir"]) / "run_result.json"
                        )
                else:
                    was_external = status == "external_running"
                    row["status"] = "failed_terminal" if was_external else "failed"
                    row["failure_kind"] = (
                        "external_result_unverified"
                        if was_external
                        else "stale_process"
                    )
                    row["ended_at"] = str(now)
                    if started_at > 0:
                        row["gpu_seconds"] = str(max(0.0, now - started_at))
                changed = True
            if changed:
                self._write_locked()

    def add_recovered(
        self,
        seed: int,
        method: str,
        work_dir: Path,
        payload: Mapping[str, Any],
    ) -> None:
        with self._lock:
            if any(
                row["work_dir"] == str(work_dir) and row["status"] == "completed"
                for row in self._rows
            ):
                return
            attempt = self.attempt_count(seed, method) + 1
            row_id = f"{seed}:{method}:recovered:{attempt}"
            row = {field: "" for field in REGISTRY_FIELDS}
            row.update(
                {
                    "row_id": row_id,
                    "run_id": self.run_id,
                    "seed": str(seed),
                    "requested_seed": str(seed),
                    "actual_seed": str(payload.get("actual_seed", seed)),
                    "method": method,
                    "experiment_fingerprint": str(
                        payload.get("experiment_fingerprint", "")
                    ),
                    "status": "completed",
                    "attempt": str(attempt),
                    "work_dir": str(work_dir),
                    "pid": str(payload.get("pid", "") or ""),
                    "exit_code": str(payload.get("exit_code", 0)),
                    "physical_gpu_index": str(payload.get("gpu_index", "")),
                    "gpu_uuid": str(payload.get("gpu_uuid", "")),
                    "started_at": str(payload.get("started_at", "") or ""),
                    "ended_at": str(payload.get("ended_at", "") or ""),
                    "gpu_seconds": str(payload.get("gpu_seconds", 0.0)),
                    "full_run": "1" if payload.get("full_run") else "0",
                    "final_map": str(payload.get("final_map", "")),
                    "environment_json": json.dumps(
                        payload.get("environment", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "command_json": json.dumps(
                        payload.get("command", []), ensure_ascii=False
                    ),
                    "result_json_path": str(work_dir / "run_result.json"),
                }
            )
            self._rows.append(row)
            self._write_locked()

    def add_terminal_failure(
        self,
        seed: int,
        method: str,
        *,
        gpu_uuid: str,
        failure_kind: str,
        failure_detail: str,
    ) -> None:
        with self._lock:
            if self.method_has_terminal_failure(seed, method):
                return
            now = time.time()
            attempt = self.attempt_count(seed, method) + 1
            row = {field: "" for field in REGISTRY_FIELDS}
            row.update(
                {
                    "row_id": f"{seed}:{method}:terminal:{time.time_ns()}",
                    "run_id": self.run_id,
                    "seed": str(seed),
                    "requested_seed": str(seed),
                    "method": method,
                    "status": "failed_terminal",
                    "attempt": str(attempt),
                    "gpu_uuid": gpu_uuid,
                    "failure_kind": failure_kind,
                    "failure_detail": failure_detail,
                    "started_at": str(now),
                    "ended_at": str(now),
                    "gpu_seconds": "0.0",
                    "full_run": "0",
                }
            )
            self._rows.append(row)
            self._write_locked()


def _load_result_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def recover_completed_attempt(
    registry: Registry,
    seed_dir: Path,
    seed: int,
    method: str,
    *,
    expected_fingerprint: str,
    expected_gpu_uuid: str,
) -> bool:
    for result_path in sorted(seed_dir.glob("attempt_*/run_result.json")):
        payload = _load_result_payload(result_path)
        if payload is None:
            continue
        if valid_completed_result(
            result_path.parent,
            seed,
            method,
            expected_fingerprint=expected_fingerprint,
            expected_gpu_uuid=expected_gpu_uuid,
        ):
            registry.add_recovered(seed, method, result_path.parent, payload)
            return True
    return False


def merge_cfg_options(*groups: Iterable[str]) -> list[str]:
    merged: dict[str, str] = {}
    order: list[str] = []
    for group in groups:
        for option in group:
            if "=" not in option or "\x00" in option:
                raise ValueError(f"invalid cfg option: {option!r}")
            key = option.split("=", 1)[0]
            if key not in merged:
                order.append(key)
            merged[key] = option
    return [merged[key] for key in order]


def wait_for_gpu_release(
    pid: int,
    compute_apps_probe: Callable[[], list[ComputeApp]],
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
    poll_seconds: float,
    timeout_seconds: float,
) -> bool:
    deadline = clock() + timeout_seconds
    while True:
        try:
            if all(app.pid != pid for app in compute_apps_probe()):
                return True
        except (OSError, RuntimeError, ValueError):
            pass
        if clock() >= deadline:
            return False
        sleeper(poll_seconds)


def _short_uuid(gpu_uuid: str) -> str:
    value = gpu_uuid.removeprefix("GPU-")
    return re.sub(r"[^A-Za-z0-9]", "", value)[:12] or "unknown"


def _seed_method_dir(config: SchedulerConfig, method: str, seed: int) -> Path:
    corruption = "unknown"
    for option in config.common_cfg_options:
        if option.startswith("corrupt="):
            corruption = option.split("=", 1)[1]
            break
    return config.research_root / "runs" / corruption / method / f"seed_{seed}"


def _max_disk_attempt(seed_dir: Path) -> int:
    maximum = 0
    for path in seed_dir.glob("attempt_*"):
        match = re.match(r"attempt_(\d+)(?:_|$)", path.name)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum


def _build_experiment_spec(
    config: SchedulerConfig,
    seed: int,
    gpu: GPUInfo,
    method: MethodSpec,
    work_dir: Path,
) -> ExperimentSpec:
    return ExperimentSpec(
        python=config.python,
        project_root=config.project_root,
        config=config.config,
        work_dir=work_dir,
        seed=seed,
        method=method.name,
        gpu_index=gpu.index,
        gpu_uuid=gpu.uuid,
        method_env=dict(method.environment),
        cfg_options=merge_cfg_options(config.common_cfg_options, method.cfg_options),
        monitor_interval_seconds=config.monitor_interval_seconds,
        stall_timeout_seconds=config.stall_timeout_seconds,
        gpu_verify_timeout_seconds=config.gpu_verify_timeout_seconds,
        terminate_grace_seconds=config.terminate_grace_seconds,
    )


def _method_fingerprint(
    config: SchedulerConfig,
    seed: int,
    method: MethodSpec,
    gpu: Optional[GPUInfo] = None,
) -> str:
    placeholder_gpu = gpu or GPUInfo(0, "GPU-UNASSIGNED", "", 0, 0, 0, 0)
    spec = _build_experiment_spec(
        config,
        seed,
        placeholder_gpu,
        method,
        _seed_method_dir(config, method.name, seed),
    )
    return str(compute_experiment_fingerprint(spec)["sha256"])


def run_seed_block(
    config: SchedulerConfig,
    seed: int,
    gpu: GPUInfo,
    methods: Sequence[MethodSpec],
    registry: Registry,
    *,
    run_one: Callable[..., RunOutcome] = run_experiment,
    compute_apps_probe: Callable[[], list[ComputeApp]] = query_compute_apps,
    wait_release: Callable[..., bool] = wait_for_gpu_release,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[RunOutcome]:
    """Run all methods for one seed sequentially on one already-locked GPU."""

    applicable_methods = [method for method in methods if method.applies_to(seed)]
    if not applicable_methods:
        raise ValueError(f"no applicable methods for seed {seed}")
    known_gpu_uuids = registry.seed_gpu_uuids(seed)
    if len(known_gpu_uuids) > 1:
        raise ValueError(
            f"seed {seed} has conflicting registry GPU UUIDs: "
            f"{sorted(known_gpu_uuids)}"
        )
    if known_gpu_uuids and gpu.uuid not in known_gpu_uuids:
        raise ValueError(
            f"seed {seed} is pinned to {next(iter(known_gpu_uuids))}, "
            f"not {gpu.uuid}"
        )
    by_name = {method.name: method for method in applicable_methods}
    ordered_names = randomized_method_order(
        list(by_name), seed=seed, order_seed=config.order_seed
    )
    outcomes: list[RunOutcome] = []

    for method_order, method_name in enumerate(ordered_names):
        method = by_name[method_name]
        seed_dir = _seed_method_dir(config, method_name, seed)
        expected_fingerprint = _method_fingerprint(config, seed, method, gpu)
        if registry.completed(
            seed,
            method_name,
            expected_fingerprint=expected_fingerprint,
            expected_gpu_uuid=gpu.uuid,
        ) or recover_completed_attempt(
            registry,
            seed_dir,
            seed,
            method_name,
            expected_fingerprint=expected_fingerprint,
            expected_gpu_uuid=gpu.uuid,
        ):
            continue

        attempts_used = max(
            registry.attempt_count(seed, method_name),
            registry.max_attempt_number(seed, method_name),
            _max_disk_attempt(seed_dir),
        )
        while attempts_used < config.max_attempts:
            attempt = attempts_used + 1
            work_dir = seed_dir / (
                f"attempt_{attempt}_fp_{expected_fingerprint[:12]}_"
                f"gpu_{_short_uuid(gpu.uuid)}"
            )
            spec = _build_experiment_spec(config, seed, gpu, method, work_dir)
            cfg_options = spec.cfg_options
            command = build_train_command(spec)
            launch_env = build_environment(
                os.environ, method.environment, gpu.index, config.python
            )
            row_id = f"{seed}:{method_name}:{attempt}"
            started_at = time.time()
            row = {
                "row_id": row_id,
                "run_id": config.run_id,
                "seed": seed,
                "requested_seed": seed,
                "method": method_name,
                "method_order": method_order,
                "experiment_fingerprint": expected_fingerprint,
                "status": "starting",
                "attempt": attempt,
                "work_dir": str(work_dir.resolve()),
                "physical_gpu_index": gpu.index,
                "gpu_uuid": gpu.uuid,
                "gpu_name": gpu.name,
                "free_memory_mib_at_start": gpu.memory_free_mib,
                "started_at": started_at,
                "full_run": False,
                "config_path": str(config.config.resolve()),
                "cfg_options_json": json.dumps(cfg_options, ensure_ascii=False),
                "environment_json": json.dumps(
                    relevant_environment(launch_env),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "command_json": json.dumps(command, ensure_ascii=False),
                "result_json_path": str(work_dir.resolve() / "run_result.json"),
            }
            reserved_id, budget_reason = registry.try_reserve(
                row,
                max_gpu_hours=config.max_gpu_hours,
                max_full_runs=config.max_full_runs,
                now=started_at,
            )
            if reserved_id is None:
                return outcomes
            work_dir.mkdir(parents=True, exist_ok=True)

            def on_started(pid: int, current_row_id: str = row_id) -> None:
                registry.update(current_row_id, status="running", pid=pid)

            def stop_check() -> Optional[str]:
                return registry.gpu_budget_stop_reason(config.max_gpu_hours)

            try:
                outcome = run_one(
                    spec,
                    on_started=on_started,
                    stop_check=stop_check,
                )
            except Exception as error:  # Keep the block resumable.
                ended_at = time.time()
                outcome = RunOutcome(
                    status="failed",
                    success=False,
                    method=method_name,
                    seed=seed,
                    gpu_index=gpu.index,
                    gpu_uuid=gpu.uuid,
                    work_dir=str(work_dir.resolve()),
                    command=command,
                    environment=relevant_environment(launch_env),
                    failure_kind="runner_exception",
                    failure_detail=f"{type(error).__name__}: {error}",
                    started_at=started_at,
                    ended_at=ended_at,
                    gpu_seconds=max(0.0, ended_at - started_at),
                    experiment_fingerprint=expected_fingerprint,
                )

            registry.update(
                row_id,
                status=outcome.status,
                pid=outcome.pid,
                exit_code=outcome.exit_code,
                failure_kind=outcome.failure_kind,
                failure_detail=outcome.failure_detail,
                actual_seed=outcome.actual_seed,
                ended_at=outcome.ended_at or time.time(),
                gpu_seconds=outcome.gpu_seconds,
                full_run=outcome.full_run,
                final_map=outcome.final_map,
                environment_json=json.dumps(
                    outcome.environment, ensure_ascii=False, sort_keys=True
                ),
                command_json=json.dumps(outcome.command, ensure_ascii=False),
                experiment_fingerprint=(
                    outcome.experiment_fingerprint or expected_fingerprint
                ),
            )
            outcomes.append(outcome)
            attempts_used = attempt

            if outcome.pid is not None:
                released = wait_release(
                    outcome.pid,
                    compute_apps_probe,
                    sleeper,
                    clock,
                    config.gpu_release_poll_seconds,
                    config.gpu_release_timeout_seconds,
                )
                if not released:
                    registry.update(
                        row_id,
                        status="partial",
                        failure_kind="gpu_release_timeout",
                        failure_detail=(
                            f"PID {outcome.pid} remained visible after "
                            f"{config.gpu_release_timeout_seconds:.1f}s"
                        ),
                    )
                    return outcomes
            if outcome.success:
                break
            if outcome.failure_kind == "budget_exhausted":
                return outcomes

    return outcomes


def _seed_terminal(
    config: SchedulerConfig,
    registry: Registry,
    seed: int,
    methods: Sequence[MethodSpec],
) -> bool:
    applicable_methods = [method for method in methods if method.applies_to(seed)]
    if not applicable_methods:
        return False
    if registry.seed_has_terminal_failure(seed):
        return True
    preferred_gpu_uuid = registry.preferred_gpu_uuid(seed)
    for method in applicable_methods:
        expected_fingerprint = _method_fingerprint(config, seed, method)
        if registry.completed(
            seed,
            method.name,
            expected_fingerprint=expected_fingerprint,
            expected_gpu_uuid=preferred_gpu_uuid,
        ):
            continue
        if registry.method_has_terminal_failure(seed, method.name):
            continue
        attempts_used = max(
            registry.attempt_count(seed, method.name),
            registry.max_attempt_number(seed, method.name),
            _max_disk_attempt(_seed_method_dir(config, method.name, seed)),
        )
        if attempts_used >= config.max_attempts:
            continue
        return False
    return True


def _run_locked_seed_block(
    lock: GPULock,
    config: SchedulerConfig,
    seed: int,
    gpu: GPUInfo,
    methods: Sequence[MethodSpec],
    registry: Registry,
) -> list[RunOutcome]:
    try:
        return run_seed_block(config, seed, gpu, methods, registry)
    finally:
        lock.release()


def run_scheduler(
    config: SchedulerConfig,
    methods: Sequence[MethodSpec],
    *,
    inventory_probe: Callable[[], list[GPUInfo]] = query_gpu_inventory,
    compute_apps_probe: Callable[[], list[ComputeApp]] = query_compute_apps,
    lock_factory: Callable[[Path, str], GPULock] = GPULock,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    unique_seeds = list(dict.fromkeys(config.seeds))
    methods_by_seed: dict[int, list[MethodSpec]] = {}
    for seed in unique_seeds:
        applicable = [method for method in methods if method.applies_to(seed)]
        if not applicable:
            raise ValueError(f"no applicable methods for seed {seed}")
        methods_by_seed[seed] = applicable

    config.research_root.mkdir(parents=True, exist_ok=True)
    registry = Registry(config.registry_path, config.run_id)
    registry.reconcile()
    for seed in unique_seeds:
        known_gpu_uuids = registry.seed_gpu_uuids(seed)
        if len(known_gpu_uuids) > 1:
            raise ValueError(
                f"seed {seed} has conflicting registry GPU UUIDs: "
                f"{sorted(known_gpu_uuids)}"
            )

    pending = list(unique_seeds)
    active: dict[
        concurrent.futures.Future[list[RunOutcome]],
        tuple[int, str, GPULock],
    ] = {}
    local_active_seeds: set[int] = set()
    terminal_seeds: set[int] = set()
    no_gpu_since: Optional[float] = None
    blocked_reason: Optional[str] = None
    max_workers = max(1, min(4, config.max_workers))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or active:
            registry.reconcile()
            still_pending: list[int] = []
            for seed in pending:
                if _seed_terminal(config, registry, seed, methods_by_seed[seed]):
                    terminal_seeds.add(seed)
                else:
                    still_pending.append(seed)
            pending = still_pending

            usage = registry.budget_usage()
            if (
                usage["gpu_hours"] >= config.max_gpu_hours
                or usage["full_runs"] >= config.max_full_runs
            ):
                blocked_reason = "budget_exhausted"
                pending.clear()

            assigned_any = False
            if blocked_reason is None and pending and len(active) < max_workers:
                try:
                    inventory = inventory_probe()
                except (OSError, RuntimeError, ValueError):
                    inventory = []
                    blocked_reason = "gpu_inventory_probe_failed"
                compute_apps: list[ComputeApp] = []
                if blocked_reason is None:
                    try:
                        compute_apps = compute_apps_probe()
                    except (OSError, RuntimeError, ValueError):
                        blocked_reason = "compute_app_probe_failed"
                busy_uuids = {app.gpu_uuid for app in compute_apps}
                active_uuids = {gpu_uuid for _, gpu_uuid, _ in active.values()}
                reserved_uuids = registry.active_gpu_uuids() | active_uuids
                candidates = sorted(
                    (
                        gpu
                        for gpu in inventory
                        if blocked_reason is None
                        and gpu.uuid not in reserved_uuids
                        and eligible_gpu(
                            gpu,
                            config.required_free_mib,
                            config.utilization_limit,
                            busy_uuids,
                        )
                    ),
                    key=lambda gpu: (-gpu.memory_free_mib, gpu.index),
                )

                for seed in list(pending):
                    if len(active) >= max_workers or not candidates:
                        break
                    if seed in local_active_seeds or registry.seed_has_active_process(
                        seed
                    ):
                        continue
                    preferred_uuid = registry.preferred_gpu_uuid(seed)
                    selected_index: Optional[int] = None
                    for index, gpu in enumerate(candidates):
                        if preferred_uuid is None or gpu.uuid == preferred_uuid:
                            selected_index = index
                            break
                    if selected_index is None:
                        continue
                    gpu = candidates.pop(selected_index)
                    lock = lock_factory(config.lock_root, gpu.uuid)
                    if not lock.acquire():
                        continue
                    future = executor.submit(
                        _run_locked_seed_block,
                        lock,
                        config,
                        seed,
                        gpu,
                        methods_by_seed[seed],
                        registry,
                    )
                    active[future] = (seed, gpu.uuid, lock)
                    local_active_seeds.add(seed)
                    assigned_any = True

            if assigned_any or active:
                no_gpu_since = None

            if active:
                done, _ = concurrent.futures.wait(
                    active,
                    timeout=config.gpu_poll_interval_seconds,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    seed, gpu_uuid, lock = active.pop(future)
                    local_active_seeds.discard(seed)
                    try:
                        future.result()
                    except Exception as error:
                        lock.release()
                        detail = f"{type(error).__name__}: {error}"
                        for method in methods_by_seed[seed]:
                            expected_fingerprint = _method_fingerprint(
                                config, seed, method
                            )
                            if registry.completed(
                                seed,
                                method.name,
                                expected_fingerprint=expected_fingerprint,
                                expected_gpu_uuid=gpu_uuid,
                            ) or recover_completed_attempt(
                                registry,
                                _seed_method_dir(config, method.name, seed),
                                seed,
                                method.name,
                                expected_fingerprint=expected_fingerprint,
                                expected_gpu_uuid=gpu_uuid,
                            ):
                                continue
                            registry.add_terminal_failure(
                                seed,
                                method.name,
                                gpu_uuid=gpu_uuid,
                                failure_kind="seed_block_future_exception",
                                failure_detail=detail,
                            )
                continue

            if blocked_reason is not None:
                break
            if not pending:
                break
            if no_gpu_since is None:
                no_gpu_since = clock()
            if clock() - no_gpu_since >= config.max_gpu_wait_seconds:
                blocked_reason = "gpu_wait_timeout"
                break
            sleeper(config.gpu_poll_interval_seconds)

    registry.reconcile()
    final_usage = registry.budget_usage()
    completed_jobs = 0
    failed_jobs = 0
    unresolved_seeds: set[int] = set()
    for seed, applicable_methods in methods_by_seed.items():
        preferred_gpu_uuid = registry.preferred_gpu_uuid(seed)
        for method in applicable_methods:
            expected_fingerprint = _method_fingerprint(config, seed, method)
            if registry.completed(
                seed,
                method.name,
                expected_fingerprint=expected_fingerprint,
                expected_gpu_uuid=preferred_gpu_uuid,
            ):
                completed_jobs += 1
                continue
            terminal_failure = (
                registry.seed_has_terminal_failure(seed)
                or registry.method_has_terminal_failure(seed, method.name)
                or max(
                    registry.attempt_count(seed, method.name),
                    registry.max_attempt_number(seed, method.name),
                    _max_disk_attempt(_seed_method_dir(config, method.name, seed)),
                )
                >= config.max_attempts
            )
            if terminal_failure:
                failed_jobs += 1
            elif seed not in terminal_seeds:
                failed_jobs += 1
                unresolved_seeds.add(seed)

    if blocked_reason is not None:
        final_status = "blocked"
    elif failed_jobs:
        final_status = "partial"
    else:
        final_status = "finished"
    return {
        "status": final_status,
        "blocked_reason": blocked_reason,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "pending_seeds": sorted(set(pending) | unresolved_seeds),
        **final_usage,
        "registry": str(config.registry_path),
    }


def _validate_method(method: MethodSpec) -> None:
    if not METHOD_NAME_RE.fullmatch(method.name):
        raise ValueError(f"invalid method name: {method.name!r}")
    validate_method_environment(method.environment)
    if method.seeds is not None:
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in method.seeds
        ):
            raise ValueError(f"method {method.name!r} seeds must be integers")
        if len(set(method.seeds)) != len(method.seeds):
            raise ValueError(f"method {method.name!r} has duplicate seeds")
    merge_cfg_options(method.cfg_options)


def builtin_methods(lora_path: Path) -> dict[str, MethodSpec]:
    return {
        "no_cga": MethodSpec(
            name="no_cga",
            environment={"CGA_SCORER": "none"},
        ),
        "legacy": MethodSpec(
            name="legacy",
            environment={
                "CGA_SCORER": "sarclip",
                "CGA_BACKEND": "sarclip",
                "CGA_FILTER_MODE": "legacy",
                "CGA_EXPAND_RATIO": "0.4",
                "CGA_BLEND_DET_WEIGHT": "0.7",
                "SARCLIP_LORA": str(lora_path.resolve()),
            },
        ),
    }


def load_method_specs(
    path: Optional[Path], selected_names: Sequence[str], lora_path: Path
) -> list[MethodSpec]:
    methods = builtin_methods(lora_path)
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload.get("methods") if isinstance(payload, dict) else payload
        if not isinstance(entries, list):
            raise ValueError("method specs must be a list or {'methods': [...]} object")
        methods = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("each method spec must be an object")
            name = str(entry.get("name", ""))
            environment = entry.get("environment", {})
            cfg_options = entry.get("cfg_options", [])
            seeds = entry.get("seeds")
            if (
                not isinstance(environment, dict)
                or not isinstance(cfg_options, list)
                or (seeds is not None and not isinstance(seeds, list))
            ):
                raise ValueError(f"invalid method spec for {name!r}")
            method = MethodSpec(
                name=name,
                environment={str(k): str(v) for k, v in environment.items()},
                cfg_options=tuple(str(value) for value in cfg_options),
                seeds=(tuple(seeds) if seeds is not None else None),
            )
            _validate_method(method)
            if name in methods:
                raise ValueError(f"duplicate method: {name}")
            methods[name] = method

    names = list(selected_names) if selected_names else list(methods)
    unknown = [name for name in names if name not in methods]
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(unknown)}")
    selected = [methods[name] for name in names]
    for method in selected:
        _validate_method(method)
        lora = method.environment.get("SARCLIP_LORA")
        if lora and not Path(lora).is_file():
            raise FileNotFoundError(f"SARCLIP_LORA not found for {method.name}: {lora}")
    return selected


def build_dry_run_plan(
    config: SchedulerConfig,
    methods: Sequence[MethodSpec],
    gpu_index: int,
    gpu_uuid: str,
) -> dict[str, Any]:
    plans = []
    for seed in config.seeds:
        applicable = [method for method in methods if method.applies_to(seed)]
        if not applicable:
            raise ValueError(f"no applicable methods for seed {seed}")
        by_name = {method.name: method for method in applicable}
        order = randomized_method_order(list(by_name), seed, config.order_seed)
        for method_order, method_name in enumerate(order):
            method = by_name[method_name]
            fingerprint = _method_fingerprint(config, seed, method)
            work_dir = _seed_method_dir(config, method_name, seed) / (
                f"attempt_1_fp_{fingerprint[:12]}_" f"gpu_{_short_uuid(gpu_uuid)}"
            )
            gpu = GPUInfo(
                gpu_index, gpu_uuid, "dry-run", 0, config.required_free_mib, 0, 0
            )
            spec = _build_experiment_spec(config, seed, gpu, method, work_dir)
            environment = build_environment(
                os.environ, method.environment, gpu_index, config.python
            )
            plans.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "method_order": method_order,
                    "seed_allowlist": (
                        list(method.seeds) if method.seeds is not None else None
                    ),
                    "experiment_fingerprint": fingerprint,
                    "gpu_index": gpu_index,
                    "gpu_uuid": gpu_uuid,
                    "work_dir": str(work_dir.resolve()),
                    "command": build_train_command(spec),
                    "environment": relevant_environment(environment),
                }
            )
    return {"status": "dry_run", "jobs": plans}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("IRAOD_PYTHON"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/unbiased_teacher/sfod/"
            "unbiased_teacher_oriented_rcnn_selftraining_cga_rsar1_research.py"
        ),
    )
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--method-specs", type=Path)
    parser.add_argument(
        "--lora",
        type=Path,
        default=Path("work_dirs/sarclip_lora_rsar_train_corrupt_aabb_v1/lora_rsar.pth"),
    )
    parser.add_argument("--corruption", default="chaff")
    parser.add_argument("--cfg-option", action="append", default=[])
    parser.add_argument("--required-free-mib", type=int, default=8192)
    parser.add_argument("--smoke-peak-mib", type=int)
    parser.add_argument("--utilization-limit", type=int, default=30)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-gpu-hours", type=float, default=24.0)
    parser.add_argument("--max-full-runs", type=int, default=20)
    parser.add_argument("--gpu-poll-interval", type=float, default=60.0)
    parser.add_argument("--max-gpu-wait", type=float, default=7200.0)
    parser.add_argument("--monitor-interval", type=float, default=60.0)
    parser.add_argument("--stall-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-verify-timeout", type=float, default=300.0)
    parser.add_argument("--terminate-grace", type=float, default=30.0)
    parser.add_argument("--order-seed", type=int, default=20260714)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-gpu-index", type=int, default=0)
    parser.add_argument("--dry-run-gpu-uuid", default="GPU-DRY-RUN")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.python:
        parser.error("--python is required when IRAOD_PYTHON is unset")
    if not args.dry_run and args.smoke_peak_mib is None:
        parser.error(
            "--smoke-peak-mib is required for non-dry scheduling; "
            "measure one smoke training first"
        )

    project_root = args.project_root.resolve()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = project_root / config_path
    lora_path = args.lora
    if not lora_path.is_absolute():
        lora_path = project_root / lora_path
    method_specs_path = args.method_specs
    if method_specs_path is not None and not method_specs_path.is_absolute():
        method_specs_path = project_root / method_specs_path

    try:
        methods = load_method_specs(method_specs_path, args.method, lora_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    default_cfg_options = [
        f"corrupt={args.corruption}",
        "optimizer.lr=0.000125",
        "model.cfg.weight_l=1.0",
        "model.cfg.weight_u=0.3",
        "model.cfg.score_thr=0.9",
    ]
    try:
        common_cfg_options = merge_cfg_options(default_cfg_options, args.cfg_option)
    except ValueError as error:
        parser.error(str(error))

    required_free_mib = required_free_memory_mib(
        args.required_free_mib, args.smoke_peak_mib
    )

    config = SchedulerConfig(
        project_root=project_root,
        research_root=args.research_root.resolve(),
        python=str(Path(args.python).resolve()),
        config=config_path.resolve(),
        seeds=list(args.seed),
        common_cfg_options=common_cfg_options,
        run_id=args.research_root.resolve().name,
        required_free_mib=required_free_mib,
        utilization_limit=args.utilization_limit,
        max_workers=min(4, args.max_workers),
        max_attempts=args.max_attempts,
        max_gpu_hours=args.max_gpu_hours,
        max_full_runs=args.max_full_runs,
        gpu_poll_interval_seconds=args.gpu_poll_interval,
        max_gpu_wait_seconds=args.max_gpu_wait,
        monitor_interval_seconds=args.monitor_interval,
        stall_timeout_seconds=args.stall_timeout,
        gpu_verify_timeout_seconds=args.gpu_verify_timeout,
        terminate_grace_seconds=args.terminate_grace,
        order_seed=args.order_seed,
    )

    if args.dry_run:
        try:
            plan = build_dry_run_plan(
                config,
                methods,
                gpu_index=args.dry_run_gpu_index,
                gpu_uuid=args.dry_run_gpu_uuid,
            )
        except ValueError as error:
            parser.error(str(error))
        config.research_root.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            config.research_root / "scheduler_dry_run.json",
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return 0

    if not Path(config.python).is_file():
        parser.error(f"IRAOD Python not found: {config.python}")
    if not config.config.is_file():
        parser.error(f"config not found: {config.config}")

    try:
        result = run_scheduler(config, methods)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if result["status"] == "finished":
        return 0
    return 1 if result["status"] == "partial" else 2


if __name__ == "__main__":
    raise SystemExit(main())
