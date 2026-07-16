#!/usr/bin/env python3
"""Run and verify one IRAOD training experiment on one physical GPU.

This module deliberately accepts argv lists and environment mappings.  It does
not invoke a shell, select a GPU, retry a failed job, or compute statistics.
Those responsibilities belong to ``gpu_scheduler.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SEED_RE = re.compile(
    r"Set random seed to\s+(?P<seed>-?\d+)\s*,\s*"
    r"deterministic:\s*(?P<deterministic>True|False)"
)
PROGRESS_RE = re.compile(
    r"Epoch\s*\[(?P<epoch>\d+)\]\s*" r"\[(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\]"
)
VAL_RE = re.compile(
    r"Epoch\(val\)\s*\[(?P<epoch>\d+)\]\s*"
    r"\[(?P<iteration>\d+)\].*?mAP:\s*"
    r"(?P<map>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

FAILURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "cuda_oom",
        re.compile(
            r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|"
            r"CUDNN_STATUS_ALLOC_FAILED",
            re.IGNORECASE,
        ),
    ),
    (
        "import_error",
        re.compile(r"ImportError|ModuleNotFoundError", re.IGNORECASE),
    ),
    (
        "cga_initialization_failed",
        re.compile(
            r"(?:CGA[^\n]*(?:failed|failure|error|exception))|"
            r"(?:(?:failed|failure|error|exception)[^\n]*CGA)",
            re.IGNORECASE,
        ),
    ),
    (
        "nan_detected",
        re.compile(
            r"(?:(?:loss(?:_[A-Za-z0-9_]+)?|grad_norm|log_vars|tensor)"
            r"[^\n]*?(?:\bnan\b|[+-]?\binf(?:inity)?\b))|"
            r"(?:(?:\bnan\b|[+-]?\binf(?:inity)?\b)\s+"
            r"(?:detected|found|loss))|(?:FloatingPointError[^\n]*)",
            re.IGNORECASE,
        ),
    ),
)

RELEVANT_ENV_EXACT = {
    "CUDA_VISIBLE_DEVICES",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "IRAOD_PYTHON",
    "IRAOD_CONDA_PREFIX",
}
RELEVANT_ENV_PREFIXES = ("CGA_", "SARCLIP_")

ALLOWED_METHOD_ENV_KEYS = {
    "CGA_ADAPT_W_MAX",
    "CGA_ADAPT_W_MIN",
    "CGA_BACKEND",
    "CGA_BLEND_DET_WEIGHT",
    "CGA_BOOST_CLIP_THR",
    "CGA_BOOST_DET_THR",
    "CGA_BOOST_STRENGTH",
    "CGA_CLIP_MODEL",
    "CGA_CONFUSION_GROUPS",
    "CGA_DISAGREE_DELTA",
    "CGA_DISAGREE_SCORE_THR",
    "CGA_DROP_SCORE",
    "CGA_EXCLUDE_IDS",
    "CGA_EXPAND_RATIO",
    "CGA_FILTER_LOG_EVERY",
    "CGA_FILTER_MODE",
    "CGA_FORCE_GRAYSCALE",
    "CGA_GATE_PROB_THR",
    "CGA_PROTECT_DET_SCORE",
    "CGA_SCORER",
    "CGA_SEM_HIGH_THR",
    "CGA_SEM_LAMBDA",
    "CGA_SEM_LOW_THR",
    "CGA_SHUFFLE_SEED",
    "CGA_TAU",
    "CGA_TEMPLATES",
    "CGA_VETO_ENTROPY",
    "CGA_VETO_LABEL_LO",
    "CGA_VETO_LABEL_THR",
    "CGA_VETO_MARGIN",
    "CGA_VETO_PENALTY",
    "CGA_VETO_PRED_HI",
    "CGA_VETO_PRED_THR",
    "CGA_VETO_SKIP_CONTEXT",
    "SARCLIP_CACHE_DIR",
    "SARCLIP_DIR",
    "SARCLIP_LORA",
    "SARCLIP_MODEL",
    "SARCLIP_PRECISION",
    "SARCLIP_PRETRAINED",
}

FINGERPRINT_CODE_FILES = (
    "train.py",
    "tools/cga_research/run_experiment.py",
    "tools/cga_research/gpu_scheduler.py",
)
FINGERPRINT_CODE_GLOBS = (
    "sfod/**/*.py",
    "mmdet_extension/**/*.py",
    "configs/unbiased_teacher/**/*.py",
)


@dataclasses.dataclass(frozen=True)
class Progress:
    epoch: int
    iteration: int
    total: int


@dataclasses.dataclass(frozen=True)
class FinalVal:
    epoch: int
    iteration: int
    mean_ap: float
    log_path: str


@dataclasses.dataclass(frozen=True)
class ComputeApp:
    pid: int
    gpu_uuid: str
    used_memory_mib: int


@dataclasses.dataclass
class ExperimentSpec:
    python: str
    project_root: Path
    config: Path
    work_dir: Path
    seed: int
    method: str
    gpu_index: int
    gpu_uuid: str
    method_env: dict[str, str] = dataclasses.field(default_factory=dict)
    cfg_options: list[str] = dataclasses.field(default_factory=list)
    monitor_interval_seconds: float = 60.0
    stall_timeout_seconds: float = 900.0
    gpu_verify_timeout_seconds: float = 300.0
    terminate_grace_seconds: float = 30.0


@dataclasses.dataclass
class RunOutcome:
    status: str
    success: bool
    method: str
    seed: int
    gpu_index: int
    gpu_uuid: str
    work_dir: str
    command: list[str]
    environment: dict[str, str]
    experiment_fingerprint: Optional[str] = None
    fingerprint_components: dict[str, Any] = dataclasses.field(default_factory=dict)
    pid: Optional[int] = None
    exit_code: Optional[int] = None
    failure_kind: Optional[str] = None
    failure_detail: Optional[str] = None
    actual_seed: Optional[int] = None
    deterministic: Optional[bool] = None
    final_map: Optional[float] = None
    final_val_epoch: Optional[int] = None
    final_val_iteration: Optional[int] = None
    final_val_log: Optional[str] = None
    progress: Optional[dict[str, int]] = None
    gpu_seen: bool = False
    full_run: bool = False
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    gpu_seconds: float = 0.0
    launch_time_ns: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _validate_cfg_option(option: str) -> None:
    if "\x00" in option or "=" not in option:
        raise ValueError(f"invalid --cfg-options value: {option!r}")


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "missing"


def _read_git_head(project_root: Path) -> str:
    git_dir = project_root / ".git"
    if git_dir.is_file():
        try:
            content = git_dir.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                referenced = Path(content.split(":", 1)[1].strip())
                git_dir = (
                    referenced
                    if referenced.is_absolute()
                    else project_root / referenced
                )
        except OSError:
            return "unknown"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            return (git_dir / ref).read_text(encoding="utf-8").strip()
        return head
    except OSError:
        return "unknown"


def compute_experiment_fingerprint(spec: ExperimentSpec) -> dict[str, Any]:
    """Hash every input that can change the meaning of one experiment."""

    project_root = Path(spec.project_root).resolve()
    config_path = Path(spec.config).resolve()
    for option in spec.cfg_options:
        _validate_cfg_option(option)
    code_paths = {project_root / relative for relative in FINGERPRINT_CODE_FILES}
    for pattern in FINGERPRINT_CODE_GLOBS:
        code_paths.update(path for path in project_root.glob(pattern) if path.is_file())
    code_hashes = {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in sorted(code_paths)
    }
    method_artifacts: dict[str, dict[str, str]] = {}
    for key, value in sorted(spec.method_env.items()):
        artifact_path = Path(str(value)).expanduser()
        if not artifact_path.is_absolute():
            artifact_path = project_root / artifact_path
        if artifact_path.is_file():
            method_artifacts[key] = {
                "path": str(artifact_path.resolve()),
                "sha256": _sha256_file(artifact_path),
            }
    components: dict[str, Any] = {
        "schema_version": 2,
        "python_executable": str(Path(spec.python).expanduser().resolve()),
        "config_sha256": _sha256_file(config_path),
        "cfg_options": list(spec.cfg_options),
        "method": spec.method,
        "method_environment": {
            key: str(value) for key, value in sorted(spec.method_env.items())
        },
        "method_artifacts": method_artifacts,
        "seed": int(spec.seed),
        "git_head": _read_git_head(project_root),
        "code_files": code_hashes,
    }
    canonical = json.dumps(
        components,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), **components}


def validate_method_environment(method_environment: Mapping[str, str]) -> None:
    for key, value in method_environment.items():
        if not ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid environment key: {key!r}")
        if key not in ALLOWED_METHOD_ENV_KEYS:
            raise ValueError(f"unsupported method environment key: {key}")
        if "\x00" in str(value) or "\n" in str(value) or "\r" in str(value):
            raise ValueError(f"unsafe environment value for {key}")


def build_train_command(spec: ExperimentSpec) -> list[str]:
    """Build a shell-free training argv list."""

    command = [
        str(spec.python),
        str(spec.project_root / "train.py"),
        str(spec.config),
        "--work-dir",
        str(spec.work_dir),
        "--seed",
        str(spec.seed),
        "--deterministic",
    ]
    if spec.cfg_options:
        for option in spec.cfg_options:
            _validate_cfg_option(option)
        command.append("--cfg-options")
        command.extend(str(option) for option in spec.cfg_options)
    return command


def relevant_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in sorted(environment.items())
        if key in RELEVANT_ENV_EXACT
        or any(key.startswith(prefix) for prefix in RELEVANT_ENV_PREFIXES)
    }


def build_environment(
    base_environment: Mapping[str, str],
    method_environment: Mapping[str, str],
    gpu_index: int,
    python: str,
) -> dict[str, str]:
    """Build a clean per-method environment without CGA/SARCLIP leakage."""

    environment = {
        str(key): str(value)
        for key, value in base_environment.items()
        if not any(str(key).startswith(prefix) for prefix in RELEVANT_ENV_PREFIXES)
    }
    validate_method_environment(method_environment)
    for key, value in method_environment.items():
        environment[key] = str(value)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_index),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "IRAOD_PYTHON": str(python),
        }
    )
    return environment


def parse_compute_apps_csv(text: str) -> list[ComputeApp]:
    apps: list[ComputeApp] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        columns = [part.strip() for part in line.split(",")]
        if len(columns) != 3:
            raise ValueError(f"malformed compute-app row {line_number}: {raw_line!r}")
        apps.append(
            ComputeApp(
                pid=int(columns[0]),
                gpu_uuid=columns[1],
                used_memory_mib=int(columns[2]),
            )
        )
    return apps


def query_compute_apps() -> list[ComputeApp]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
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
    return parse_compute_apps_csv(completed.stdout)


def read_proc_environment(pid: int, proc_root: Path = Path("/proc")) -> dict[str, str]:
    raw = (proc_root / str(pid) / "environ").read_bytes()
    environment: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "replace")] = value.decode("utf-8", "replace")
    return environment


def environment_mismatches(
    actual: Mapping[str, str], expected: Mapping[str, str]
) -> list[str]:
    mismatches = []
    expected_relevant = relevant_environment(expected)
    actual_relevant = relevant_environment(actual)
    for key, expected_value in expected_relevant.items():
        actual_value = actual.get(key)
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: expected {expected_value!r}, got {actual_value!r}"
            )
    for key in actual_relevant:
        if key not in expected_relevant and any(
            key.startswith(prefix) for prefix in RELEVANT_ENV_PREFIXES
        ):
            mismatches.append(f"{key}: unexpected method environment value")
    return mismatches


def format_process_environment(environment: Mapping[str, str]) -> str:
    selected = relevant_environment(environment)
    return "".join(f"{key}={value}\n" for key, value in selected.items())


def parse_actual_seed(text: str) -> Optional[tuple[int, bool]]:
    matches = list(SEED_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return (
        int(match.group("seed")),
        match.group("deterministic") == "True",
    )


def parse_progress(text: str) -> Optional[Progress]:
    matches = list(PROGRESS_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return Progress(
        epoch=int(match.group("epoch")),
        iteration=int(match.group("iteration")),
        total=int(match.group("total")),
    )


def classify_failure(text: str) -> Optional[str]:
    for name, pattern in FAILURE_PATTERNS:
        if pattern.search(text):
            return name
    return None


def _timestamp_log_signature(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def snapshot_timestamp_logs(work_dir: Path) -> dict[str, tuple[int, int, int]]:
    """Capture timestamp-log identities before a launch.

    Filesystems may expose mtimes several milliseconds older than ``time_ns()``.
    Comparing identities and metadata is therefore safer than relying solely on
    a launch timestamp when deciding whether a log belongs to this attempt.
    """

    snapshot: dict[str, tuple[int, int, int]] = {}
    for path in work_dir.glob("*.log"):
        if path.is_file() and path.name != "run_train.log":
            snapshot[path.name] = _timestamp_log_signature(path)
    return snapshot


def discover_timestamp_logs(
    work_dir: Path,
    not_before_ns: Optional[int] = None,
    baseline: Optional[Mapping[str, tuple[int, int, int]]] = None,
) -> list[Path]:
    logs = [
        path
        for path in work_dir.glob("*.log")
        if path.is_file()
        and path.name != "run_train.log"
        and (
            baseline is not None
            and baseline.get(path.name) != _timestamp_log_signature(path)
            or baseline is None
            and (not_before_ns is None or path.stat().st_mtime_ns >= not_before_ns)
        )
    ]
    return sorted(logs, key=lambda path: (path.stat().st_mtime_ns, path.name))


def read_training_logs(
    work_dir: Path,
    not_before_ns: Optional[int] = None,
    baseline: Optional[Mapping[str, tuple[int, int, int]]] = None,
) -> str:
    paths = [
        work_dir / "run_train.log",
        *discover_timestamp_logs(work_dir, not_before_ns, baseline),
    ]
    chunks = []
    for path in paths:
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        except FileNotFoundError:
            continue
    return "\n".join(chunks)


def parse_final_ema_record(
    work_dir: Path,
    not_before_ns: Optional[int] = None,
    baseline: Optional[Mapping[str, tuple[int, int, int]]] = None,
) -> Optional[FinalVal]:
    final_record: Optional[FinalVal] = None
    for path in discover_timestamp_logs(work_dir, not_before_ns, baseline):
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = list(VAL_RE.finditer(text))
        if matches:
            match = matches[-1]
            final_record = FinalVal(
                epoch=int(match.group("epoch")),
                iteration=int(match.group("iteration")),
                mean_ap=float(match.group("map")),
                log_path=str(path),
            )
    return final_record


def parse_final_ema_map(
    work_dir: Path, not_before_ns: Optional[int] = None
) -> Optional[float]:
    """Return the last Epoch(val) mAP from the last completed timestamp log.

    The project's SemiEvalHook evaluates the student first and EMA second.  Its
    final ``Epoch(val)`` log-buffer record therefore represents the EMA result.
    ``run_train.log`` is intentionally excluded to avoid duplicated logger rows.
    """

    record = parse_final_ema_record(work_dir, not_before_ns)
    return record.mean_ap if record is not None else None


def _append_monitor_event(work_dir: Path, event: Mapping[str, Any]) -> None:
    path = work_dir / "monitor.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _capture(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr


def snapshot_process_artifacts(spec: ExperimentSpec, pid: int) -> None:
    snapshots = {
        "ps.txt": ["ps", "-fp", str(pid)],
        "nvidia_smi.txt": ["nvidia-smi"],
        "nvidia_compute_apps.txt": [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
    }
    for filename, command in snapshots.items():
        try:
            output = _capture(command)
        except (OSError, subprocess.SubprocessError) as error:
            output = f"snapshot failed: {error}\n"
        _atomic_write_text(spec.work_dir / filename, output)


def _terminate_process(
    process: subprocess.Popen[Any], grace_seconds: float
) -> Optional[int]:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        return process.wait()


def _checkpoint_exists(work_dir: Path) -> bool:
    return any(work_dir.glob("*.pth"))


def _write_outcome(spec: ExperimentSpec, outcome: RunOutcome) -> None:
    _atomic_write_json(spec.work_dir / "run_result.json", outcome.to_dict())


def run_experiment(
    spec: ExperimentSpec,
    *,
    dry_run: bool = False,
    base_environment: Optional[Mapping[str, str]] = None,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    environment_reader: Callable[[int], Mapping[str, str]] = read_proc_environment,
    compute_apps_probe: Callable[[], list[ComputeApp]] = query_compute_apps,
    snapshotter: Callable[[ExperimentSpec, int], None] = snapshot_process_artifacts,
    terminator: Callable[[subprocess.Popen[Any], float], Optional[int]] = (
        _terminate_process
    ),
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], float] = time.time,
    sleeper: Callable[[float], None] = time.sleep,
    on_started: Optional[Callable[[int], None]] = None,
    stop_check: Optional[Callable[[], Optional[str]]] = None,
) -> RunOutcome:
    """Run one task synchronously and return a structured outcome."""

    spec.project_root = Path(spec.project_root).resolve()
    spec.config = Path(spec.config).resolve()
    spec.work_dir = Path(spec.work_dir).resolve()
    spec.work_dir.mkdir(parents=True, exist_ok=True)

    command = build_train_command(spec)
    environment = build_environment(
        base_environment if base_environment is not None else os.environ,
        spec.method_env,
        spec.gpu_index,
        spec.python,
    )
    recorded_environment = relevant_environment(environment)
    fingerprint_payload = compute_experiment_fingerprint(spec)
    experiment_fingerprint = str(fingerprint_payload["sha256"])
    fingerprint_components = {
        key: value for key, value in fingerprint_payload.items() if key != "sha256"
    }
    _atomic_write_json(spec.work_dir / "command.json", {"argv": command})
    _atomic_write_json(spec.work_dir / "launch_environment.json", recorded_environment)

    if dry_run:
        outcome = RunOutcome(
            status="dry_run",
            success=True,
            method=spec.method,
            seed=spec.seed,
            gpu_index=spec.gpu_index,
            gpu_uuid=spec.gpu_uuid,
            work_dir=str(spec.work_dir),
            command=command,
            environment=recorded_environment,
            experiment_fingerprint=experiment_fingerprint,
            fingerprint_components=fingerprint_components,
        )
        _write_outcome(spec, outcome)
        return outcome

    started_at_wall = wall_clock()
    started_at_mono = clock()
    launch_time_ns = time.time_ns()
    timestamp_log_baseline = snapshot_timestamp_logs(spec.work_dir)
    process: Optional[subprocess.Popen[Any]] = None
    failure_kind: Optional[str] = None
    failure_detail: Optional[str] = None
    gpu_seen = False
    last_log_signature: Optional[tuple[int, int]] = None
    last_log_change = started_at_mono
    latest_progress: Optional[Progress] = None
    exit_code: Optional[int] = None
    last_compute_probe_ok: Optional[bool] = None

    run_log_path = spec.work_dir / "run_train.log"
    _atomic_write_text(spec.work_dir / "monitor.jsonl", "")
    try:
        run_log = run_log_path.open("w", encoding="utf-8", buffering=1)
    except OSError as error:
        ended_at = wall_clock()
        outcome = RunOutcome(
            status="failed",
            success=False,
            method=spec.method,
            seed=spec.seed,
            gpu_index=spec.gpu_index,
            gpu_uuid=spec.gpu_uuid,
            work_dir=str(spec.work_dir),
            command=command,
            environment=recorded_environment,
            experiment_fingerprint=experiment_fingerprint,
            fingerprint_components=fingerprint_components,
            failure_kind="artifact_open_failed",
            failure_detail=str(error),
            started_at=started_at_wall,
            ended_at=ended_at,
            gpu_seconds=max(0.0, ended_at - started_at_wall),
            launch_time_ns=launch_time_ns,
        )
        _write_outcome(spec, outcome)
        return outcome

    with run_log:
        try:
            process = popen_factory(
                command,
                cwd=str(spec.project_root),
                env=environment,
                stdout=run_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            ended_at = wall_clock()
            outcome = RunOutcome(
                status="failed",
                success=False,
                method=spec.method,
                seed=spec.seed,
                gpu_index=spec.gpu_index,
                gpu_uuid=spec.gpu_uuid,
                work_dir=str(spec.work_dir),
                command=command,
                environment=recorded_environment,
                experiment_fingerprint=experiment_fingerprint,
                fingerprint_components=fingerprint_components,
                failure_kind="spawn_failed",
                failure_detail=str(error),
                started_at=started_at_wall,
                ended_at=ended_at,
                gpu_seconds=max(0.0, ended_at - started_at_wall),
                launch_time_ns=launch_time_ns,
            )
            _write_outcome(spec, outcome)
            return outcome

        try:
            if on_started is not None:
                on_started(process.pid)

            try:
                actual_environment = dict(environment_reader(process.pid))
                _atomic_write_text(
                    spec.work_dir / "process_environment.txt",
                    format_process_environment(actual_environment),
                )
                mismatches = environment_mismatches(actual_environment, environment)
                if mismatches:
                    failure_kind = "environment_mismatch"
                    failure_detail = "; ".join(mismatches)
            except (OSError, ValueError) as error:
                failure_kind = "environment_unreadable"
                failure_detail = str(error)

            snapshotter(spec, process.pid)
            _append_monitor_event(
                spec.work_dir,
                {
                    "event": "started",
                    "timestamp": wall_clock(),
                    "pid": process.pid,
                    "gpu_uuid": spec.gpu_uuid,
                },
            )

            while process.poll() is None and failure_kind is None:
                now = clock()
                try:
                    log_text = run_log_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except FileNotFoundError:
                    log_text = ""

                try:
                    log_stat = run_log_path.stat()
                    current_signature = (log_stat.st_size, log_stat.st_mtime_ns)
                except FileNotFoundError:
                    current_signature = (0, 0)
                if current_signature != last_log_signature:
                    last_log_signature = current_signature
                    last_log_change = now

                latest_progress = parse_progress(log_text) or latest_progress
                detected_failure = classify_failure(log_text)
                if detected_failure is not None:
                    failure_kind = detected_failure
                    failure_detail = f"detected in {run_log_path.name}"

                actual_seed_record = parse_actual_seed(log_text)
                if failure_kind is None and actual_seed_record is not None:
                    actual_seed, deterministic = actual_seed_record
                    if actual_seed != spec.seed:
                        failure_kind = "seed_mismatch"
                        failure_detail = (
                            f"requested seed {spec.seed}, log reports {actual_seed}"
                        )
                    elif not deterministic:
                        failure_kind = "deterministic_mismatch"
                        failure_detail = "training log reports deterministic=False"

                compute_probe_ok = True
                apps: list[ComputeApp] = []
                try:
                    apps = compute_apps_probe()
                    gpu_seen = gpu_seen or any(
                        app.pid == process.pid and app.gpu_uuid == spec.gpu_uuid
                        for app in apps
                    )
                except (OSError, RuntimeError, ValueError) as error:
                    compute_probe_ok = False
                    failure_kind = "compute_app_probe_failed"
                    failure_detail = str(error)
                last_compute_probe_ok = compute_probe_ok

                if failure_kind is None and stop_check is not None:
                    stop_reason = stop_check()
                    if stop_reason:
                        failure_kind = "budget_exhausted"
                        failure_detail = str(stop_reason)

                if (
                    failure_kind is None
                    and now - last_log_change >= spec.stall_timeout_seconds
                ):
                    failure_kind = "log_stalled"
                    failure_detail = (
                        f"no log update for {spec.stall_timeout_seconds:.1f}s"
                    )

                if (
                    failure_kind is None
                    and not gpu_seen
                    and now - started_at_mono >= spec.gpu_verify_timeout_seconds
                ):
                    failure_kind = "gpu_verification_failed"
                    failure_detail = (
                        f"PID {process.pid} not observed on GPU {spec.gpu_uuid}"
                    )

                _append_monitor_event(
                    spec.work_dir,
                    {
                        "event": "poll",
                        "timestamp": wall_clock(),
                        "monotonic": now,
                        "pid": process.pid,
                        "alive": process.poll() is None,
                        "log_size": current_signature[0],
                        "log_mtime_ns": current_signature[1],
                        "progress": (
                            dataclasses.asdict(latest_progress)
                            if latest_progress is not None
                            else None
                        ),
                        "gpu_seen": gpu_seen,
                        "compute_probe_ok": compute_probe_ok,
                        "compute_apps": [dataclasses.asdict(app) for app in apps],
                        "failure_kind": failure_kind,
                    },
                )
                if failure_kind is not None:
                    break
                sleeper(spec.monitor_interval_seconds)
        except BaseException as error:
            failure_kind = "runner_exception"
            failure_detail = f"{type(error).__name__}: {error}"
        finally:
            if process.poll() is None:
                try:
                    terminator(process, spec.terminate_grace_seconds)
                except BaseException as error:
                    if failure_kind is None:
                        failure_kind = "termination_failed"
                        failure_detail = f"{type(error).__name__}: {error}"
            exit_code = process.poll()
            if exit_code is None:
                try:
                    exit_code = process.wait()
                except BaseException as error:
                    failure_kind = failure_kind or "wait_failed"
                    failure_detail = (
                        failure_detail or f"{type(error).__name__}: {error}"
                    )

    ended_at_wall = wall_clock()
    actual_seed: Optional[int] = None
    deterministic: Optional[bool] = None
    final_record: Optional[FinalVal] = None
    try:
        all_logs = read_training_logs(
            spec.work_dir, launch_time_ns, timestamp_log_baseline
        )
        final_detected_failure = classify_failure(all_logs)
        if failure_kind is None and final_detected_failure is not None:
            failure_kind = final_detected_failure
            failure_detail = "detected during final log validation"

        actual_seed_record = parse_actual_seed(all_logs)
        actual_seed = actual_seed_record[0] if actual_seed_record else None
        deterministic = actual_seed_record[1] if actual_seed_record else None
        latest_progress = parse_progress(all_logs) or latest_progress
        final_record = parse_final_ema_record(
            spec.work_dir, launch_time_ns, timestamp_log_baseline
        )
    except BaseException as error:
        failure_kind = failure_kind or "runner_exception"
        failure_detail = failure_detail or f"{type(error).__name__}: {error}"
    final_map = final_record.mean_ap if final_record is not None else None

    if failure_kind is None and exit_code != 0:
        failure_kind = "nonzero_exit"
        failure_detail = f"training process exited with code {exit_code}"
    if failure_kind is None and actual_seed is None:
        failure_kind = "seed_not_logged"
        failure_detail = "no 'Set random seed' record found"
    if failure_kind is None and actual_seed != spec.seed:
        failure_kind = "seed_mismatch"
        failure_detail = f"requested seed {spec.seed}, final log reports {actual_seed}"
    if failure_kind is None and deterministic is not True:
        failure_kind = "deterministic_mismatch"
        failure_detail = f"final deterministic flag is {deterministic!r}"
    if failure_kind is None and not gpu_seen:
        failure_kind = "gpu_verification_failed"
        failure_detail = f"PID {process.pid} was never observed on {spec.gpu_uuid}"
    if failure_kind is None and final_map is None:
        failure_kind = "missing_final_ema_eval"
        failure_detail = "no final Epoch(val) mAP in a timestamp log"
    if failure_kind is None and final_record is not None and final_record.epoch != 1:
        failure_kind = "unexpected_val_epoch"
        failure_detail = f"expected val epoch 1, got {final_record.epoch}"
    if failure_kind is None and latest_progress is None:
        failure_kind = "missing_training_progress"
        failure_detail = "no Epoch [n][iteration/total] training record found"
    if (
        failure_kind is None
        and latest_progress is not None
        and latest_progress.epoch != 1
    ):
        failure_kind = "unexpected_train_epoch"
        failure_detail = f"expected train epoch 1, got {latest_progress.epoch}"
    if (
        failure_kind is None
        and final_record is not None
        and latest_progress is not None
        and final_record.iteration != latest_progress.total
    ):
        failure_kind = "incomplete_training"
        failure_detail = (
            f"final runner iteration {final_record.iteration} != "
            f"training total {latest_progress.total}"
        )

    success = failure_kind is None
    full_run = final_map is not None or _checkpoint_exists(spec.work_dir)
    outcome = RunOutcome(
        status="completed" if success else "failed",
        success=success,
        method=spec.method,
        seed=spec.seed,
        gpu_index=spec.gpu_index,
        gpu_uuid=spec.gpu_uuid,
        work_dir=str(spec.work_dir),
        command=command,
        environment=recorded_environment,
        experiment_fingerprint=experiment_fingerprint,
        fingerprint_components=fingerprint_components,
        pid=process.pid,
        exit_code=exit_code,
        failure_kind=failure_kind,
        failure_detail=failure_detail,
        actual_seed=actual_seed,
        deterministic=deterministic,
        final_map=final_map,
        final_val_epoch=(final_record.epoch if final_record else None),
        final_val_iteration=(final_record.iteration if final_record else None),
        final_val_log=(final_record.log_path if final_record else None),
        progress=(dataclasses.asdict(latest_progress) if latest_progress else None),
        gpu_seen=gpu_seen,
        full_run=full_run,
        started_at=started_at_wall,
        ended_at=ended_at_wall,
        gpu_seconds=max(0.0, ended_at_wall - started_at_wall),
        launch_time_ns=launch_time_ns,
    )
    _write_outcome(spec, outcome)
    _append_monitor_event(
        spec.work_dir,
        {
            "event": "finished",
            "timestamp": ended_at_wall,
            "pid": process.pid if process is not None else None,
            "status": outcome.status,
            "failure_kind": outcome.failure_kind,
            "exit_code": outcome.exit_code,
            "final_map": outcome.final_map,
            "compute_probe_ok": last_compute_probe_ok,
        },
    )
    return outcome


def _parse_key_values(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected KEY=VALUE, got {value!r}")
        key, item_value = value.split("=", 1)
        if key in parsed:
            raise ValueError(f"duplicate environment key: {key}")
        parsed[key] = item_value
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=os.environ.get("IRAOD_PYTHON"))
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--cfg-option", action="append", default=[])
    parser.add_argument("--monitor-interval", type=float, default=60.0)
    parser.add_argument("--stall-timeout", type=float, default=900.0)
    parser.add_argument("--gpu-verify-timeout", type=float, default=300.0)
    parser.add_argument("--terminate-grace", type=float, default=30.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if not args.python:
        parser.error("--python is required when IRAOD_PYTHON is unset")
    try:
        method_env = _parse_key_values(args.env)
    except ValueError as error:
        parser.error(str(error))
    spec = ExperimentSpec(
        python=args.python,
        project_root=args.project_root,
        config=args.config,
        work_dir=args.work_dir,
        seed=args.seed,
        method=args.method,
        gpu_index=args.gpu_index,
        gpu_uuid=args.gpu_uuid,
        method_env=method_env,
        cfg_options=list(args.cfg_option),
        monitor_interval_seconds=args.monitor_interval,
        stall_timeout_seconds=args.stall_timeout,
        gpu_verify_timeout_seconds=args.gpu_verify_timeout,
        terminate_grace_seconds=args.terminate_grace,
    )
    outcome = run_experiment(spec, dry_run=args.dry_run)
    print(json.dumps(outcome.to_dict(), ensure_ascii=False, sort_keys=True))
    return 0 if outcome.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
