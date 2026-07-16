#!/usr/bin/env python3
"""Run every inference declared by one frozen final-test bundle.

The runner is deliberately fail closed.  It verifies the frozen bundle before
planning, verifies the byte-level dataset manifest before execution, and
repeats both checks immediately before every declared run.  It never accepts a
method, seed, checkpoint, dataset path, or output directory from the command
line.  Those values come only from the frozen manifest.

``--dry-run`` verifies frozen non-dataset inputs and prints the exact command
plan.  It does not rescan dataset roots, query GPUs, create output directories,
or start a model process.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cga_research import (  # noqa: E402
    build_data_manifest as data_manifest_tool,
)
from tools.cga_research import (  # noqa: E402
    final_test_manifest as manifest_tool,
)
from tools.cga_research import gpu_scheduler  # noqa: E402
from tools.cga_research import run_experiment  # noqa: E402


CHECKPOINT_BASENAME = "iter_4235_ema.pth"
PREDICTIONS_BASENAME = "predictions.pkl"
STDOUT_BASENAME = "test_stdout.log"
ONLINE_METADATA_BASENAME = "online_eval.json"
FRAMEWORK_EVAL_BASENAME = "framework_eval.json"
ATTEMPT_TERMINAL_BASENAME = "attempt_terminal.json"
CORE_OUTPUT_NAMES = frozenset(
    {
        PREDICTIONS_BASENAME,
        STDOUT_BASENAME,
        ONLINE_METADATA_BASENAME,
        FRAMEWORK_EVAL_BASENAME,
    }
)
EXPECTED_OUTPUT_NAMES = CORE_OUTPUT_NAMES
POSTPROCESS_BASENAME = "per_class_ap.json"
ATTEMPTS_DIRNAME = "attempts"
OUTPUT_ROOT_NAME = "final_test_outputs"
LOCK_ROOT_NAME = "final_test_locks"
REGISTRY_BASENAME = "final_test_registry.jsonl"
ALLOWED_RUN_OUTPUT_NAMES = CORE_OUTPUT_NAMES | {
    ATTEMPTS_DIRNAME,
    POSTPROCESS_BASENAME,
}
ATTEMPT_FIXED_ARTIFACT_NAMES = frozenset(
    {PREDICTIONS_BASENAME, STDOUT_BASENAME, ONLINE_METADATA_BASENAME}
)
ALLOWED_ATTEMPT_NAMES = ATTEMPT_FIXED_ARTIFACT_NAMES | {ATTEMPT_TERMINAL_BASENAME}
FRAMEWORK_EVAL_RE = re.compile(r"^eval_[0-9]{8}_[0-9]{6}\.json$")
DISABLED_CGA_ENV = {
    "CGA_BACKEND": "none",
    "CGA_FILTER_MODE": "none",
    "CGA_SCORER": "none",
}
MAX_WORKERS = 4
MAX_ATTEMPTS = 3
REGISTRY_SCHEMA_VERSION = 2
ONLINE_METADATA_SCHEMA_VERSION = 1
ATTEMPT_TERMINAL_SCHEMA_VERSION = 1
DEFAULT_GPU_VERIFY_TIMEOUT = 300.0
DEFAULT_MONITOR_INTERVAL = 60.0
DEFAULT_RUN_TIMEOUT = 21600.0
DEFAULT_STALL_TIMEOUT = 900.0
DEFAULT_TERMINATE_GRACE = 30.0
SAFE_SCALAR_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REGISTRY_EVENTS = {"started", "failed", "completed"}

REQUIRED_RUNTIME_FILES = (
    Path(__file__).resolve(),
    (PROJECT_ROOT / "test.py").resolve(),
    Path(manifest_tool.__file__).resolve(),
    Path(data_manifest_tool.__file__).resolve(),
    Path(gpu_scheduler.__file__).resolve(),
    Path(run_experiment.__file__).resolve(),
    (PROJECT_ROOT / "tools/cga_research/evaluate_predictions.py").resolve(),
    (PROJECT_ROOT / "tools/cga_research/evaluate_final_test_predictions.py").resolve(),
)
RSAR_CLASS_ORDER = (
    "ship",
    "aircraft",
    "car",
    "tank",
    "bridge",
    "harbor",
)


class FinalTestBundleError(RuntimeError):
    """Raised when final-test execution cannot be authorized safely."""


def _require_finite_interval(
    value: float,
    role: str,
    *,
    allow_zero: bool = False,
    maximum: Optional[float] = None,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (value < 0 if allow_zero else value <= 0)
        or (maximum is not None and value > maximum)
    ):
        lower = "non-negative" if allow_zero else "positive"
        upper = f" and at most {maximum:g}" if maximum is not None else ""
        raise FinalTestBundleError(f"{role} must be finite, {lower}{upper}")


@dataclasses.dataclass(frozen=True)
class FrozenRun:
    method: str
    seed: int
    training_fingerprint: str
    checkpoint_path: Path
    checkpoint_sha256: str
    output_dir: Path
    method_environment: Mapping[str, str]
    hyperparameters: Mapping[str, Any]

    @property
    def predictions_path(self) -> Path:
        return self.output_dir / PREDICTIONS_BASENAME

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / STDOUT_BASENAME

    @property
    def online_metadata_path(self) -> Path:
        return self.output_dir / ONLINE_METADATA_BASENAME

    @property
    def framework_eval_path(self) -> Path:
        return self.output_dir / FRAMEWORK_EVAL_BASENAME


@dataclasses.dataclass(frozen=True)
class FrozenBundle:
    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]
    project_root: Path
    config_path: Path
    data_manifest_path: Path
    data_manifest_sha256: str
    annotation_root: Path
    image_root: Path
    corruption: str
    runs: tuple[FrozenRun, ...]


@dataclasses.dataclass(frozen=True)
class RunResult:
    method: str
    seed: int
    status: str
    attempts: int
    gpu_uuid: Optional[str] = None
    detail: Optional[str] = None


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise FinalTestBundleError(f"cannot serialize JSON safely: {error}") from error
    return (rendered + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FinalTestBundleError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _file_identity(
    path: Path, role: str, *, require_nonempty: bool = True
) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        raise FinalTestBundleError(f"cannot inspect {role} {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise FinalTestBundleError(f"{role} must be a regular non-symlink file: {path}")
    if require_nonempty and info.st_size <= 0:
        raise FinalTestBundleError(f"{role} must not be empty: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": info.st_size,
    }


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolve_project_file(project_root: Path, raw: Any, role: str) -> Path:
    if type(raw) is not str or not raw.strip() or "\x00" in raw:
        raise FinalTestBundleError(f"{role} must be a non-empty path string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestBundleError(f"{role} does not exist: {candidate}") from error
    if not resolved.is_file():
        raise FinalTestBundleError(f"{role} is not a regular file: {resolved}")
    return resolved


def _declared_output_dir(project_root: Path, raw: Any, role: str) -> Path:
    if type(raw) is not str or not raw.strip() or "\x00" in raw:
        raise FinalTestBundleError(f"{role} must be a non-empty path string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return _lexical_absolute(candidate)


def _safe_component(value: str, role: str) -> str:
    if (
        type(value) is not str
        or value in {".", ".."}
        or SAFE_SCALAR_RE.fullmatch(value) is None
    ):
        raise FinalTestBundleError(f"{role} is not a safe path component")
    return value


def expected_run_output_dir(manifest_path: Path, method: str, seed: int) -> Path:
    method_component = _safe_component(method, "method")
    if type(seed) is not int or seed < 0:
        raise FinalTestBundleError("seed is invalid")
    return (
        _lexical_absolute(manifest_path).parent
        / OUTPUT_ROOT_NAME
        / method_component
        / f"seed_{seed}"
    )


def registry_path_for_manifest(manifest_path: Path) -> Path:
    return _lexical_absolute(manifest_path).parent / REGISTRY_BASENAME


def _reject_symlink_components(path: Path, role: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise FinalTestBundleError(
                f"cannot inspect {role} path component {current}: {error}"
            ) from error
        if stat.S_ISLNK(info.st_mode):
            raise FinalTestBundleError(
                f"{role} contains a symbolic-link component: {current}"
            )


def _load_verified_manifest(path: Path) -> tuple[str, dict[str, Any]]:
    manifest_path = _lexical_absolute(path)
    _reject_symlink_components(manifest_path, "frozen manifest")
    digest = manifest_tool.verify_manifest(manifest_path)
    payload = manifest_tool.load_json(manifest_path)
    if _sha256_file(manifest_path) != digest:
        raise FinalTestBundleError("frozen manifest changed after verification")
    canonical_digest = hashlib.sha256(
        manifest_tool.canonical_json_bytes(payload)
    ).hexdigest()
    if canonical_digest != digest:
        raise FinalTestBundleError("frozen manifest digest is not stable")
    return digest, payload


def _registered_runtime_paths(
    payload: Mapping[str, Any], project_root: Path
) -> set[Path]:
    registered: set[Path] = set()
    for index, artifact in enumerate(payload["selection"]["runtime_files"]):
        registered.add(
            _resolve_project_file(
                project_root,
                artifact["path"],
                f"selection.runtime_files[{index}].path",
            )
        )
    return registered


def _require_runtime_files_frozen(
    payload: Mapping[str, Any], project_root: Path
) -> None:
    registered = _registered_runtime_paths(payload, project_root)
    missing = [path for path in REQUIRED_RUNTIME_FILES if path not in registered]
    if missing:
        raise FinalTestBundleError(
            "frozen manifest does not register required runtime files: "
            + ", ".join(str(path) for path in missing)
        )


def _dataset_fields(
    payload: Mapping[str, Any], *, verify_data: bool
) -> tuple[Path, Path, str]:
    try:
        split = payload["split"]
        corruption = payload["corruption"]
        class_order = payload["class_order"]
        annotation_root = payload["annotations"]["root"]
        image_root = payload["images"]["root"]
    except (KeyError, TypeError) as error:
        raise FinalTestBundleError(
            "dataset manifest is missing split/corruption/root fields"
        ) from error
    if split != "test":
        raise FinalTestBundleError("dataset manifest split must equal 'test'")
    if type(class_order) is not list or tuple(class_order) != RSAR_CLASS_ORDER:
        raise FinalTestBundleError(
            "dataset manifest class_order must equal the fixed RSAR order"
        )
    if type(corruption) is not str or SAFE_SCALAR_RE.fullmatch(corruption) is None:
        raise FinalTestBundleError("dataset manifest corruption is not a safe scalar")
    paths: list[Path] = []
    for raw, role in (
        (annotation_root, "annotations.root"),
        (image_root, "images.root"),
    ):
        if type(raw) is not str or not raw.strip() or "\x00" in raw:
            raise FinalTestBundleError(f"dataset manifest {role} is invalid")
        path = Path(raw)
        if not path.is_absolute():
            raise FinalTestBundleError(f"dataset manifest {role} must be absolute")
        lexical = _lexical_absolute(path)
        if verify_data:
            try:
                resolved = lexical.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise FinalTestBundleError(
                    f"dataset manifest {role} does not exist: {lexical}"
                ) from error
            if not resolved.is_dir():
                raise FinalTestBundleError(
                    f"dataset manifest {role} is not a directory: {resolved}"
                )
            lexical = resolved
        if not any(part.casefold() == "test" for part in lexical.parts):
            raise FinalTestBundleError(
                f"dataset manifest {role} lacks an exact 'test' component"
            )
        paths.append(lexical)
    return paths[0], paths[1], corruption


def load_frozen_bundle(manifest_path: Path, *, verify_data: bool) -> FrozenBundle:
    """Verify and materialize one frozen bundle.

    Dataset bytes are intentionally not scanned when ``verify_data`` is false;
    that mode exists only for ``--dry-run``.
    """

    manifest_digest, payload = _load_verified_manifest(manifest_path)
    project_root_raw = payload["project_root"]
    if type(project_root_raw) is not str or not Path(project_root_raw).is_absolute():
        raise FinalTestBundleError("manifest.project_root must be absolute")
    try:
        project_root = Path(project_root_raw).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestBundleError("manifest.project_root does not exist") from error
    if not project_root.is_dir():
        raise FinalTestBundleError("manifest.project_root is not a directory")
    _require_runtime_files_frozen(payload, project_root)

    selection = payload["selection"]
    config_path = _resolve_project_file(
        project_root, selection["config"]["path"], "selection.config.path"
    )
    data_manifest_path = _resolve_project_file(
        project_root,
        selection["data_manifest"]["path"],
        "selection.data_manifest.path",
    )
    declared_data_digest = selection["data_manifest"]["sha256"]
    if type(declared_data_digest) is not str or not SHA256_RE.fullmatch(
        declared_data_digest
    ):
        raise FinalTestBundleError("selection.data_manifest.sha256 is invalid")
    if verify_data:
        data_digest = data_manifest_tool.verify_manifest(data_manifest_path)
        if data_digest != declared_data_digest:
            raise FinalTestBundleError(
                "dataset manifest digest differs from frozen selection"
            )
    else:
        data_digest = declared_data_digest
    data_payload = manifest_tool.load_json(data_manifest_path)
    if _sha256_file(data_manifest_path) != declared_data_digest:
        raise FinalTestBundleError("dataset manifest changed after bundle verification")
    annotation_root, image_root, corruption = _dataset_fields(
        data_payload, verify_data=verify_data
    )

    runs: list[FrozenRun] = []
    for arm_index, arm in enumerate(payload["arms"]):
        method = arm["method"]
        hyperparameters = dict(arm["hyperparameters"])
        declared_corruption = hyperparameters.get("corrupt")
        if declared_corruption is not None and declared_corruption != corruption:
            raise FinalTestBundleError(
                f"arms[{arm_index}].hyperparameters.corrupt differs from "
                "the dataset manifest"
            )
        for run_index, run in enumerate(arm["runs"]):
            checkpoint = run["checkpoint"]
            checkpoint_path = _resolve_project_file(
                project_root,
                checkpoint["path"],
                f"arms[{arm_index}].runs[{run_index}].checkpoint.path",
            )
            if checkpoint_path.name != CHECKPOINT_BASENAME:
                raise FinalTestBundleError(
                    f"checkpoint basename must equal {CHECKPOINT_BASENAME}"
                )
            output_dir = _declared_output_dir(
                project_root,
                run["output_dir"],
                f"arms[{arm_index}].runs[{run_index}].output_dir",
            )
            expected_output = expected_run_output_dir(
                manifest_path, method, run["seed"]
            )
            if output_dir != expected_output:
                raise FinalTestBundleError(
                    "frozen output_dir must use the fixed manifest-relative "
                    f"layout: expected {expected_output}, got {output_dir}"
                )
            for data_root in (annotation_root, image_root):
                if output_dir == data_root or output_dir.is_relative_to(data_root):
                    raise FinalTestBundleError(
                        f"frozen output_dir is inside a dataset root: " f"{output_dir}"
                    )
            runs.append(
                FrozenRun(
                    method=method,
                    seed=run["seed"],
                    training_fingerprint=run["training_fingerprint"],
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint["sha256"],
                    output_dir=output_dir,
                    method_environment=dict(arm["method_environment"]),
                    hyperparameters=hyperparameters,
                )
            )
    if not runs:
        raise FinalTestBundleError("frozen bundle contains no runs")
    return FrozenBundle(
        manifest_path=_lexical_absolute(manifest_path),
        manifest_sha256=manifest_digest,
        payload=payload,
        project_root=project_root,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_digest,
        annotation_root=annotation_root,
        image_root=image_root,
        corruption=corruption,
        runs=tuple(runs),
    )


def build_test_command(
    bundle: FrozenBundle,
    run: FrozenRun,
    python: Path,
    *,
    predictions_path: Optional[Path] = None,
) -> list[str]:
    for path, role in (
        (bundle.annotation_root, "annotation root"),
        (bundle.image_root, "image root"),
    ):
        if any(character in str(path) for character in ("\x00", ",", "\n", "\r")):
            raise FinalTestBundleError(
                f"{role} cannot be encoded safely for cfg-options"
            )
    return [
        str(python),
        str(REQUIRED_RUNTIME_FILES[1]),
        str(bundle.config_path),
        str(run.checkpoint_path),
        "--out",
        str(predictions_path or run.predictions_path),
        "--work-dir",
        str((predictions_path or run.predictions_path).parent),
        "--gpu-ids",
        "0",
        "--cfg-options",
        f"corrupt={bundle.corruption}",
        "model.ema_config=None",
        "model.ema_ckpt=None",
        f"data.test.ann_file={bundle.annotation_root}",
        f"data.test.img_prefix={bundle.image_root}",
        "--eval",
        "mAP",
        "--eval-options",
        "iou_thr=0.5",
        "nproc=4",
    ]


def build_run_environment(
    base: Mapping[str, str], gpu_uuid: str, python: Path
) -> dict[str, str]:
    if (
        type(gpu_uuid) is not str
        or not gpu_uuid
        or any(character in gpu_uuid for character in ("\x00", "\n", "\r"))
    ):
        raise FinalTestBundleError("GPU UUID is invalid")
    python = _lexical_absolute(python).resolve(strict=True)
    conda_prefix = python.parent.parent
    if (
        python.parent.name != "bin"
        or not (conda_prefix / "conda-meta").is_dir()
        or not (conda_prefix / "lib").is_dir()
    ):
        raise FinalTestBundleError(
            f"current Python is not in a verifiable conda prefix: {python}"
        )
    blocked_exact = {
        "PYTHONPATH",
        "PYTHONHOME",
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "IRAOD_PYTHON",
        "IRAOD_CONDA_PREFIX",
        "CUDA_VISIBLE_DEVICES",
        "CONDA_PREFIX",
    }
    environment = {
        str(key): str(value)
        for key, value in base.items()
        if str(key) not in blocked_exact
        and not str(key).startswith("CGA_")
        and not str(key).startswith("SARCLIP_")
    }
    environment.update(DISABLED_CGA_ENV)
    environment.update(
        {
            "CONDA_PREFIX": str(conda_prefix),
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "IRAOD_CONDA_PREFIX": str(conda_prefix),
            "IRAOD_PYTHON": str(python),
            "LD_LIBRARY_PATH": str(conda_prefix / "lib"),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    unexpected = {
        key: value
        for key, value in environment.items()
        if key.startswith("CGA_") and DISABLED_CGA_ENV.get(key) != value
    }
    if unexpected:
        raise FinalTestBundleError(
            f"CGA environment was not disabled completely: {unexpected}"
        )
    return environment


def relevant_environment(environment: Mapping[str, str]) -> dict[str, str]:
    keys = {
        "CUDA_VISIBLE_DEVICES",
        "IRAOD_CONDA_PREFIX",
        "IRAOD_PYTHON",
        "LD_LIBRARY_PATH",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
        *DISABLED_CGA_ENV,
    }
    return {key: environment[key] for key in sorted(keys) if key in environment}


def _reject_duplicate_registry_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalTestBundleError(f"duplicate key in final-test registry: {key!r}")
        result[key] = value
    return result


def _reject_registry_constant(value: str) -> None:
    raise FinalTestBundleError(f"non-finite value in final-test registry: {value}")


def _run_identity(bundle: FrozenBundle, run: FrozenRun) -> str:
    material = (f"{bundle.manifest_sha256}\0{run.method}\0{run.seed}").encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _open_directory(path: Path, role: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot safely open {role} {path}: {error}"
        ) from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise FinalTestBundleError(f"{role} is not a directory: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _require_single_link_regular(
    descriptor: int,
    path: Path,
    *,
    directory_fd: Optional[int] = None,
    name: Optional[str] = None,
) -> os.stat_result:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise FinalTestBundleError(
            f"file must be regular with exactly one link: {path}"
        )
    if directory_fd is not None and name is not None:
        try:
            path_info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as error:
            raise FinalTestBundleError(
                f"cannot revalidate opened path {path}: {error}"
            ) from error
        if (
            path_info.st_dev != info.st_dev
            or path_info.st_ino != info.st_ino
            or not stat.S_ISREG(path_info.st_mode)
            or path_info.st_nlink != 1
        ):
            raise FinalTestBundleError(
                f"opened file identity changed or is multiply linked: {path}"
            )
    return info


def _ensure_directory_chain(base: Path, components: Sequence[str]) -> Path:
    """Create/open a relative directory chain using no-follow dirfds."""

    _reject_symlink_components(base, "fixed directory root")
    descriptor = _open_directory(base, "fixed directory root")
    current = base
    try:
        for raw_component in components:
            component = _safe_component(raw_component, "directory component")
            created = False
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
                created = True
            except FileExistsError:
                pass
            if created:
                os.fsync(descriptor)
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            child = os.open(component, flags, dir_fd=descriptor)
            if not stat.S_ISDIR(os.fstat(child).st_mode):
                os.close(child)
                raise FinalTestBundleError(
                    f"directory component is not a directory: {component}"
                )
            os.close(descriptor)
            descriptor = child
            current = current / component
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot create fixed directory chain below {base}: {error}"
        ) from error
    finally:
        os.close(descriptor)
    return current


class RunLease:
    """Exclusive per-run flock inherited by the model subprocess."""

    def __init__(self, bundle: FrozenBundle, run: FrozenRun) -> None:
        self.run_identity = _run_identity(bundle, run)
        lock_root = _ensure_directory_chain(
            bundle.manifest_path.parent, (LOCK_ROOT_NAME, "runs")
        )
        self.path = lock_root / f"{self.run_identity}.lock"
        root_fd = _open_directory(lock_root, "final-test lock root")
        flags = os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        self.fd = -1
        try:
            try:
                self.fd = os.open(
                    self.path.name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=root_fd,
                )
                os.fsync(root_fd)
            except FileExistsError:
                self.fd = os.open(self.path.name, flags, dir_fd=root_fd)
            _require_single_link_regular(
                self.fd,
                self.path,
                directory_fd=root_fd,
                name=self.path.name,
            )
        except OSError as error:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise FinalTestBundleError(
                f"cannot safely open run lease {self.path}: {error}"
            ) from error
        except Exception:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise
        finally:
            os.close(root_fd)
        self.acquired = False

    def acquire(self) -> bool:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.acquired = True
        os.set_inheritable(self.fd, True)
        payload = _canonical_json_bytes(
            {
                "pid": os.getpid(),
                "run_identity": self.run_identity,
                "updated_at_utc": _utc_now(),
            }
        )
        try:
            os.ftruncate(self.fd, 0)
            os.lseek(self.fd, 0, os.SEEK_SET)
            offset = 0
            while offset < len(payload):
                written = os.write(self.fd, payload[offset:])
                if written <= 0:
                    raise FinalTestBundleError("short write to run lease")
                offset += written
            os.fsync(self.fd)
        except BaseException:
            self.close()
            raise
        return True

    def close(self) -> None:
        if self.fd < 0:
            return
        # Do not issue LOCK_UN here.  The child inherits this same open-file
        # description.  Closing only the parent's descriptor preserves the
        # flock while a child remains alive after a runner crash/interrupt.
        os.close(self.fd)
        self.fd = -1
        self.acquired = False

    def __enter__(self) -> "RunLease":
        if not self.acquire():
            self.close()
            raise FinalTestBundleError(f"run lease is already held: {self.path}")
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class GPUExecutionLease:
    """Secure GPU flock inherited by the launched test process."""

    def __init__(self, lock_root: Path, gpu_uuid: str) -> None:
        if type(gpu_uuid) is not str or not gpu_uuid:
            raise FinalTestBundleError("GPU UUID is invalid for locking")
        filename = hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest() + ".lock"
        self.path = lock_root / filename
        self.gpu_uuid = gpu_uuid
        self.fd = -1
        root_fd = _open_directory(lock_root, "GPU lock root")
        flags = os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            try:
                self.fd = os.open(
                    self.path.name,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=root_fd,
                )
                os.fsync(root_fd)
            except FileExistsError:
                self.fd = os.open(self.path.name, flags, dir_fd=root_fd)
            _require_single_link_regular(
                self.fd,
                self.path,
                directory_fd=root_fd,
                name=self.path.name,
            )
        except OSError as error:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise FinalTestBundleError(
                f"cannot safely open GPU lease {self.path}: {error}"
            ) from error
        except Exception:
            if self.fd >= 0:
                os.close(self.fd)
                self.fd = -1
            raise
        finally:
            os.close(root_fd)
        self.acquired = False

    def acquire(self) -> bool:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self.acquired = True
        os.set_inheritable(self.fd, True)
        payload = _canonical_json_bytes(
            {
                "gpu_uuid": self.gpu_uuid,
                "pid": os.getpid(),
                "updated_at_utc": _utc_now(),
            }
        )
        try:
            os.ftruncate(self.fd, 0)
            os.lseek(self.fd, 0, os.SEEK_SET)
            offset = 0
            while offset < len(payload):
                written = os.write(self.fd, payload[offset:])
                if written <= 0:
                    raise FinalTestBundleError("short write to GPU lease")
                offset += written
            os.fsync(self.fd)
        except BaseException:
            self.close()
            raise
        return True

    def close(self) -> None:
        if self.fd < 0:
            return
        os.close(self.fd)
        self.fd = -1
        self.acquired = False

    def release(self) -> None:
        self.close()


@dataclasses.dataclass(frozen=True)
class AttemptPaths:
    number: int
    directory: Path
    predictions: Path
    stdout: Path
    online_metadata: Path
    terminal: Path


def _create_attempt_paths(
    bundle: FrozenBundle, run: FrozenRun, attempt: int
) -> AttemptPaths:
    run_root = _ensure_directory_chain(
        bundle.manifest_path.parent,
        (
            OUTPUT_ROOT_NAME,
            _safe_component(run.method, "method"),
            f"seed_{run.seed}",
            ATTEMPTS_DIRNAME,
        ),
    )
    if run_root.parent != run.output_dir:
        raise FinalTestBundleError("attempt root differs from frozen output_dir")
    attempt_name = f"attempt_{attempt:03d}"
    parent_fd = _open_directory(run_root, "attempt root")
    try:
        os.mkdir(attempt_name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError as error:
        raise FinalTestBundleError(
            f"attempt directory already exists: {run_root / attempt_name}"
        ) from error
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot create attempt directory: {error}"
        ) from error
    finally:
        os.close(parent_fd)
    directory = run_root / attempt_name
    return AttemptPaths(
        number=attempt,
        directory=directory,
        predictions=directory / PREDICTIONS_BASENAME,
        stdout=directory / STDOUT_BASENAME,
        online_metadata=directory / ONLINE_METADATA_BASENAME,
        terminal=directory / ATTEMPT_TERMINAL_BASENAME,
    )


class AppendOnlyRegistry:
    """Strict JSONL registry that is never rewritten or truncated."""

    REQUIRED_FIELDS = {
        "schema_version",
        "event",
        "recorded_at_utc",
        "run_identity",
        "manifest_path",
        "manifest_sha256",
        "method",
        "seed",
        "training_fingerprint",
        "checkpoint_path",
        "checkpoint_sha256",
        "output_dir",
        "attempt",
        "attempt_dir",
        "lease_path",
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "free_memory_mib_at_start",
        "pid",
        "pid_start_ticks",
        "gpu_verified",
        "command",
        "environment",
        "returncode",
        "artifacts",
        "failure",
    }

    def __init__(self, path: Path) -> None:
        self.path = _lexical_absolute(path)
        self._lock = threading.Lock()
        _reject_symlink_components(self.path.parent, "registry parent")
        if self.path.exists() and self.path.is_symlink():
            raise FinalTestBundleError(f"registry must not be a symlink: {self.path}")

    @classmethod
    def validate_record(cls, record: Any, location: str) -> dict[str, Any]:
        if type(record) is not dict:
            raise FinalTestBundleError(f"{location} must be a JSON object")
        keys = set(record)
        if keys != cls.REQUIRED_FIELDS:
            raise FinalTestBundleError(
                f"{location} fields differ from registry schema: "
                f"missing={sorted(cls.REQUIRED_FIELDS - keys)}, "
                f"extra={sorted(keys - cls.REQUIRED_FIELDS)}"
            )
        if record["schema_version"] != REGISTRY_SCHEMA_VERSION:
            raise FinalTestBundleError(f"{location} has unsupported schema_version")
        if record["event"] not in REGISTRY_EVENTS:
            raise FinalTestBundleError(f"{location} has invalid event")
        if type(record["attempt"]) is not int or not 1 <= record["attempt"] <= 3:
            raise FinalTestBundleError(f"{location} has invalid attempt")
        if type(record["seed"]) is not int or record["seed"] < 0:
            raise FinalTestBundleError(f"{location} has invalid seed")
        for field in (
            "run_identity",
            "manifest_sha256",
            "training_fingerprint",
            "checkpoint_sha256",
        ):
            if (
                type(record[field]) is not str
                or SHA256_RE.fullmatch(record[field]) is None
            ):
                raise FinalTestBundleError(f"{location}.{field} is invalid")
        for field in (
            "recorded_at_utc",
            "manifest_path",
            "method",
            "checkpoint_path",
            "output_dir",
            "attempt_dir",
            "lease_path",
            "gpu_uuid",
            "gpu_name",
        ):
            if type(record[field]) is not str or not record[field]:
                raise FinalTestBundleError(f"{location}.{field} is invalid")
        for field in (
            "gpu_index",
            "free_memory_mib_at_start",
        ):
            if type(record[field]) is not int or record[field] < 0:
                raise FinalTestBundleError(f"{location}.{field} is invalid")
        if type(record["gpu_verified"]) is not bool:
            raise FinalTestBundleError(f"{location}.gpu_verified must be boolean")
        if type(record["command"]) is not list or not all(
            type(item) is str for item in record["command"]
        ):
            raise FinalTestBundleError(f"{location}.command must be a string array")
        if type(record["environment"]) is not dict:
            raise FinalTestBundleError(f"{location}.environment must be an object")
        if type(record["artifacts"]) is not dict:
            raise FinalTestBundleError(f"{location}.artifacts must be an object")
        cga_environment = {
            key: value
            for key, value in record["environment"].items()
            if key.startswith("CGA_")
        }
        if cga_environment != DISABLED_CGA_ENV:
            raise FinalTestBundleError(
                f"{location}.environment does not disable all CGA settings"
            )
        for name, identity in record["artifacts"].items():
            if name not in CORE_OUTPUT_NAMES or type(identity) is not dict:
                raise FinalTestBundleError(
                    f"{location}.artifacts contains an invalid entry"
                )
            if set(identity) != {"path", "sha256", "size_bytes"}:
                raise FinalTestBundleError(
                    f"{location}.artifacts[{name!r}] has invalid fields"
                )
            if (
                type(identity["path"]) is not str
                or not Path(identity["path"]).is_absolute()
                or type(identity["sha256"]) is not str
                or SHA256_RE.fullmatch(identity["sha256"]) is None
                or type(identity["size_bytes"]) is not int
                or identity["size_bytes"] < 0
            ):
                raise FinalTestBundleError(
                    f"{location}.artifacts[{name!r}] identity is invalid"
                )
        event = record["event"]
        if event == "started":
            if (
                type(record["pid"]) is not int
                or record["pid"] <= 0
                or type(record["pid_start_ticks"]) is not int
                or record["pid_start_ticks"] <= 0
                or record["gpu_verified"]
                or record["returncode"] is not None
                or record["artifacts"]
                or record["failure"] is not None
            ):
                raise FinalTestBundleError(
                    f"{location} has invalid started-event fields"
                )
        elif event == "completed":
            if (
                type(record["pid"]) is not int
                or record["pid"] <= 0
                or type(record["pid_start_ticks"]) is not int
                or record["pid_start_ticks"] <= 0
                or not record["gpu_verified"]
                or record["returncode"] != 0
                or set(record["artifacts"]) != CORE_OUTPUT_NAMES
                or record["failure"] is not None
                or any(
                    identity["size_bytes"] <= 0
                    for identity in record["artifacts"].values()
                )
            ):
                raise FinalTestBundleError(
                    f"{location} has invalid completed-event fields"
                )
        else:
            pid = record["pid"]
            ticks = record["pid_start_ticks"]
            if (pid is None) != (ticks is None):
                raise FinalTestBundleError(
                    f"{location} has incomplete failed PID identity"
                )
            if pid is not None and (
                type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0
            ):
                raise FinalTestBundleError(
                    f"{location} has invalid failed PID identity"
                )
            if (
                (
                    record["returncode"] is not None
                    and type(record["returncode"]) is not int
                )
                or type(record["failure"]) is not str
                or not record["failure"]
            ):
                raise FinalTestBundleError(
                    f"{location} has invalid failed-event fields"
                )
        return record

    def read(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        parent_fd = _open_directory(self.path.parent, "registry parent")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            try:
                descriptor = os.open(self.path.name, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                return []
            try:
                _require_single_link_regular(
                    descriptor,
                    self.path,
                    directory_fd=parent_fd,
                    name=self.path.name,
                )
            except Exception:
                os.close(descriptor)
                raise
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                for line_number, raw in enumerate(handle, start=1):
                    if not raw.endswith("\n"):
                        raise FinalTestBundleError(
                            "registry line " f"{line_number} is not newline terminated"
                        )
                    try:
                        record = json.loads(
                            raw,
                            object_pairs_hook=_reject_duplicate_registry_keys,
                            parse_constant=_reject_registry_constant,
                        )
                    except FinalTestBundleError:
                        raise
                    except json.JSONDecodeError as error:
                        raise FinalTestBundleError(
                            "registry line " f"{line_number} is invalid JSON: {error}"
                        ) from error
                    records.append(
                        self.validate_record(record, f"registry line {line_number}")
                    )
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            raise FinalTestBundleError(f"cannot read registry: {error}") from error
        finally:
            os.close(parent_fd)
        return records

    def append(self, record: Mapping[str, Any]) -> None:
        validated = self.validate_record(dict(record), "new registry record")
        content = _canonical_json_bytes(validated)
        flags = os.O_WRONLY | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        with self._lock:
            parent_fd = _open_directory(self.path.parent, "registry parent")
            try:
                created = False
                try:
                    descriptor = os.open(
                        self.path.name,
                        flags | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent_fd,
                    )
                    created = True
                except FileExistsError:
                    descriptor = os.open(self.path.name, flags, dir_fd=parent_fd)
                try:
                    _require_single_link_regular(
                        descriptor,
                        self.path,
                        directory_fd=parent_fd,
                        name=self.path.name,
                    )
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    written = os.write(descriptor, content)
                    if written != len(content):
                        raise FinalTestBundleError(
                            "short append to final-test registry"
                        )
                    os.fsync(descriptor)
                    if created:
                        os.fsync(parent_fd)
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)
            except OSError as error:
                raise FinalTestBundleError(
                    f"cannot append final-test registry {self.path}: {error}"
                ) from error
            finally:
                os.close(parent_fd)


def _records_for_run(
    records: Sequence[Mapping[str, Any]], bundle: FrozenBundle, run: FrozenRun
) -> list[Mapping[str, Any]]:
    identity = _run_identity(bundle, run)
    selected: list[Mapping[str, Any]] = []
    for record in records:
        same_coordinates = (
            record["manifest_sha256"] == bundle.manifest_sha256
            and record["method"] == run.method
            and record["seed"] == run.seed
        )
        if record["run_identity"] == identity or same_coordinates:
            _validate_record_binding(record, bundle, run)
            selected.append(record)
    return selected


def _validate_bundle_registry_records(
    records: Sequence[Mapping[str, Any]], bundle: FrozenBundle
) -> None:
    runs_by_identity = {_run_identity(bundle, run): run for run in bundle.runs}
    runs_by_coordinates = {(run.method, run.seed): run for run in bundle.runs}
    if len(runs_by_identity) != len(bundle.runs) or len(runs_by_coordinates) != len(
        bundle.runs
    ):
        raise FinalTestBundleError("frozen bundle run identities are not unique")
    manifest_path = str(bundle.manifest_path)
    for record in records:
        identity_match = record["run_identity"] in runs_by_identity
        path_match = record["manifest_path"] == manifest_path
        digest_match = record["manifest_sha256"] == bundle.manifest_sha256
        if not (identity_match or path_match or digest_match):
            continue
        if not path_match or not digest_match:
            raise FinalTestBundleError(
                "registry record partially matches the current frozen manifest"
            )
        identity_run = runs_by_identity.get(record["run_identity"])
        coordinate_run = runs_by_coordinates.get((record["method"], record["seed"]))
        if identity_run is None or identity_run != coordinate_run:
            raise FinalTestBundleError(
                "registry record does not map to exactly one frozen run"
            )
        _validate_record_binding(record, bundle, identity_run)


def _read_validated_bundle_registry(
    registry: AppendOnlyRegistry, bundle: FrozenBundle
) -> list[dict[str, Any]]:
    _validate_registry_path(registry, bundle)
    records = registry.read()
    _validate_bundle_registry_records(records, bundle)
    return records


def _validate_record_binding(
    record: Mapping[str, Any], bundle: FrozenBundle, run: FrozenRun
) -> None:
    expected = {
        "run_identity": _run_identity(bundle, run),
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "method": run.method,
        "seed": run.seed,
        "training_fingerprint": run.training_fingerprint,
        "checkpoint_path": str(run.checkpoint_path),
        "checkpoint_sha256": run.checkpoint_sha256,
        "output_dir": str(run.output_dir),
        "lease_path": str(_expected_lease_path(bundle, run)),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise FinalTestBundleError(
                f"registry {key} mismatch for " f"{run.method}/seed_{run.seed}"
            )
    attempt_dir = _attempt_paths(run, record["attempt"]).directory
    if record.get("attempt_dir") != str(attempt_dir):
        raise FinalTestBundleError("registry attempt_dir mismatch")
    command = record.get("command")
    if not command:
        raise FinalTestBundleError("registry command is empty")
    try:
        command_python = Path(command[0]).resolve(strict=True)
        current_python = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestBundleError(
            f"registry Python cannot be resolved: {error}"
        ) from error
    if command_python != current_python or command[0] != str(current_python):
        raise FinalTestBundleError(
            "registry command Python differs from current sys.executable"
        )
    expected_command = build_test_command(
        bundle,
        run,
        command_python,
        predictions_path=attempt_dir / PREDICTIONS_BASENAME,
    )
    if command != expected_command:
        raise FinalTestBundleError("registry command mismatch")
    if record["environment"].get("CUDA_VISIBLE_DEVICES") != record["gpu_uuid"]:
        raise FinalTestBundleError(
            "registry CUDA_VISIBLE_DEVICES differs from gpu_uuid"
        )
    expected_environment = relevant_environment(
        build_run_environment({}, record["gpu_uuid"], command_python)
    )
    if record["environment"] != expected_environment:
        raise FinalTestBundleError(
            "registry environment differs from deterministic runtime values"
        )


def _attempt_count(records: Sequence[Mapping[str, Any]]) -> int:
    return len({record["attempt"] for record in records})


def _attempt_paths(run: FrozenRun, attempt: int) -> AttemptPaths:
    directory = run.output_dir / ATTEMPTS_DIRNAME / f"attempt_{attempt:03d}"
    return AttemptPaths(
        number=attempt,
        directory=directory,
        predictions=directory / PREDICTIONS_BASENAME,
        stdout=directory / STDOUT_BASENAME,
        online_metadata=directory / ONLINE_METADATA_BASENAME,
        terminal=directory / ATTEMPT_TERMINAL_BASENAME,
    )


def _expected_lease_path(bundle: FrozenBundle, run: FrozenRun) -> Path:
    return (
        bundle.manifest_path.parent
        / LOCK_ROOT_NAME
        / "runs"
        / f"{_run_identity(bundle, run)}.lock"
    )


def _inspect_output_dir(path: Path) -> set[str]:
    _reject_symlink_components(path, "frozen output_dir")
    if not path.exists():
        return set()
    try:
        info = path.lstat()
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot inspect output_dir {path}: {error}"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise FinalTestBundleError(
            f"output_dir must be a non-symlink directory: {path}"
        )
    names: set[str] = set()
    try:
        entries = list(os.scandir(path))
    except OSError as error:
        raise FinalTestBundleError(f"cannot scan output_dir {path}: {error}") from error
    for entry in entries:
        if entry.name not in ALLOWED_RUN_OUTPUT_NAMES:
            raise FinalTestBundleError(
                f"output_dir contains unknown entry: {Path(entry.path)}"
            )
        info = entry.stat(follow_symlinks=False)
        if entry.name == ATTEMPTS_DIRNAME:
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise FinalTestBundleError(
                    "attempts entry must be a non-symlink directory: " f"{entry.path}"
                )
            _inspect_attempt_directories(path)
        elif (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise FinalTestBundleError(
                "output_dir entry must be a regular single-link file: " f"{entry.path}"
            )
        names.add(entry.name)
    return names


def _inspect_attempt_directories(run_output_dir: Path) -> dict[int, Path]:
    attempt_root = run_output_dir / ATTEMPTS_DIRNAME
    if not attempt_root.exists():
        return {}
    root_fd = _open_directory(attempt_root, "attempt root")
    os.close(root_fd)
    result: dict[int, Path] = {}
    try:
        entries = list(os.scandir(attempt_root))
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot scan attempt root {attempt_root}: {error}"
        ) from error
    for entry in entries:
        match = re.fullmatch(r"attempt_([0-9]{3})", entry.name)
        if match is None:
            raise FinalTestBundleError(
                f"attempt root contains unknown entry: {entry.path}"
            )
        attempt = int(match.group(1))
        if not 1 <= attempt <= MAX_ATTEMPTS or attempt in result:
            raise FinalTestBundleError(
                f"attempt directory number is invalid: {entry.path}"
            )
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise FinalTestBundleError(
                f"attempt entry must be a non-symlink directory: {entry.path}"
            )
        _inspect_attempt_directory(Path(entry.path))
        result[attempt] = Path(entry.path)
    return result


def _inspect_attempt_directory(path: Path) -> set[str]:
    descriptor = _open_directory(path, "attempt directory")
    os.close(descriptor)
    names: set[str] = set()
    try:
        entries = list(os.scandir(path))
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot scan attempt directory {path}: {error}"
        ) from error
    for entry in entries:
        if (
            entry.name not in ALLOWED_ATTEMPT_NAMES
            and FRAMEWORK_EVAL_RE.fullmatch(entry.name) is None
        ):
            raise FinalTestBundleError(
                f"attempt directory contains unknown entry: {entry.path}"
            )
        info = entry.stat(follow_symlinks=False)
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
        ):
            raise FinalTestBundleError(
                "attempt artifact must be regular with exactly one link: "
                f"{entry.path}"
            )
        names.add(entry.name)
    return names


def _exclusive_create_fd(directory_fd: int, name: str, *, mode: int = 0o600) -> int:
    _safe_component(name, "output filename")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        return os.open(name, flags, mode, dir_fd=directory_fd)
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot exclusively create {name}: {error}"
        ) from error


def _exclusive_write_json_at(
    directory: Path, name: str, payload: Mapping[str, Any]
) -> None:
    directory_fd = _open_directory(directory, "attempt directory")
    descriptor: Optional[int] = None
    try:
        descriptor = _exclusive_create_fd(directory_fd, name)
        _require_single_link_regular(
            descriptor,
            directory / name,
            directory_fd=directory_fd,
            name=name,
        )
        content = _canonical_json_bytes(payload)
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise FinalTestBundleError(f"short write while creating {name}")
            offset += written
        os.fsync(descriptor)
        os.fsync(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _file_identity_at(
    directory: Path,
    name: str,
    *,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    directory_fd = _open_directory(directory, "artifact directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        _require_single_link_regular(
            descriptor,
            directory / name,
            directory_fd=directory_fd,
            name=name,
        )
        return _file_identity_from_descriptor(
            descriptor,
            directory / name,
            require_nonempty=require_nonempty,
        )
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot safely inspect {directory / name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


def _file_identity_from_descriptor(
    descriptor: int,
    path: Path,
    *,
    require_nonempty: bool = True,
    require_single_link: bool = True,
) -> dict[str, Any]:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or (require_single_link and info.st_nlink != 1):
        raise FinalTestBundleError(
            f"artifact must be regular with exactly one link: {path}"
        )
    if require_nonempty and info.st_size <= 0:
        raise FinalTestBundleError(f"artifact must not be empty: {path}")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    if size != info.st_size:
        raise FinalTestBundleError(f"artifact changed while hashing: {path}")
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _identity_at_dirfd(
    directory_fd: int,
    directory: Path,
    name: str,
    *,
    require_nonempty: bool = True,
) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        _require_single_link_regular(
            descriptor,
            directory / name,
            directory_fd=directory_fd,
            name=name,
        )
        return _file_identity_from_descriptor(
            descriptor,
            directory / name,
            require_nonempty=require_nonempty,
        )
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot safely inspect {directory / name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _copy_to_publication_staging(
    source_descriptor: int,
    staging_descriptor: int,
    *,
    source_name: str,
    destination_name: str,
    expected_identity: Mapping[str, Any],
) -> tuple[str, int]:
    os.lseek(source_descriptor, 0, os.SEEK_SET)
    copied_digest = hashlib.sha256()
    copied_size = 0
    while True:
        chunk = os.read(source_descriptor, 1024 * 1024)
        if not chunk:
            break
        copied_digest.update(chunk)
        copied_size += len(chunk)
        offset = 0
        while offset < len(chunk):
            written = os.write(staging_descriptor, chunk[offset:])
            if written <= 0:
                raise FinalTestBundleError(
                    f"short canonical staging write: {destination_name}"
                )
            offset += written
    copied = copied_digest.hexdigest(), copied_size
    if copied != _identity_value(expected_identity):
        raise FinalTestBundleError(
            f"attempt artifact changed while staging: {source_name}"
        )
    return copied


def _verify_existing_canonical(
    directory_fd: int,
    directory: Path,
    destination_name: str,
    expected_identity: Mapping[str, Any],
) -> bool:
    """Return whether an independently published canonical file is valid."""

    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    read_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        try:
            descriptor = os.open(destination_name, read_flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return False
        _require_single_link_regular(
            descriptor,
            directory / destination_name,
            directory_fd=directory_fd,
            name=destination_name,
        )
        current = _file_identity_from_descriptor(
            descriptor,
            directory / destination_name,
        )
        if _identity_value(current) != _identity_value(expected_identity):
            raise FinalTestBundleError(
                "canonical output already exists with drift: " f"{destination_name}"
            )
        return True
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot safely verify canonical output {destination_name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_anonymous_publication_staging(directory_fd: int) -> int:
    tmpfile_flag = getattr(os, "O_TMPFILE", 0)
    if not tmpfile_flag:
        raise FinalTestBundleError(
            "this platform does not support anonymous O_TMPFILE publication"
        )
    flags = os.O_RDWR | tmpfile_flag | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(".", flags, 0o400, dir_fd=directory_fd)
    except OSError as error:
        raise FinalTestBundleError(
            "canonical output filesystem does not support secure anonymous "
            f"publication staging: {error}"
        ) from error
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 0:
        os.close(descriptor)
        raise FinalTestBundleError(
            "anonymous publication staging has an unsafe file identity"
        )
    return descriptor


def _preflight_anonymous_publication(directory: Path) -> None:
    directory_fd = _open_directory(directory, "canonical output directory")
    staging_descriptor: Optional[int] = None
    try:
        staging_descriptor = _open_anonymous_publication_staging(directory_fd)
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(directory_fd)


def _link_anonymous_file_noreplace(
    staging_descriptor: int,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically expose an O_TMPFILE inode without replacing any path."""

    os.link(
        f"/proc/self/fd/{staging_descriptor}",
        destination_name,
        dst_dir_fd=destination_directory_fd,
        follow_symlinks=True,
    )


def _publish_exclusive_copy(
    source_directory: Path,
    destination_directory: Path,
    source_name: str,
    destination_name: str,
    expected_identity: Mapping[str, Any],
) -> None:
    source_fd = _open_directory(source_directory, "attempt directory")
    destination_fd = _open_directory(
        destination_directory, "canonical output directory"
    )
    source_descriptor: Optional[int] = None
    staging_descriptor: Optional[int] = None
    try:
        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        read_flags |= getattr(os, "O_NOFOLLOW", 0)
        source_descriptor = os.open(source_name, read_flags, dir_fd=source_fd)
        _require_single_link_regular(
            source_descriptor,
            source_directory / source_name,
            directory_fd=source_fd,
            name=source_name,
        )
        current_source = _file_identity_from_descriptor(
            source_descriptor, source_directory / source_name
        )
        if _identity_value(current_source) != _identity_value(expected_identity):
            raise FinalTestBundleError(
                "attempt artifact changed before publication: " f"{source_name}"
            )
        if _verify_existing_canonical(
            destination_fd,
            destination_directory,
            destination_name,
            expected_identity,
        ):
            return

        staging_descriptor = _open_anonymous_publication_staging(destination_fd)
        copied = _copy_to_publication_staging(
            source_descriptor,
            staging_descriptor,
            source_name=source_name,
            destination_name=destination_name,
            expected_identity=expected_identity,
        )
        os.fsync(staging_descriptor)
        staged_info = os.fstat(staging_descriptor)
        if not stat.S_ISREG(staged_info.st_mode) or staged_info.st_nlink != 0:
            raise FinalTestBundleError(
                "anonymous staging identity changed before publication"
            )
        staged = _file_identity_from_descriptor(
            staging_descriptor,
            destination_directory / f"<anonymous:{destination_name}>",
            require_single_link=False,
        )
        if _identity_value(staged) != copied:
            raise FinalTestBundleError(
                f"canonical staging differs after copy: {destination_name}"
            )
        try:
            _link_anonymous_file_noreplace(
                staging_descriptor, destination_fd, destination_name
            )
        except FileExistsError:
            if not _verify_existing_canonical(
                destination_fd,
                destination_directory,
                destination_name,
                expected_identity,
            ):
                raise FinalTestBundleError(
                    f"canonical output disappeared during publication: "
                    f"{destination_name}"
                )
            return
        os.fsync(destination_fd)
        published_info = os.fstat(staging_descriptor)
        if published_info.st_nlink != 1:
            raise FinalTestBundleError("anonymous staging was not linked exactly once")
        published = _identity_at_dirfd(
            destination_fd, destination_directory, destination_name
        )
        if _identity_value(published) != copied:
            raise FinalTestBundleError(
                f"canonical output differs after publication: {destination_name}"
            )
    except OSError as error:
        raise FinalTestBundleError(
            f"secure canonical publication failed: {error}"
        ) from error
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        os.close(source_fd)
        os.close(destination_fd)


def _identity_value(identity: Mapping[str, Any]) -> tuple[str, int]:
    return identity["sha256"], identity["size_bytes"]


def _canonical_artifact_identities(run: FrozenRun) -> dict[str, Any]:
    return {
        name: _file_identity_at(run.output_dir, name)
        for name in sorted(CORE_OUTPUT_NAMES)
    }


def _framework_eval_names(directory: Path) -> list[str]:
    names = _inspect_attempt_directory(directory)
    matches = sorted(name for name in names if FRAMEWORK_EVAL_RE.fullmatch(name))
    if len(matches) > 1:
        raise FinalTestBundleError(
            "attempt directory contains multiple framework eval JSON files"
        )
    return matches


def _validate_framework_eval_json(directory: Path, name: str) -> None:
    directory_fd = _open_directory(directory, "framework eval directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        _require_single_link_regular(
            descriptor,
            directory / name,
            directory_fd=directory_fd,
            name=name,
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot safely read framework eval JSON: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_registry_keys,
            parse_constant=_reject_registry_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalTestBundleError(
            f"framework eval evidence is invalid JSON: {error}"
        ) from error
    if type(payload) is not dict or not payload:
        raise FinalTestBundleError("framework eval JSON must be a nonempty object")


def _partial_attempt_artifacts(paths: AttemptPaths) -> dict[str, Any]:
    names = _inspect_attempt_directory(paths.directory)
    artifacts = {
        name: _file_identity_at(paths.directory, name, require_nonempty=False)
        for name in sorted(names & ATTEMPT_FIXED_ARTIFACT_NAMES)
    }
    framework_names = _framework_eval_names(paths.directory)
    if framework_names:
        artifacts[FRAMEWORK_EVAL_BASENAME] = _file_identity_at(
            paths.directory,
            framework_names[0],
            require_nonempty=False,
        )
    return artifacts


def _successful_attempt_base_artifacts(
    paths: AttemptPaths,
) -> dict[str, Any]:
    framework_names = _framework_eval_names(paths.directory)
    if len(framework_names) != 1:
        raise FinalTestBundleError(
            "successful attempt must create exactly one eval_*.json"
        )
    framework_name = framework_names[0]
    _validate_framework_eval_json(paths.directory, framework_name)
    return {
        PREDICTIONS_BASENAME: _file_identity_at(paths.directory, PREDICTIONS_BASENAME),
        STDOUT_BASENAME: _file_identity_at(paths.directory, STDOUT_BASENAME),
        FRAMEWORK_EVAL_BASENAME: _file_identity_at(paths.directory, framework_name),
    }


def _fsync_attempt_artifacts(paths: AttemptPaths) -> None:
    names = _inspect_attempt_directory(paths.directory)
    directory_fd = _open_directory(paths.directory, "attempt directory")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        for name in sorted(names - {ATTEMPT_TERMINAL_BASENAME}):
            descriptor = os.open(name, flags, dir_fd=directory_fd)
            try:
                _require_single_link_regular(
                    descriptor,
                    paths.directory / name,
                    directory_fd=directory_fd,
                    name=name,
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory_fd)
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot fsync attempt evidence {paths.directory}: {error}"
        ) from error
    finally:
        os.close(directory_fd)


def _read_json_at(directory: Path, name: str) -> dict[str, Any]:
    directory_fd = _open_directory(directory, "JSON parent")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        _require_single_link_regular(
            descriptor,
            directory / name,
            directory_fd=directory_fd,
            name=name,
        )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot safely read {directory / name}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_registry_keys,
            parse_constant=_reject_registry_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FinalTestBundleError(
            f"invalid JSON evidence {directory / name}: {error}"
        ) from error
    if type(payload) is not dict:
        raise FinalTestBundleError(
            f"JSON evidence must be an object: {directory / name}"
        )
    if raw != _canonical_json_bytes(payload):
        raise FinalTestBundleError(
            f"JSON evidence is not canonical: {directory / name}"
        )
    return payload


def _terminal_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("schema_version")
    payload["terminal_schema_version"] = ATTEMPT_TERMINAL_SCHEMA_VERSION
    return payload


def _record_from_terminal(
    payload: Mapping[str, Any], *, artifacts: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    record = dict(payload)
    if record.pop("terminal_schema_version", None) != (ATTEMPT_TERMINAL_SCHEMA_VERSION):
        raise FinalTestBundleError("attempt terminal schema is unsupported")
    record["schema_version"] = REGISTRY_SCHEMA_VERSION
    if artifacts is not None:
        record["artifacts"] = dict(artifacts)
    return AppendOnlyRegistry.validate_record(record, "attempt terminal")


def _load_attempt_terminal(
    bundle: FrozenBundle, run: FrozenRun, paths: AttemptPaths
) -> dict[str, Any]:
    payload = _read_json_at(paths.directory, ATTEMPT_TERMINAL_BASENAME)
    record = _record_from_terminal(payload)
    if record["event"] == "started":
        raise FinalTestBundleError("attempt terminal cannot be a started event")
    _validate_record_binding(record, bundle, run)
    actual = _partial_attempt_artifacts(paths)
    if actual != record["artifacts"]:
        raise FinalTestBundleError(
            f"attempt_{paths.number:03d} artifact identity mismatch"
        )
    return record


def _validate_online_metadata(
    paths: AttemptPaths,
    bundle: FrozenBundle,
    run: FrozenRun,
    terminal_record: Mapping[str, Any],
) -> None:
    payload = _read_json_at(paths.directory, ONLINE_METADATA_BASENAME)
    required = {
        "schema_version",
        "status",
        "completed_at_utc",
        "run_identity",
        "manifest_path",
        "manifest_sha256",
        "data_manifest_path",
        "data_manifest_sha256",
        "method",
        "seed",
        "training_fingerprint",
        "checkpoint_path",
        "checkpoint_sha256",
        "output_dir",
        "attempt",
        "attempt_dir",
        "lease_path",
        "class_order",
        "gpu",
        "pid",
        "pid_start_ticks",
        "gpu_verified",
        "command",
        "environment",
        "returncode",
        "artifacts",
        "metrics_parsed_by_runner",
    }
    if set(payload) != required:
        raise FinalTestBundleError("online_eval.json fields differ from schema")
    expected = {
        "schema_version": ONLINE_METADATA_SCHEMA_VERSION,
        "status": "completed",
        "run_identity": _run_identity(bundle, run),
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "data_manifest_path": str(bundle.data_manifest_path),
        "data_manifest_sha256": bundle.data_manifest_sha256,
        "method": run.method,
        "seed": run.seed,
        "training_fingerprint": run.training_fingerprint,
        "checkpoint_path": str(run.checkpoint_path),
        "checkpoint_sha256": run.checkpoint_sha256,
        "output_dir": str(run.output_dir),
        "attempt": terminal_record["attempt"],
        "attempt_dir": str(paths.directory),
        "lease_path": str(_expected_lease_path(bundle, run)),
        "class_order": list(RSAR_CLASS_ORDER),
        "pid": terminal_record["pid"],
        "pid_start_ticks": terminal_record["pid_start_ticks"],
        "gpu_verified": True,
        "command": terminal_record["command"],
        "environment": terminal_record["environment"],
        "returncode": 0,
        "metrics_parsed_by_runner": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FinalTestBundleError(f"online_eval.json {key} mismatch")
    if payload.get("gpu", {}).get("uuid") != terminal_record["gpu_uuid"]:
        raise FinalTestBundleError("online_eval.json GPU UUID mismatch")
    expected_embedded_artifacts = CORE_OUTPUT_NAMES - {ONLINE_METADATA_BASENAME}
    if set(payload["artifacts"]) != expected_embedded_artifacts:
        raise FinalTestBundleError("online_eval.json embedded artifact set mismatch")
    for name in expected_embedded_artifacts:
        if payload["artifacts"].get(name) != terminal_record["artifacts"].get(name):
            raise FinalTestBundleError(f"online_eval.json {name} identity mismatch")


def _completed_record(
    registry: AppendOnlyRegistry, bundle: FrozenBundle, run: FrozenRun
) -> Optional[Mapping[str, Any]]:
    all_records = _read_validated_bundle_registry(registry, bundle)
    records = _records_for_run(all_records, bundle, run)
    _validate_history_shape(records)
    completed = [record for record in records if record["event"] == "completed"]
    if len(completed) > 1:
        raise FinalTestBundleError(
            "multiple completed records for " f"method={run.method}, seed={run.seed}"
        )
    if not completed:
        return None
    record = completed[0]
    _validate_record_binding(record, bundle, run)
    names = _inspect_output_dir(run.output_dir)
    if not CORE_OUTPUT_NAMES.issubset(names) or ATTEMPTS_DIRNAME not in names:
        raise FinalTestBundleError(
            "completed output set mismatch for " f"{run.output_dir}: {sorted(names)}"
        )
    actual = _canonical_artifact_identities(run)
    if actual != record["artifacts"]:
        raise FinalTestBundleError("completed output artifact identity mismatch")
    terminal = _validate_completed_history_evidence(
        bundle=bundle,
        run=run,
        records=records,
        completed_record=record,
    )
    paths = _attempt_paths(run, record["attempt"])
    _validate_online_metadata(paths, bundle, run, terminal)
    return record


def _validate_registry_path(registry: AppendOnlyRegistry, bundle: FrozenBundle) -> None:
    expected = registry_path_for_manifest(bundle.manifest_path)
    if registry.path != expected:
        raise FinalTestBundleError(
            f"registry must use fixed manifest-local path: {expected}"
        )


def _registry_record(
    *,
    event: str,
    bundle: FrozenBundle,
    run: FrozenRun,
    attempt: int,
    attempt_paths: AttemptPaths,
    lease: RunLease,
    gpu: gpu_scheduler.GPUInfo,
    pid: Optional[int],
    pid_start_ticks: Optional[int],
    gpu_verified: bool,
    command: Sequence[str],
    environment: Mapping[str, str],
    returncode: Optional[int],
    artifacts: Mapping[str, Any],
    failure: Optional[str],
) -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "event": event,
        "recorded_at_utc": _utc_now(),
        "run_identity": _run_identity(bundle, run),
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "method": run.method,
        "seed": run.seed,
        "training_fingerprint": run.training_fingerprint,
        "checkpoint_path": str(run.checkpoint_path),
        "checkpoint_sha256": run.checkpoint_sha256,
        "output_dir": str(run.output_dir),
        "attempt": attempt,
        "attempt_dir": str(attempt_paths.directory),
        "lease_path": str(lease.path),
        "gpu_index": gpu.index,
        "gpu_uuid": gpu.uuid,
        "gpu_name": gpu.name,
        "free_memory_mib_at_start": gpu.memory_free_mib,
        "pid": pid,
        "pid_start_ticks": pid_start_ticks,
        "gpu_verified": gpu_verified,
        "command": list(command),
        "environment": dict(environment),
        "returncode": returncode,
        "artifacts": dict(artifacts),
        "failure": failure,
    }


def _metadata_payload(
    *,
    bundle: FrozenBundle,
    run: FrozenRun,
    attempt: int,
    attempt_paths: AttemptPaths,
    lease: RunLease,
    gpu: gpu_scheduler.GPUInfo,
    pid: int,
    pid_start_ticks: int,
    command: Sequence[str],
    environment: Mapping[str, str],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ONLINE_METADATA_SCHEMA_VERSION,
        "status": "completed",
        "completed_at_utc": _utc_now(),
        "run_identity": _run_identity(bundle, run),
        "manifest_path": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "data_manifest_path": str(bundle.data_manifest_path),
        "data_manifest_sha256": bundle.data_manifest_sha256,
        "method": run.method,
        "seed": run.seed,
        "training_fingerprint": run.training_fingerprint,
        "checkpoint_path": str(run.checkpoint_path),
        "checkpoint_sha256": run.checkpoint_sha256,
        "output_dir": str(run.output_dir),
        "attempt": attempt,
        "attempt_dir": str(attempt_paths.directory),
        "lease_path": str(lease.path),
        "class_order": list(RSAR_CLASS_ORDER),
        "gpu": {"index": gpu.index, "name": gpu.name, "uuid": gpu.uuid},
        "pid": pid,
        "pid_start_ticks": pid_start_ticks,
        "gpu_verified": True,
        "command": list(command),
        "environment": dict(environment),
        "returncode": 0,
        "artifacts": dict(artifacts),
        "metrics_parsed_by_runner": False,
    }


def _find_current_run(bundle: FrozenBundle, expected: FrozenRun) -> FrozenRun:
    matches = [
        run
        for run in bundle.runs
        if run.method == expected.method and run.seed == expected.seed
    ]
    if len(matches) != 1 or matches[0] != expected:
        raise FinalTestBundleError(
            "frozen run identity changed for " f"{expected.method}/seed_{expected.seed}"
        )
    return matches[0]


def _validate_python_executable(python: Path) -> Path:
    try:
        selected = python.expanduser().resolve(strict=True)
        current = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestBundleError(
            f"cannot resolve Python executable: {error}"
        ) from error
    if not selected.is_file():
        raise FinalTestBundleError(f"Python is not a file: {selected}")
    if selected != current:
        raise FinalTestBundleError(
            "--python must exactly equal the current sys.executable: "
            f"expected {current}, got {selected}"
        )
    return selected


def _proc_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError as error:
        raise FinalTestBundleError(
            f"cannot read process identity for PID {pid}: {error}"
        ) from error
    closing = raw.rfind(")")
    if closing < 0:
        raise FinalTestBundleError(f"malformed /proc stat for PID {pid}")
    fields = raw[closing + 1 :].split()
    try:
        start_ticks = int(fields[19])
    except (IndexError, ValueError) as error:
        raise FinalTestBundleError(
            f"malformed process start time for PID {pid}"
        ) from error
    if start_ticks <= 0:
        raise FinalTestBundleError(f"invalid process start time for PID {pid}")
    return start_ticks


def _orphan_started_detail(
    started: Mapping[str, Any],
    process_start_probe: Callable[[int], int],
) -> str:
    pid = started["pid"]
    expected_ticks = started["pid_start_ticks"]
    try:
        current_ticks = process_start_probe(pid)
    except (FinalTestBundleError, OSError, ValueError):
        identity = "PID is absent or its identity cannot be proven"
    else:
        if current_ticks == expected_ticks:
            identity = "the recorded child PID is still alive"
        else:
            identity = "the PID has been reused by another process"
    return (
        "orphan started attempt has no immutable terminal evidence; "
        f"{identity}; automatic rerun is forbidden"
    )


def _terminate_process_group(process: Any, *, grace_seconds: float) -> Optional[int]:
    try:
        current = process.poll()
    except (AttributeError, OSError):
        current = None
    if current is not None:
        try:
            return int(process.wait(timeout=0))
        except (AttributeError, OSError, subprocess.TimeoutExpired):
            return int(current)

    try:
        if isinstance(process, subprocess.Popen):
            os.killpg(int(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except (AttributeError, OSError, ProcessLookupError):
        pass
    try:
        return int(process.wait(timeout=grace_seconds))
    except (AttributeError, OSError, subprocess.TimeoutExpired):
        pass

    try:
        if isinstance(process, subprocess.Popen):
            os.killpg(int(process.pid), signal.SIGKILL)
        else:
            process.kill()
    except (AttributeError, OSError, ProcessLookupError):
        pass
    try:
        return int(process.wait(timeout=grace_seconds))
    except (AttributeError, OSError, subprocess.TimeoutExpired, TypeError):
        return None


def _verify_process_gpu_binding(
    process: Any,
    gpu: gpu_scheduler.GPUInfo,
    *,
    compute_apps_probe: Callable[[], list[run_experiment.ComputeApp]],
    timeout_seconds: float,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
) -> None:
    deadline = clock() + timeout_seconds
    while True:
        try:
            apps = compute_apps_probe()
        except (OSError, RuntimeError, ValueError) as error:
            raise FinalTestBundleError(
                f"GPU process verification query failed: {error}"
            ) from error
        matches = [app for app in apps if app.pid == int(process.pid)]
        if matches:
            actual = {app.gpu_uuid for app in matches}
            if actual != {gpu.uuid}:
                raise FinalTestBundleError(
                    "child PID is bound to unexpected GPU UUID(s): "
                    f"expected {gpu.uuid}, got {sorted(actual)}"
                )
            return
        try:
            returncode = process.poll()
        except (AttributeError, OSError) as error:
            raise FinalTestBundleError(
                f"cannot poll child during GPU verification: {error}"
            ) from error
        if returncode is not None:
            raise FinalTestBundleError(
                "child exited before its PID appeared on the selected GPU UUID"
            )
        remaining = deadline - clock()
        if remaining <= 0:
            raise FinalTestBundleError(
                "child PID did not appear on the selected GPU UUID before "
                "the verification timeout"
            )
        sleeper(min(1.0, remaining))


def _monitor_process(
    process: Any,
    stdout_descriptor: int,
    *,
    monitor_interval: float,
    run_timeout: float,
    stall_timeout: float,
    terminate_grace: float,
    sleeper: Callable[[float], None],
    clock: Callable[[], float],
) -> tuple[Optional[int], Optional[str]]:
    started = clock()
    last_activity = started
    info = os.fstat(stdout_descriptor)
    signature = (info.st_size, info.st_mtime_ns)
    while True:
        try:
            returncode = process.poll()
        except (AttributeError, OSError) as error:
            returncode = _terminate_process_group(
                process, grace_seconds=terminate_grace
            )
            return returncode, f"cannot poll child process: {error}"
        if returncode is not None:
            try:
                return int(process.wait(timeout=0)), None
            except (AttributeError, OSError, subprocess.TimeoutExpired):
                return int(returncode), None
        now = clock()
        current = os.fstat(stdout_descriptor)
        current_signature = (current.st_size, current.st_mtime_ns)
        if current_signature != signature:
            signature = current_signature
            last_activity = now
        failure: Optional[str] = None
        if now - started >= run_timeout:
            failure = f"run timeout exceeded ({run_timeout:.1f}s)"
        elif now - last_activity >= stall_timeout:
            failure = f"stdout stall timeout exceeded ({stall_timeout:.1f}s)"
        if failure is not None:
            returncode = _terminate_process_group(
                process, grace_seconds=terminate_grace
            )
            return returncode, failure
        sleeper(monitor_interval)


def _validate_history_shape(records: Sequence[Mapping[str, Any]]) -> None:
    completed_positions = [
        index for index, record in enumerate(records) if record["event"] == "completed"
    ]
    if len(completed_positions) > 1:
        raise FinalTestBundleError("multiple completed records for one run")
    if completed_positions and completed_positions[0] != len(records) - 1:
        raise FinalTestBundleError("registry contains records after completion")
    for attempt in sorted({record["attempt"] for record in records}):
        grouped = [record for record in records if record["attempt"] == attempt]
        started = [record for record in grouped if record["event"] == "started"]
        terminal = [record for record in grouped if record["event"] != "started"]
        if len(started) > 1 or len(terminal) > 1:
            raise FinalTestBundleError(
                f"duplicate registry transition for attempt {attempt}"
            )
        if terminal:
            _validate_started_terminal_count(started, terminal[0], attempt)
        if (
            started
            and terminal
            and grouped.index(started[0]) > grouped.index(terminal[0])
        ):
            raise FinalTestBundleError(
                f"attempt {attempt} terminal precedes started record"
            )


def _validate_started_terminal_count(
    started: Sequence[Mapping[str, Any]],
    terminal: Mapping[str, Any],
    attempt: int,
) -> None:
    expected_started = 0 if terminal["pid"] is None else 1
    if len(started) != expected_started:
        raise FinalTestBundleError(
            f"attempt {attempt} has invalid started/terminal pairing"
        )
    if terminal["event"] == "completed" and expected_started != 1:
        raise FinalTestBundleError(
            f"completed attempt {attempt} lacks a started transition"
        )


def _validate_terminal_against_registry(
    terminal: Mapping[str, Any], record: Mapping[str, Any]
) -> None:
    for key in AppendOnlyRegistry.REQUIRED_FIELDS - {
        "schema_version",
        "artifacts",
    }:
        if terminal[key] != record[key]:
            raise FinalTestBundleError(f"attempt terminal {key} differs from registry")
    if terminal["event"] == "completed":
        for name in CORE_OUTPUT_NAMES:
            if _identity_value(terminal["artifacts"][name]) != (
                _identity_value(record["artifacts"][name])
            ):
                raise FinalTestBundleError(
                    f"attempt terminal {name} differs from registry"
                )
    elif terminal["artifacts"] != record["artifacts"]:
        raise FinalTestBundleError("failed attempt artifacts differ from registry")


def _validate_started_terminal_pair(
    started: Mapping[str, Any], terminal: Mapping[str, Any]
) -> None:
    excluded = {
        "schema_version",
        "event",
        "recorded_at_utc",
        "gpu_verified",
        "returncode",
        "artifacts",
        "failure",
    }
    for key in AppendOnlyRegistry.REQUIRED_FIELDS - excluded:
        if started[key] != terminal[key]:
            raise FinalTestBundleError(f"started/terminal {key} identity mismatch")


def _validate_completed_history_evidence(
    *,
    bundle: FrozenBundle,
    run: FrozenRun,
    records: Sequence[Mapping[str, Any]],
    completed_record: Mapping[str, Any],
) -> Mapping[str, Any]:
    directories = _inspect_attempt_directories(run.output_dir)
    registry_attempts = {record["attempt"] for record in records}
    directory_attempts = set(directories)
    if directory_attempts != registry_attempts:
        raise FinalTestBundleError(
            "attempt directory set differs from completed registry history"
        )
    if not directory_attempts or sorted(directory_attempts) != list(
        range(1, max(directory_attempts) + 1)
    ):
        raise FinalTestBundleError(
            "completed attempt directories must be contiguous from one"
        )
    completed_attempt = completed_record["attempt"]
    if completed_attempt != max(directory_attempts):
        raise FinalTestBundleError("completed event is not the final attempt")

    completed_terminal: Optional[Mapping[str, Any]] = None
    for attempt in sorted(directory_attempts):
        grouped = [record for record in records if record["attempt"] == attempt]
        started = [record for record in grouped if record["event"] == "started"]
        terminal_records = [
            record for record in grouped if record["event"] != "started"
        ]
        if len(terminal_records) != 1:
            raise FinalTestBundleError(f"attempt {attempt} lacks one registry terminal")
        registered_terminal = terminal_records[0]
        expected_event = "completed" if attempt == completed_attempt else "failed"
        if registered_terminal["event"] != expected_event:
            raise FinalTestBundleError(
                f"attempt {attempt} has unexpected terminal event"
            )
        _validate_started_terminal_count(started, registered_terminal, attempt)
        paths = _attempt_paths(run, attempt)
        terminal = _load_attempt_terminal(bundle, run, paths)
        _validate_terminal_against_registry(terminal, registered_terminal)
        if started:
            _validate_started_terminal_pair(started[0], terminal)
        if attempt == completed_attempt:
            completed_terminal = terminal

    if completed_terminal is None:
        raise FinalTestBundleError("completed terminal evidence is missing")
    return completed_terminal


def _publish_completed_attempt(
    *,
    registry: AppendOnlyRegistry,
    bundle: FrozenBundle,
    run: FrozenRun,
    paths: AttemptPaths,
    terminal: Mapping[str, Any],
) -> Mapping[str, Any]:
    if terminal["event"] != "completed":
        raise FinalTestBundleError("cannot publish a failed attempt")
    _validate_online_metadata(paths, bundle, run, terminal)
    _ensure_directory_chain(
        bundle.manifest_path.parent,
        (
            OUTPUT_ROOT_NAME,
            _safe_component(run.method, "method"),
            f"seed_{run.seed}",
        ),
    )
    for name in sorted(CORE_OUTPUT_NAMES):
        source_name = name
        if name == FRAMEWORK_EVAL_BASENAME:
            source_name = Path(terminal["artifacts"][name]["path"]).name
            if FRAMEWORK_EVAL_RE.fullmatch(source_name) is None:
                raise FinalTestBundleError(
                    "terminal framework eval source name is invalid"
                )
        _publish_exclusive_copy(
            paths.directory,
            run.output_dir,
            source_name,
            name,
            terminal["artifacts"][name],
        )
    canonical = _canonical_artifact_identities(run)
    completed_record = _record_from_terminal(
        _terminal_payload(terminal), artifacts=canonical
    )
    registry.append(completed_record)
    verified = _completed_record(registry, bundle, run)
    if verified is None:
        raise FinalTestBundleError("completed record vanished after append")
    return verified


def _reconcile_incomplete_history(
    *,
    registry: AppendOnlyRegistry,
    bundle: FrozenBundle,
    run: FrozenRun,
    process_start_probe: Callable[[int], int],
) -> tuple[int, Optional[Mapping[str, Any]]]:
    records = _records_for_run(
        _read_validated_bundle_registry(registry, bundle), bundle, run
    )
    _validate_history_shape(records)
    completed = _completed_record(registry, bundle, run)
    if completed is not None:
        return _attempt_count(records), completed

    names = _inspect_output_dir(run.output_dir)
    directories = _inspect_attempt_directories(run.output_dir)
    if directories and sorted(directories) != list(range(1, max(directories) + 1)):
        raise FinalTestBundleError("attempt directories are not contiguous")
    record_attempts = {record["attempt"] for record in records}
    for attempt in sorted(record_attempts - set(directories)):
        raise FinalTestBundleError(
            f"registry attempt {attempt} has no immutable attempt directory"
        )

    for attempt, directory in sorted(directories.items()):
        paths = _attempt_paths(run, attempt)
        if directory != paths.directory:
            raise FinalTestBundleError("attempt directory path drift detected")
        grouped = [record for record in records if record["attempt"] == attempt]
        started = [record for record in grouped if record["event"] == "started"]
        registered_terminal = [
            record for record in grouped if record["event"] != "started"
        ]
        attempt_names = _inspect_attempt_directory(paths.directory)
        has_terminal = ATTEMPT_TERMINAL_BASENAME in attempt_names
        if registered_terminal:
            _validate_started_terminal_count(started, registered_terminal[0], attempt)
            if not has_terminal:
                raise FinalTestBundleError(
                    f"registered attempt {attempt} lacks terminal evidence"
                )
            terminal = _load_attempt_terminal(bundle, run, paths)
            _validate_terminal_against_registry(terminal, registered_terminal[0])
            if started:
                _validate_started_terminal_pair(started[0], terminal)
            continue
        if started:
            if not has_terminal:
                raise FinalTestBundleError(
                    _orphan_started_detail(started[0], process_start_probe)
                )
            terminal = _load_attempt_terminal(bundle, run, paths)
            _validate_started_terminal_pair(started[0], terminal)
            if terminal["event"] == "completed":
                recovered = _publish_completed_attempt(
                    registry=registry,
                    bundle=bundle,
                    run=run,
                    paths=paths,
                    terminal=terminal,
                )
                return len(directories), recovered
            registry.append(terminal)
            records.append(terminal)
            continue
        if has_terminal:
            terminal = _load_attempt_terminal(bundle, run, paths)
            if terminal["event"] != "failed" or terminal["pid"] is not None:
                raise FinalTestBundleError(
                    "unregistered attempt terminal cannot be trusted"
                )
            registry.append(terminal)
            records.append(terminal)
            continue
        raise FinalTestBundleError(
            f"unregistered attempt directory is ambiguous: {paths.directory}"
        )

    if names & CORE_OUTPUT_NAMES:
        raise FinalTestBundleError(
            "canonical outputs exist without a verified completed record"
        )
    records = _records_for_run(
        _read_validated_bundle_registry(registry, bundle), bundle, run
    )
    _validate_history_shape(records)
    return _attempt_count(records), None


def run_one_frozen_run(
    *,
    initial_bundle: FrozenBundle,
    run: FrozenRun,
    gpu: gpu_scheduler.GPUInfo,
    registry: AppendOnlyRegistry,
    python: Path,
    max_attempts: int,
    base_environment: Mapping[str, str],
    popen_factory: Callable[..., Any] = subprocess.Popen,
    compute_apps_probe: Callable[[], list[run_experiment.ComputeApp]] = (
        run_experiment.query_compute_apps
    ),
    process_start_probe: Callable[[int], int] = _proc_start_ticks,
    gpu_verify_timeout: float = DEFAULT_GPU_VERIFY_TIMEOUT,
    monitor_interval: float = DEFAULT_MONITOR_INTERVAL,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    after_attempt_terminal: Optional[
        Callable[[AttemptPaths, Mapping[str, Any]], None]
    ] = None,
    gpu_lease_fd: Optional[int] = None,
) -> RunResult:
    """Run or strictly skip one manifest-declared inference."""

    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise FinalTestBundleError("max_attempts must be between 1 and 3")
    _require_finite_interval(monitor_interval, "monitor_interval", maximum=60)
    for value, role in (
        (gpu_verify_timeout, "gpu_verify_timeout"),
        (run_timeout, "run_timeout"),
        (stall_timeout, "stall_timeout"),
        (terminate_grace, "terminate_grace"),
    ):
        _require_finite_interval(value, role)
    python = _validate_python_executable(python)

    bundle = load_frozen_bundle(initial_bundle.manifest_path, verify_data=True)
    if bundle.manifest_sha256 != initial_bundle.manifest_sha256:
        raise FinalTestBundleError("manifest digest changed before a run")
    current_run = _find_current_run(bundle, run)
    _validate_registry_path(registry, bundle)

    with RunLease(bundle, current_run) as lease:
        used, completed = _reconcile_incomplete_history(
            registry=registry,
            bundle=bundle,
            run=current_run,
            process_start_probe=process_start_probe,
        )
        if completed is not None:
            return RunResult(
                method=run.method,
                seed=run.seed,
                status="skipped_verified",
                attempts=used,
                gpu_uuid=completed["gpu_uuid"],
            )
        if used >= max_attempts:
            raise FinalTestBundleError(
                f"attempt limit reached for run {_run_identity(bundle, run)}"
            )
        output_directory = _ensure_directory_chain(
            bundle.manifest_path.parent,
            (
                OUTPUT_ROOT_NAME,
                _safe_component(current_run.method, "method"),
                f"seed_{current_run.seed}",
            ),
        )
        if output_directory != current_run.output_dir:
            raise FinalTestBundleError(
                "publication preflight directory differs from frozen output_dir"
            )
        _preflight_anonymous_publication(output_directory)

        environment = build_run_environment(base_environment, gpu.uuid, python)
        recorded_environment = relevant_environment(environment)
        last_failure: Optional[str] = None

        for attempt in range(used + 1, max_attempts + 1):
            refreshed = load_frozen_bundle(bundle.manifest_path, verify_data=True)
            if refreshed.manifest_sha256 != bundle.manifest_sha256:
                raise FinalTestBundleError("manifest digest changed between attempts")
            current_run = _find_current_run(refreshed, run)
            bundle = refreshed
            _inspect_output_dir(current_run.output_dir)
            paths = _create_attempt_paths(bundle, current_run, attempt)
            command = build_test_command(
                bundle,
                current_run,
                python,
                predictions_path=paths.predictions,
            )

            directory_fd = _open_directory(paths.directory, "attempt directory")
            stdout_descriptor = _exclusive_create_fd(directory_fd, STDOUT_BASENAME)
            os.close(directory_fd)
            stdout_handle = os.fdopen(stdout_descriptor, "wb", buffering=0)
            stdout_handle.write(
                (
                    "=== final-test attempt " f"{attempt} started {_utc_now()} ===\n"
                ).encode("utf-8")
            )
            os.fsync(stdout_descriptor)

            process: Optional[Any] = None
            pid: Optional[int] = None
            pid_start_ticks: Optional[int] = None
            returncode: Optional[int] = None
            gpu_verified = False
            failure: Optional[str] = None
            try:
                try:
                    inherited_fds = [lease.fd]
                    if gpu_lease_fd is not None:
                        if gpu_lease_fd < 0:
                            raise FinalTestBundleError("GPU lease FD is invalid")
                        inherited_fds.append(gpu_lease_fd)
                    process = popen_factory(
                        list(command),
                        cwd=str(bundle.project_root),
                        env=environment,
                        stdout=stdout_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=tuple(inherited_fds),
                    )
                except (OSError, subprocess.SubprocessError) as error:
                    failure = f"subprocess launch failed: {error}"
                if process is not None:
                    pid = int(process.pid)
                    if pid <= 0:
                        raise FinalTestBundleError("child PID is invalid")
                    pid_start_ticks = process_start_probe(pid)
                    started_record = _registry_record(
                        event="started",
                        bundle=bundle,
                        run=current_run,
                        attempt=attempt,
                        attempt_paths=paths,
                        lease=lease,
                        gpu=gpu,
                        pid=pid,
                        pid_start_ticks=pid_start_ticks,
                        gpu_verified=False,
                        command=command,
                        environment=recorded_environment,
                        returncode=None,
                        artifacts={},
                        failure=None,
                    )
                    try:
                        registry.append(started_record)
                    except BaseException:
                        _terminate_process_group(process, grace_seconds=terminate_grace)
                        raise
                    try:
                        _verify_process_gpu_binding(
                            process,
                            gpu,
                            compute_apps_probe=compute_apps_probe,
                            timeout_seconds=gpu_verify_timeout,
                            sleeper=sleeper,
                            clock=clock,
                        )
                        gpu_verified = True
                    except FinalTestBundleError as error:
                        failure = str(error)
                        returncode = _terminate_process_group(
                            process, grace_seconds=terminate_grace
                        )
                    if failure is None:
                        returncode, failure = _monitor_process(
                            process,
                            stdout_descriptor,
                            monitor_interval=monitor_interval,
                            run_timeout=run_timeout,
                            stall_timeout=stall_timeout,
                            terminate_grace=terminate_grace,
                            sleeper=sleeper,
                            clock=clock,
                        )
                        if returncode != 0 and failure is None:
                            failure = f"test.py returned {returncode}"
                    if failure is not None and returncode is None:
                        raise FinalTestBundleError(
                            "child termination could not be proven; "
                            "attempt remains orphaned and cannot be retried"
                        )
            except BaseException:
                if process is not None:
                    _terminate_process_group(process, grace_seconds=terminate_grace)
                raise
            finally:
                stdout_handle.close()

            _fsync_attempt_artifacts(paths)

            if process is None:
                partial = _partial_attempt_artifacts(paths)
                failed_record = _registry_record(
                    event="failed",
                    bundle=bundle,
                    run=current_run,
                    attempt=attempt,
                    attempt_paths=paths,
                    lease=lease,
                    gpu=gpu,
                    pid=None,
                    pid_start_ticks=None,
                    gpu_verified=False,
                    command=command,
                    environment=recorded_environment,
                    returncode=None,
                    artifacts=partial,
                    failure=failure or "unknown launch failure",
                )
                _exclusive_write_json_at(
                    paths.directory,
                    ATTEMPT_TERMINAL_BASENAME,
                    _terminal_payload(failed_record),
                )
                registry.append(failed_record)
                last_failure = failed_record["failure"]
                continue

            if returncode == 0 and failure is None and gpu_verified:
                base_artifacts = _successful_attempt_base_artifacts(paths)
                _exclusive_write_json_at(
                    paths.directory,
                    ONLINE_METADATA_BASENAME,
                    _metadata_payload(
                        bundle=bundle,
                        run=current_run,
                        attempt=attempt,
                        attempt_paths=paths,
                        lease=lease,
                        gpu=gpu,
                        pid=pid,
                        pid_start_ticks=pid_start_ticks,
                        command=command,
                        environment=recorded_environment,
                        artifacts=base_artifacts,
                    ),
                )
                attempt_artifacts = _partial_attempt_artifacts(paths)
                if set(attempt_artifacts) != CORE_OUTPUT_NAMES:
                    raise FinalTestBundleError(
                        "successful attempt artifact set is incomplete"
                    )
                terminal_record = _registry_record(
                    event="completed",
                    bundle=bundle,
                    run=current_run,
                    attempt=attempt,
                    attempt_paths=paths,
                    lease=lease,
                    gpu=gpu,
                    pid=pid,
                    pid_start_ticks=pid_start_ticks,
                    gpu_verified=True,
                    command=command,
                    environment=recorded_environment,
                    returncode=0,
                    artifacts=attempt_artifacts,
                    failure=None,
                )
                _exclusive_write_json_at(
                    paths.directory,
                    ATTEMPT_TERMINAL_BASENAME,
                    _terminal_payload(terminal_record),
                )
                if after_attempt_terminal is not None:
                    after_attempt_terminal(paths, terminal_record)
                terminal = _load_attempt_terminal(bundle, current_run, paths)
                completed = _publish_completed_attempt(
                    registry=registry,
                    bundle=bundle,
                    run=current_run,
                    paths=paths,
                    terminal=terminal,
                )
                return RunResult(
                    method=current_run.method,
                    seed=current_run.seed,
                    status="completed",
                    attempts=attempt,
                    gpu_uuid=completed["gpu_uuid"],
                )

            partial = _partial_attempt_artifacts(paths)
            failed_record = _registry_record(
                event="failed",
                bundle=bundle,
                run=current_run,
                attempt=attempt,
                attempt_paths=paths,
                lease=lease,
                gpu=gpu,
                pid=pid,
                pid_start_ticks=pid_start_ticks,
                gpu_verified=gpu_verified,
                command=command,
                environment=recorded_environment,
                returncode=returncode,
                artifacts=partial,
                failure=failure or "unknown process failure",
            )
            _exclusive_write_json_at(
                paths.directory,
                ATTEMPT_TERMINAL_BASENAME,
                _terminal_payload(failed_record),
            )
            registry.append(failed_record)
            last_failure = failed_record["failure"]

        return RunResult(
            method=run.method,
            seed=run.seed,
            status="failed",
            attempts=max_attempts,
            gpu_uuid=gpu.uuid,
            detail=last_failure,
        )


def _run_with_lock(
    lock: GPUExecutionLease,
    **kwargs: Any,
) -> RunResult:
    try:
        return run_one_frozen_run(gpu_lease_fd=lock.fd, **kwargs)
    finally:
        lock.release()


def _validate_gpu_inventory(
    inventory: Sequence[gpu_scheduler.GPUInfo],
) -> None:
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for gpu in inventory:
        if type(gpu.index) is not int or gpu.index < 0:
            raise FinalTestBundleError("GPU inventory index is invalid")
        if type(gpu.uuid) is not str or not gpu.uuid:
            raise FinalTestBundleError("GPU inventory UUID is invalid")
        if type(gpu.name) is not str or not gpu.name:
            raise FinalTestBundleError("GPU inventory name is invalid")
        memory_values = (
            gpu.memory_used_mib,
            gpu.memory_free_mib,
            gpu.memory_total_mib,
        )
        if any(type(value) is not int or value < 0 for value in memory_values):
            raise FinalTestBundleError("GPU inventory memory is invalid")
        if (
            gpu.memory_total_mib <= 0
            or gpu.memory_used_mib > gpu.memory_total_mib
            or gpu.memory_free_mib > gpu.memory_total_mib
            or gpu.memory_used_mib + gpu.memory_free_mib > gpu.memory_total_mib
        ):
            raise FinalTestBundleError(
                "GPU inventory used/free/total memory is inconsistent"
            )
        if (
            type(gpu.utilization_percent) is not int
            or not 0 <= gpu.utilization_percent <= 100
        ):
            raise FinalTestBundleError(
                "GPU inventory utilization must be between 0 and 100"
            )
        if gpu.index in seen_indices or gpu.uuid in seen_uuids:
            raise FinalTestBundleError("GPU inventory contains duplicates")
        seen_indices.add(gpu.index)
        seen_uuids.add(gpu.uuid)


def execute_bundle(
    *,
    bundle: FrozenBundle,
    registry: AppendOnlyRegistry,
    python: Path,
    max_workers: int,
    max_attempts: int,
    required_free_mib: int,
    utilization_limit: int,
    gpu_poll_interval: float,
    max_gpu_wait: float,
    base_environment: Mapping[str, str],
    gpu_verify_timeout: float = DEFAULT_GPU_VERIFY_TIMEOUT,
    monitor_interval: float = DEFAULT_MONITOR_INTERVAL,
    run_timeout: float = DEFAULT_RUN_TIMEOUT,
    stall_timeout: float = DEFAULT_STALL_TIMEOUT,
    terminate_grace: float = DEFAULT_TERMINATE_GRACE,
    inventory_probe: Callable[[], list[gpu_scheduler.GPUInfo]] = (
        gpu_scheduler.query_gpu_inventory
    ),
    compute_apps_probe: Callable[[], list[run_experiment.ComputeApp]] = (
        run_experiment.query_compute_apps
    ),
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    process_start_probe: Callable[[int], int] = _proc_start_ticks,
) -> list[RunResult]:
    if not 1 <= max_workers <= MAX_WORKERS:
        raise FinalTestBundleError("max_workers must be between 1 and 4")
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise FinalTestBundleError("max_attempts must be between 1 and 3")
    if required_free_mib < 8192:
        raise FinalTestBundleError("required_free_mib must be at least 8192")
    if not 0 <= utilization_limit <= 100:
        raise FinalTestBundleError("utilization_limit must be between 0 and 100")
    _require_finite_interval(gpu_poll_interval, "gpu_poll_interval")
    _require_finite_interval(max_gpu_wait, "max_gpu_wait", allow_zero=True)
    _require_finite_interval(monitor_interval, "monitor_interval", maximum=60)
    for value, role in (
        (gpu_verify_timeout, "gpu_verify_timeout"),
        (run_timeout, "run_timeout"),
        (stall_timeout, "stall_timeout"),
        (terminate_grace, "terminate_grace"),
    ):
        _require_finite_interval(value, role)
    python = _validate_python_executable(python)
    _validate_registry_path(registry, bundle)
    lock_root = _ensure_directory_chain(
        bundle.manifest_path.parent, (LOCK_ROOT_NAME, "gpus")
    )

    pending: list[FrozenRun] = []
    active: dict[
        concurrent.futures.Future[RunResult], tuple[str, GPUExecutionLease]
    ] = {}
    results: list[RunResult] = []
    no_gpu_since: Optional[float] = None
    # A completed run should not need an available GPU merely to be skipped.
    # Pending runs are verified once more immediately before their subprocess.
    for run in bundle.runs:
        refreshed = load_frozen_bundle(bundle.manifest_path, verify_data=True)
        if refreshed.manifest_sha256 != bundle.manifest_sha256:
            raise FinalTestBundleError(
                "manifest digest changed during completion preflight"
            )
        current_run = _find_current_run(refreshed, run)
        completed = _completed_record(registry, refreshed, current_run)
        if completed is None:
            pending.append(run)
            continue
        all_records = _read_validated_bundle_registry(registry, refreshed)
        results.append(
            RunResult(
                method=run.method,
                seed=run.seed,
                status="skipped_verified",
                attempts=_attempt_count(
                    _records_for_run(all_records, refreshed, current_run)
                ),
                gpu_uuid=completed["gpu_uuid"],
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while pending or active:
            if pending and len(active) < max_workers:
                try:
                    inventory = inventory_probe()
                    compute_apps = compute_apps_probe()
                except (OSError, RuntimeError, ValueError) as error:
                    raise FinalTestBundleError(
                        f"GPU safety query failed: {error}"
                    ) from error
                _validate_gpu_inventory(inventory)
                busy = {app.gpu_uuid for app in compute_apps}
                active_uuids = {uuid for uuid, _ in active.values()}
                candidates = sorted(
                    (
                        gpu
                        for gpu in inventory
                        if gpu.uuid not in active_uuids
                        and gpu_scheduler.eligible_gpu(
                            gpu, required_free_mib, utilization_limit, busy
                        )
                    ),
                    key=lambda gpu: (-gpu.memory_free_mib, gpu.index),
                )
                assigned = False
                while pending and candidates and len(active) < max_workers:
                    run = pending[0]
                    selected_index: Optional[int] = None
                    selected_lock: Optional[GPUExecutionLease] = None
                    for index, gpu in enumerate(candidates):
                        lock = GPUExecutionLease(lock_root, gpu.uuid)
                        if lock.acquire():
                            selected_index = index
                            selected_lock = lock
                            break
                        lock.release()
                    if selected_index is None or selected_lock is None:
                        break
                    gpu = candidates.pop(selected_index)
                    pending.pop(0)
                    future = executor.submit(
                        _run_with_lock,
                        selected_lock,
                        initial_bundle=bundle,
                        run=run,
                        gpu=gpu,
                        registry=registry,
                        python=python,
                        max_attempts=max_attempts,
                        base_environment=base_environment,
                        popen_factory=popen_factory,
                        compute_apps_probe=compute_apps_probe,
                        process_start_probe=process_start_probe,
                        gpu_verify_timeout=gpu_verify_timeout,
                        monitor_interval=monitor_interval,
                        run_timeout=run_timeout,
                        stall_timeout=stall_timeout,
                        terminate_grace=terminate_grace,
                        sleeper=sleeper,
                        clock=clock,
                    )
                    active[future] = (gpu.uuid, selected_lock)
                    assigned = True
                if assigned:
                    no_gpu_since = None
                elif not active:
                    if no_gpu_since is None:
                        no_gpu_since = clock()
                    if clock() - no_gpu_since >= max_gpu_wait:
                        raise FinalTestBundleError("no safe GPU became available")

            if active:
                done, _ = concurrent.futures.wait(
                    active,
                    timeout=gpu_poll_interval,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    active.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as error:
                        if isinstance(error, FinalTestBundleError):
                            detail = str(error)
                        else:
                            detail = f"{type(error).__name__}: {error}"
                        results.append(
                            RunResult(
                                method="<scheduler-error>",
                                seed=-1,
                                status="failed",
                                attempts=0,
                                detail=detail,
                            )
                        )
            elif pending:
                sleeper(min(gpu_poll_interval, max(0.0, max_gpu_wait)))
    return results


def verify_completed_run(manifest_path: Path, method: str, seed: int) -> dict[str, Any]:
    """Return the unique, fully identity-verified completed registry record."""

    _safe_component(method, "method")
    if type(seed) is not int or seed < 0:
        raise FinalTestBundleError("seed must be a non-negative integer")
    bundle = load_frozen_bundle(manifest_path, verify_data=True)
    matches = [run for run in bundle.runs if run.method == method and run.seed == seed]
    if len(matches) != 1:
        raise FinalTestBundleError(
            "frozen manifest must contain exactly one matching run; "
            f"found {len(matches)}"
        )
    registry = AppendOnlyRegistry(registry_path_for_manifest(bundle.manifest_path))
    completed = _completed_record(registry, bundle, matches[0])
    if completed is None:
        raise FinalTestBundleError(
            f"no verified completed record for {method}/seed_{seed}"
        )
    return dict(completed)


def dry_run_plan(bundle: FrozenBundle, python: Path) -> dict[str, Any]:
    python = _validate_python_executable(python)

    def plan_run(run: FrozenRun) -> dict[str, Any]:
        attempt_directory = run.output_dir / ATTEMPTS_DIRNAME / "attempt_{attempt:03d}"
        predictions = attempt_directory / PREDICTIONS_BASENAME
        return {
            "method": run.method,
            "seed": run.seed,
            "checkpoint": str(run.checkpoint_path),
            "checkpoint_sha256": run.checkpoint_sha256,
            "output_dir": str(run.output_dir),
            "attempt_dir_template": str(attempt_directory),
            "outputs": {
                "predictions": str(run.predictions_path),
                "stdout": str(run.stdout_path),
                "online_eval_metadata": str(run.online_metadata_path),
                "framework_eval": str(run.framework_eval_path),
            },
            "command_template": build_test_command(
                bundle, run, python, predictions_path=predictions
            ),
            "environment": relevant_environment(
                build_run_environment(os.environ, "<assigned-gpu-uuid>", python)
            ),
            "gpu": "assigned only during execution",
        }

    return {
        "status": "dry_run",
        "manifest": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "data_manifest": str(bundle.data_manifest_path),
        "data_manifest_sha256": bundle.data_manifest_sha256,
        "dataset_bytes_verified": False,
        "gpu_queried": False,
        "registry": str(registry_path_for_manifest(bundle.manifest_path)),
        "runs": [plan_run(run) for run in bundle.runs],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("IRAOD_PYTHON", sys.executable)),
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    parser.add_argument("--required-free-mib", type=int, default=8192)
    parser.add_argument("--utilization-limit", type=int, default=30)
    parser.add_argument("--gpu-poll-interval", type=float, default=60.0)
    parser.add_argument("--max-gpu-wait", type=float, default=7200.0)
    parser.add_argument(
        "--gpu-verify-timeout",
        type=float,
        default=DEFAULT_GPU_VERIFY_TIMEOUT,
    )
    parser.add_argument(
        "--monitor-interval", type=float, default=DEFAULT_MONITOR_INTERVAL
    )
    parser.add_argument("--run-timeout", type=float, default=DEFAULT_RUN_TIMEOUT)
    parser.add_argument("--stall-timeout", type=float, default=DEFAULT_STALL_TIMEOUT)
    parser.add_argument(
        "--terminate-grace", type=float, default=DEFAULT_TERMINATE_GRACE
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        python = _validate_python_executable(args.python)
        if not 1 <= args.max_workers <= MAX_WORKERS:
            raise FinalTestBundleError("--max-workers must be between 1 and 4")
        if not 1 <= args.max_attempts <= MAX_ATTEMPTS:
            raise FinalTestBundleError("--max-attempts must be between 1 and 3")
        _require_finite_interval(args.gpu_poll_interval, "--gpu-poll-interval")
        _require_finite_interval(args.max_gpu_wait, "--max-gpu-wait", allow_zero=True)
        _require_finite_interval(args.gpu_verify_timeout, "--gpu-verify-timeout")
        _require_finite_interval(
            args.monitor_interval, "--monitor-interval", maximum=60
        )
        _require_finite_interval(args.run_timeout, "--run-timeout")
        _require_finite_interval(args.stall_timeout, "--stall-timeout")
        _require_finite_interval(args.terminate_grace, "--terminate-grace")
        bundle = load_frozen_bundle(
            args.manifest,
            verify_data=not args.dry_run,
        )
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_plan(bundle, python),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        registry = AppendOnlyRegistry(registry_path_for_manifest(bundle.manifest_path))
        results = execute_bundle(
            bundle=bundle,
            registry=registry,
            python=python,
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
            required_free_mib=args.required_free_mib,
            utilization_limit=args.utilization_limit,
            gpu_poll_interval=args.gpu_poll_interval,
            max_gpu_wait=args.max_gpu_wait,
            base_environment=os.environ,
            gpu_verify_timeout=args.gpu_verify_timeout,
            monitor_interval=args.monitor_interval,
            run_timeout=args.run_timeout,
            stall_timeout=args.stall_timeout,
            terminate_grace=args.terminate_grace,
        )
        summary = {
            "manifest": str(bundle.manifest_path),
            "manifest_sha256": bundle.manifest_sha256,
            "registry": str(registry.path),
            "runs": [dataclasses.asdict(result) for result in results],
            "completed": sum(result.status == "completed" for result in results),
            "skipped_verified": sum(
                result.status == "skipped_verified" for result in results
            ),
            "failed": sum(result.status == "failed" for result in results),
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["failed"] == 0 else 2
    except (
        FinalTestBundleError,
        manifest_tool.ManifestError,
        data_manifest_tool.DataManifestError,
        OSError,
    ) as error:
        print(f"final-test bundle error: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
