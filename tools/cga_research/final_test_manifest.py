#!/usr/bin/env python3
"""Freeze and verify the inputs for a single, final test evaluation.

This tool never runs evaluation and never reads evaluation metrics.  ``create``
accepts a fully declared draft, verifies every declared local artifact, and
writes canonical JSON plus a detached SHA-256 sidecar.  ``verify`` checks the
sidecar, canonical representation, schema, checkpoint metadata, and every
artifact again.  Any missing or changed input is an error.

Bundle draft schema (unknown fields are rejected)::

    {
      "schema_version": 2,
      "state": "frozen_before_first_test",
      "project_root": "/absolute/path/to/IRAOD-New",
      "arms": [{
        "method": "chosen_method",
        "method_environment": {"CGA_FILTER_MODE": "legacy"},
        "hyperparameters": {"model.cfg.score_thr": 0.9},
        "runs": [{
          "seed": 41,
          "training_fingerprint": "<64 lowercase hex characters>",
          "checkpoint": {
            "path": "work_dirs/.../iter_4235_ema.pth",
            "size_bytes": 123,
            "sha256": "<64 lowercase hex characters>",
            "meta": {"iter": 4235, "epoch": 1, "seed": 41}
          },
          "output_dir": "work_dirs/final_test/chosen_method/seed_41"
        }]
      }],
      "aggregation": {
        "required_seed_set": [41],
        "paired_comparator": "no_cga",
        "exclude_failed_seed": false,
        "mean": true,
        "sample_std_ddof": 1,
        "bootstrap_samples": 10000,
        "bootstrap_seed": 20260715,
        "paired_t_test": true,
        "exact_sign_flip_permutation": true,
        "cohens_dz": true,
        "holm_correction": true
      },
      "selection": {
        "evidence": [{"path": "...", "sha256": "..."}],
        "config": {"path": "...", "sha256": "..."},
        "runtime_files": [{"path": "...", "sha256": "..."}],
        "data_manifest": {"path": "...", "sha256": "..."}
      }
    }

Relative artifact paths are resolved from ``project_root``.  The detached hash
for ``manifest.json`` is written as ``manifest.json.sha256``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2
FROZEN_STATE = "frozen_before_first_test"
CHECKPOINT_BASENAME = "iter_4235_ema.pth"
CHECKPOINT_ITER = 4235
CHECKPOINT_EPOCH = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ManifestError(ValueError):
    """Raised when a manifest cannot be trusted."""


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ManifestError(f"non-finite JSON value is forbidden: {value}")


def _load_json_bytes(content: bytes, source: Path) -> dict[str, Any]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ManifestError(f"{source}: JSON must be UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ManifestError:
        raise
    except json.JSONDecodeError as error:
        raise ManifestError(f"{source}: invalid JSON: {error}") from error
    if type(payload) is not dict:
        raise ManifestError(f"{source}: root must be an object")
    return payload


def load_json(path: Path) -> dict[str, Any]:
    try:
        return _load_json_bytes(path.read_bytes(), path)
    except OSError as error:
        raise ManifestError(f"cannot read {path}: {error}") from error


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ManifestError(
            f"manifest is not canonical-JSON serializable: {error}"
        ) from error
    return (text + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ManifestError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _require_unpublished(path: Path, role: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ManifestError(f"cannot inspect {role} {path}: {error}") from error
    kind = "symlink" if stat.S_ISLNK(info.st_mode) else "existing path"
    raise ManifestError(f"{role} is write-once and already exists as a {kind}: {path}")


def _stage_bytes(path: Path, content: bytes) -> tuple[Path, tuple[int, int]]:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            info = os.fstat(handle.fileno())
        return temporary, (info.st_dev, info.st_ino)
    except OSError as error:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise ManifestError(f"cannot stage frozen output {path}: {error}") from error


def _unlink_owned(path: Path, identity: tuple[int, int]) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ManifestError(
            f"cannot inspect published output {path}: {error}"
        ) from error
    if (info.st_dev, info.st_ino) != identity:
        return
    try:
        path.unlink()
    except OSError as error:
        raise ManifestError(
            f"cannot roll back published output {path}: {error}"
        ) from error


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ManifestError(f"cannot open output directory {path}: {error}") from error
    try:
        os.fsync(descriptor)
    except OSError as error:
        raise ManifestError(f"cannot fsync output directory {path}: {error}") from error
    finally:
        os.close(descriptor)


def _publish_frozen_bundle(
    output_path: Path,
    content: bytes,
    sidecar_content: bytes,
    expected_digest: str,
) -> str:
    checksum_path = sidecar_path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ManifestError(
            f"cannot create output directory {output_path.parent}: {error}"
        ) from error
    _require_unpublished(output_path, "frozen manifest")
    _require_unpublished(checksum_path, "frozen manifest sidecar")

    staged: list[Path] = []
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        staged_sidecar, sidecar_identity = _stage_bytes(checksum_path, sidecar_content)
        staged.append(staged_sidecar)
        staged_manifest, manifest_identity = _stage_bytes(output_path, content)
        staged.append(staged_manifest)

        for staged_path, target, identity, role in (
            (
                staged_sidecar,
                checksum_path,
                sidecar_identity,
                "frozen manifest sidecar",
            ),
            (staged_manifest, output_path, manifest_identity, "frozen manifest"),
        ):
            try:
                os.link(staged_path, target)
            except FileExistsError as error:
                raise ManifestError(
                    f"{role} appeared concurrently; refusing to overwrite: {target}"
                ) from error
            except OSError as error:
                raise ManifestError(
                    f"cannot publish {role} {target}: {error}"
                ) from error
            published.append((target, identity))

        _fsync_directory(output_path.parent)
        verified_digest = verify_manifest(output_path)
        if verified_digest != expected_digest:
            raise ManifestError("post-create manifest verification changed its digest")
        return verified_digest
    except BaseException:
        rollback_error: ManifestError | None = None
        for target, identity in reversed(published):
            try:
                _unlink_owned(target, identity)
            except ManifestError as error:
                rollback_error = rollback_error or error
        try:
            _fsync_directory(output_path.parent)
        except ManifestError as error:
            rollback_error = rollback_error or error
        if rollback_error is not None:
            raise rollback_error
        raise
    finally:
        for temporary in staged:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def sidecar_path(manifest_path: Path) -> Path:
    return Path(f"{manifest_path}.sha256")


def _require_regular_non_symlink(path: Path, role: str) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ManifestError(f"cannot inspect {role} {path}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ManifestError(f"{role} must be a regular non-symlink file: {path}")


def _require_object(
    value: Any,
    location: str,
    required_keys: set[str],
) -> dict[str, Any]:
    if type(value) is not dict:
        raise ManifestError(f"{location} must be an object")
    keys = set(value)
    missing = sorted(required_keys - keys)
    extra = sorted(keys - required_keys)
    if missing:
        raise ManifestError(f"{location} missing required fields: {', '.join(missing)}")
    if extra:
        raise ManifestError(f"{location} has unknown fields: {', '.join(extra)}")
    return value


def _require_string(value: Any, location: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise ManifestError(f"{location} must be a non-empty string")
    return value


def _require_int(value: Any, location: str) -> int:
    if type(value) is not int:
        raise ManifestError(f"{location} must be an integer")
    return value


def _require_bool(value: Any, location: str) -> bool:
    if type(value) is not bool:
        raise ManifestError(f"{location} must be a boolean")
    return value


def _require_sha256(value: Any, location: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise ManifestError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_list(value: Any, location: str) -> list[Any]:
    if type(value) is not list or not value:
        raise ManifestError(f"{location} must be a non-empty array")
    return value


def _validate_json_value(value: Any, location: str) -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ManifestError(f"{location} must be finite")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str or not key:
                raise ManifestError(f"{location} keys must be non-empty strings")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise ManifestError(f"{location} contains a non-JSON value: {type(value).__name__}")


def _validate_method_environment(value: Any, location: str) -> None:
    if type(value) is not dict:
        raise ManifestError(f"{location} must be an object")
    for key, item in value.items():
        if type(key) is not str or ENV_KEY_RE.fullmatch(key) is None:
            raise ManifestError(f"{location} has an invalid environment key: {key!r}")
        if type(item) is not str or "\x00" in item:
            raise ManifestError(f"{location}.{key} must be a string without NUL")


def _validate_hyperparameters(value: Any, location: str) -> None:
    if type(value) is not dict:
        raise ManifestError(f"{location} must be an object")
    _validate_json_value(value, location)


def _resolve_artifact(project_root: Path, raw_path: Any, location: str) -> Path:
    path_text = _require_string(raw_path, f"{location}.path")
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ManifestError(f"{location}.path does not exist: {candidate}") from error
    if not path.is_file():
        raise ManifestError(f"{location}.path is not a regular file: {path}")
    return path


def _validate_artifact(
    value: Any,
    project_root: Path,
    location: str,
) -> Path:
    artifact = _require_object(value, location, {"path", "sha256"})
    path = _resolve_artifact(project_root, artifact["path"], location)
    declared = _require_sha256(artifact["sha256"], f"{location}.sha256")
    actual = sha256_file(path)
    if actual != declared:
        raise ManifestError(
            f"{location}.sha256 mismatch for {path}: "
            f"declared {declared}, actual {actual}"
        )
    return path


def _load_checkpoint_meta(checkpoint_path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise ManifestError(
            "PyTorch is required to inspect checkpoint metadata"
        ) from error

    try:
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise ManifestError(
            f"cannot safely load checkpoint metadata from {checkpoint_path}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise ManifestError(f"checkpoint root is not a mapping: {checkpoint_path}")
    meta = checkpoint.get("meta")
    if not isinstance(meta, Mapping):
        raise ManifestError(f"checkpoint has no mapping metadata: {checkpoint_path}")
    return meta


def _validate_checkpoint(
    value: Any,
    project_root: Path,
    run_seed: int,
    location: str,
) -> Path:
    checkpoint = _require_object(
        value,
        location,
        {"path", "size_bytes", "sha256", "meta"},
    )
    raw_path = _require_string(checkpoint["path"], f"{location}.path")
    if Path(raw_path).name != CHECKPOINT_BASENAME:
        raise ManifestError(f"{location}.path basename must be {CHECKPOINT_BASENAME!r}")
    path = _resolve_artifact(project_root, raw_path, location)

    size_bytes = _require_int(checkpoint["size_bytes"], f"{location}.size_bytes")
    if size_bytes <= 0:
        raise ManifestError(f"{location}.size_bytes must be positive")
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise ManifestError(f"cannot stat checkpoint {path}: {error}") from error
    if actual_size != size_bytes:
        raise ManifestError(
            f"{location}.size_bytes mismatch for {path}: "
            f"declared {size_bytes}, actual {actual_size}"
        )

    declared_sha256 = _require_sha256(checkpoint["sha256"], f"{location}.sha256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != declared_sha256:
        raise ManifestError(
            f"{location}.sha256 mismatch for {path}: "
            f"declared {declared_sha256}, actual {actual_sha256}"
        )

    declared_meta = _require_object(
        checkpoint["meta"],
        f"{location}.meta",
        {"iter", "epoch", "seed"},
    )
    declared_iter = _require_int(declared_meta["iter"], f"{location}.meta.iter")
    declared_epoch = _require_int(declared_meta["epoch"], f"{location}.meta.epoch")
    declared_seed = _require_int(declared_meta["seed"], f"{location}.meta.seed")
    if declared_iter != CHECKPOINT_ITER:
        raise ManifestError(f"{location}.meta.iter must equal {CHECKPOINT_ITER}")
    if declared_epoch != CHECKPOINT_EPOCH:
        raise ManifestError(f"{location}.meta.epoch must equal {CHECKPOINT_EPOCH}")
    if declared_seed != run_seed:
        raise ManifestError(f"{location}.meta.seed must equal run.seed")

    actual_meta = _load_checkpoint_meta(path)
    for key, expected in (
        ("iter", CHECKPOINT_ITER),
        ("epoch", CHECKPOINT_EPOCH),
        ("seed", run_seed),
    ):
        actual = actual_meta.get(key)
        if type(actual) is not int or actual != expected:
            raise ManifestError(
                f"{location} metadata {key!r} mismatch for {path}: "
                f"expected {expected}, actual {actual!r}"
            )
    return path


def _validate_output_dir(
    value: Any,
    project_root: Path,
    location: str,
    *,
    require_empty: bool,
) -> Path:
    path_text = _require_string(value, location)
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    if candidate.is_symlink():
        raise ManifestError(f"{location} must not be a symbolic link: {candidate}")
    try:
        path = candidate.resolve(strict=False)
        if path.exists():
            if not path.is_dir():
                raise ManifestError(f"{location} is not a directory: {path}")
            if require_empty and next(path.iterdir(), None) is not None:
                raise ManifestError(
                    f"{location} must not already contain results: {path}"
                )
    except ManifestError:
        raise
    except (OSError, RuntimeError) as error:
        raise ManifestError(
            f"cannot inspect {location} {candidate}: {error}"
        ) from error
    return path


def _validate_aggregation(value: Any) -> tuple[set[int], str]:
    aggregation = _require_object(
        value,
        "aggregation",
        {
            "required_seed_set",
            "paired_comparator",
            "exclude_failed_seed",
            "mean",
            "sample_std_ddof",
            "bootstrap_samples",
            "bootstrap_seed",
            "paired_t_test",
            "exact_sign_flip_permutation",
            "cohens_dz",
            "holm_correction",
        },
    )
    seed_values = _require_nonempty_list(
        aggregation["required_seed_set"], "aggregation.required_seed_set"
    )
    required_seeds: set[int] = set()
    for index, value in enumerate(seed_values):
        seed = _require_int(value, f"aggregation.required_seed_set[{index}]")
        if seed < 0:
            raise ManifestError(
                "aggregation.required_seed_set seeds must be non-negative"
            )
        if seed in required_seeds:
            raise ManifestError(
                f"aggregation.required_seed_set contains duplicate seed {seed}"
            )
        required_seeds.add(seed)

    comparator = _require_string(
        aggregation["paired_comparator"], "aggregation.paired_comparator"
    )
    if _require_bool(
        aggregation["exclude_failed_seed"], "aggregation.exclude_failed_seed"
    ):
        raise ManifestError("aggregation.exclude_failed_seed must be false")
    if not _require_bool(aggregation["mean"], "aggregation.mean"):
        raise ManifestError("aggregation.mean must be true")
    ddof = _require_int(aggregation["sample_std_ddof"], "aggregation.sample_std_ddof")
    if ddof != 1:
        raise ManifestError("aggregation.sample_std_ddof must equal 1")
    bootstrap_samples = _require_int(
        aggregation["bootstrap_samples"], "aggregation.bootstrap_samples"
    )
    if bootstrap_samples <= 0:
        raise ManifestError("aggregation.bootstrap_samples must be positive")
    bootstrap_seed = _require_int(
        aggregation["bootstrap_seed"], "aggregation.bootstrap_seed"
    )
    if bootstrap_seed < 0:
        raise ManifestError("aggregation.bootstrap_seed must be non-negative")
    for field in (
        "paired_t_test",
        "exact_sign_flip_permutation",
        "cohens_dz",
        "holm_correction",
    ):
        if not _require_bool(aggregation[field], f"aggregation.{field}"):
            raise ManifestError(f"aggregation.{field} must be true")
    return required_seeds, comparator


def validate_manifest(
    payload: Any,
    *,
    require_empty_output_dirs: bool = True,
) -> tuple[set[Path], set[Path]]:
    """Recursively validate the schema and all declared local identities.

    ``require_empty_output_dirs`` is true while freezing a bundle.  A later
    integrity verification must permit already generated results while still
    validating every frozen input and the output-directory identities.
    """

    root = _require_object(
        payload,
        "manifest",
        {
            "schema_version",
            "state",
            "project_root",
            "arms",
            "aggregation",
            "selection",
        },
    )
    schema_version = _require_int(root["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must equal {SCHEMA_VERSION}")
    state = _require_string(root["state"], "state")
    if state != FROZEN_STATE:
        raise ManifestError(f"state must equal {FROZEN_STATE!r}")

    project_root_text = _require_string(root["project_root"], "project_root")
    project_root_raw = Path(project_root_text)
    if not project_root_raw.is_absolute():
        raise ManifestError("project_root must be an absolute path")
    try:
        project_root = project_root_raw.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ManifestError(
            f"project_root does not exist: {project_root_raw}"
        ) from error
    if not project_root.is_dir():
        raise ManifestError(f"project_root is not a directory: {project_root}")

    required_seeds, comparator = _validate_aggregation(root["aggregation"])
    arms = _require_nonempty_list(root["arms"], "arms")
    if len(arms) < 2:
        raise ManifestError("arms must contain at least two comparison methods")

    artifact_paths: set[Path] = set()
    checkpoint_paths: set[Path] = set()
    output_dirs: set[Path] = set()
    method_names: set[str] = set()
    for arm_index, value in enumerate(arms):
        arm_location = f"arms[{arm_index}]"
        arm = _require_object(
            value,
            arm_location,
            {"method", "method_environment", "hyperparameters", "runs"},
        )
        method = _require_string(arm["method"], f"{arm_location}.method")
        if method in method_names:
            raise ManifestError(f"arms contains duplicate method {method!r}")
        method_names.add(method)
        _validate_method_environment(
            arm["method_environment"], f"{arm_location}.method_environment"
        )
        _validate_hyperparameters(
            arm["hyperparameters"], f"{arm_location}.hyperparameters"
        )

        runs = _require_nonempty_list(arm["runs"], f"{arm_location}.runs")
        run_seeds: set[int] = set()
        for run_index, value in enumerate(runs):
            run_location = f"{arm_location}.runs[{run_index}]"
            run = _require_object(
                value,
                run_location,
                {"seed", "training_fingerprint", "checkpoint", "output_dir"},
            )
            seed = _require_int(run["seed"], f"{run_location}.seed")
            if seed < 0:
                raise ManifestError(f"{run_location}.seed must be non-negative")
            if seed in run_seeds:
                raise ManifestError(
                    f"{arm_location}.runs contains duplicate seed {seed}"
                )
            run_seeds.add(seed)
            _require_sha256(
                run["training_fingerprint"],
                f"{run_location}.training_fingerprint",
            )
            checkpoint_path = _validate_checkpoint(
                run["checkpoint"],
                project_root,
                seed,
                f"{run_location}.checkpoint",
            )
            if checkpoint_path in checkpoint_paths:
                raise ManifestError(
                    f"checkpoint path is reused across runs: {checkpoint_path}"
                )
            checkpoint_paths.add(checkpoint_path)
            artifact_paths.add(checkpoint_path)

            output_dir = _validate_output_dir(
                run["output_dir"],
                project_root,
                f"{run_location}.output_dir",
                require_empty=require_empty_output_dirs,
            )
            if output_dir in output_dirs:
                raise ManifestError(f"output_dir is reused across runs: {output_dir}")
            output_dirs.add(output_dir)

        if run_seeds != required_seeds:
            missing = sorted(required_seeds - run_seeds)
            extra = sorted(run_seeds - required_seeds)
            raise ManifestError(
                f"{arm_location}.runs must exactly cover required_seed_set; "
                f"missing={missing}, extra={extra}"
            )

    if comparator not in method_names:
        raise ManifestError(
            f"aggregation.paired_comparator does not name an arm: {comparator!r}"
        )

    selection = _require_object(
        root["selection"],
        "selection",
        {"evidence", "config", "runtime_files", "data_manifest"},
    )
    evidence = _require_nonempty_list(selection["evidence"], "selection.evidence")
    for index, artifact in enumerate(evidence):
        artifact_paths.add(
            _validate_artifact(
                artifact,
                project_root,
                f"selection.evidence[{index}]",
            )
        )
    artifact_paths.add(
        _validate_artifact(selection["config"], project_root, "selection.config")
    )
    runtime_files = _require_nonempty_list(
        selection["runtime_files"], "selection.runtime_files"
    )
    for index, artifact in enumerate(runtime_files):
        artifact_paths.add(
            _validate_artifact(
                artifact,
                project_root,
                f"selection.runtime_files[{index}]",
            )
        )
    artifact_paths.add(
        _validate_artifact(
            selection["data_manifest"],
            project_root,
            "selection.data_manifest",
        )
    )
    return artifact_paths, output_dirs


def _resolved_output_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ManifestError(f"cannot resolve output path {path}: {error}") from error


def create_manifest(draft_path: Path, output_path: Path) -> str:
    payload = load_json(draft_path)
    protected_paths, planned_output_dirs = validate_manifest(payload)
    try:
        protected_paths.add(draft_path.resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise ManifestError(
            f"cannot resolve draft path {draft_path}: {error}"
        ) from error
    resolved_output = _resolved_output_path(output_path)
    resolved_sidecar = _resolved_output_path(sidecar_path(output_path))
    collisions = protected_paths.intersection({resolved_output, resolved_sidecar})
    if collisions:
        collision = sorted(str(path) for path in collisions)[0]
        raise ManifestError(f"output would overwrite a frozen input: {collision}")
    for planned_dir in planned_output_dirs:
        for manifest_artifact in (resolved_output, resolved_sidecar):
            if (
                manifest_artifact == planned_dir
                or manifest_artifact.is_relative_to(planned_dir)
                or planned_dir.is_relative_to(manifest_artifact)
            ):
                raise ManifestError(
                    "manifest output conflicts with a planned test output: "
                    f"{manifest_artifact} vs {planned_dir}"
                )
    content = canonical_json_bytes(payload)
    digest = hashlib.sha256(content).hexdigest()
    return _publish_frozen_bundle(
        output_path,
        content,
        f"{digest}  {output_path.name}\n".encode("utf-8"),
        digest,
    )


def verify_manifest(manifest_path: Path) -> str:
    _require_regular_non_symlink(manifest_path, "frozen manifest")
    _require_regular_non_symlink(sidecar_path(manifest_path), "frozen manifest sidecar")
    try:
        content = manifest_path.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read {manifest_path}: {error}") from error

    digest = hashlib.sha256(content).hexdigest()
    expected_sidecar = f"{digest}  {manifest_path.name}\n".encode("utf-8")
    checksum_path = sidecar_path(manifest_path)
    try:
        actual_sidecar = checksum_path.read_bytes()
    except OSError as error:
        raise ManifestError(f"cannot read {checksum_path}: {error}") from error
    if actual_sidecar != expected_sidecar:
        raise ManifestError(f"manifest SHA-256 sidecar mismatch: {checksum_path}")

    payload = _load_json_bytes(content, manifest_path)
    if content != canonical_json_bytes(payload):
        raise ManifestError(f"manifest is not canonical JSON: {manifest_path}")
    validate_manifest(payload, require_empty_output_dirs=False)
    return digest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="validate a draft and freeze it")
    create.add_argument("--draft", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="verify a frozen manifest")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if args.command == "create":
            digest = create_manifest(args.draft, args.output)
            print(f"created {args.output} sha256={digest}")
        else:
            digest = verify_manifest(args.manifest)
            print(f"verified {args.manifest} sha256={digest}")
    except ManifestError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
