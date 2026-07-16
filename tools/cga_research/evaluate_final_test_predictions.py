#!/usr/bin/env python3
"""Evaluate one manifest-authorized final-test ``predictions.pkl``.

This is the only CGA research entry point that permits dataset paths with an
exact ``test`` component.  Authorization is fail closed: the canonical frozen
manifest and detached sidecar, every frozen artifact, the byte-level dataset
manifest, the unique runner completion record, registered runtime tools, and
output paths are verified before pickle is loaded.  The manifest schema does
not register a prediction filename, so it is fixed to
``<run.output_dir>/predictions.pkl``.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import pickle
import stat
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cga_research import build_data_manifest as data_manifest_tool  # noqa: E402
from tools.cga_research import evaluate_predictions as offline_evaluator  # noqa: E402
from tools.cga_research import final_test_manifest as manifest_tool  # noqa: E402
from tools.cga_research import run_final_test_bundle as runner_tool  # noqa: E402
from tools.cga_research.evaluate_predictions import (  # noqa: E402
    OfflineEvaluationError,
    RSAR_CLASSES,
    _jsonable,
    evaluate_validated_predictions,
    force_disable_cga,
    sha256_bytes,
    sha256_file,
)
from tools.cga_research.final_test_manifest import (  # noqa: E402
    ManifestError,
    canonical_json_bytes,
    load_json,
    sidecar_path,
    verify_manifest,
)


PREDICTIONS_BASENAME = "predictions.pkl"
POSTPROCESS_BASENAME = runner_tool.POSTPROCESS_BASENAME
DEFAULT_IOU_THRESHOLD = 0.5
REQUIRED_RUNTIME_FILES = (
    Path(__file__).resolve(),
    Path(offline_evaluator.__file__).resolve(),
    Path(manifest_tool.__file__).resolve(),
    Path(data_manifest_tool.__file__).resolve(),
    Path(runner_tool.__file__).resolve(),
)


class FinalTestEvaluationError(RuntimeError):
    """Raised when final-test authorization or path binding fails."""


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create a JSON result without ever replacing a prior file."""

    rendered = (
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        # Hard-link publication is atomic and fails if `path` appeared after
        # authorization; unlike os.replace, it never clobbers prior evidence.
        os.link(temporary, path)
    except FileExistsError as error:
        raise FinalTestEvaluationError(
            f"refusing to replace an existing final-test result: {path}"
        ) from error
    except OSError as error:
        raise FinalTestEvaluationError(
            f"cannot atomically create final-test result {path}: {error}"
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


@dataclasses.dataclass(frozen=True)
class FinalTestAuthorization:
    manifest_path: Path
    manifest_sha256: str
    manifest_payload: Mapping[str, Any]
    project_root: Path
    method: str
    seed: int
    arm: Mapping[str, Any]
    run: Mapping[str, Any]
    output_dir: Path
    predictions_path: Path
    output_path: Path
    config_path: Path
    data_manifest_path: Path
    data_manifest_sha256: str
    data_manifest_payload: Mapping[str, Any]
    annotation_root: Path
    image_root: Path
    corruption: str
    registry_path: Path
    completion_record: Mapping[str, Any]
    predictions_sha256: str
    predictions_size_bytes: int


def _lexical_absolute(path: Path) -> Path:
    """Normalize ``.``/``..`` without accepting a symlink alias."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolve_project_path(project_root: Path, value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FinalTestEvaluationError(f"{role} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        return candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestEvaluationError(f"{role} does not exist: {candidate}") from error


def _resolve_declared_output_dir(project_root: Path, value: Any, role: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise FinalTestEvaluationError(f"{role} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    if candidate.is_symlink():
        raise FinalTestEvaluationError(f"{role} must not be a symlink: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestEvaluationError(f"{role} does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise FinalTestEvaluationError(f"{role} is not a directory: {resolved}")
    return resolved


def _verified_manifest_payload(manifest_path: Path) -> tuple[str, dict[str, Any]]:
    """Fully verify a frozen manifest, then bind the payload to that digest."""

    lexical_manifest = _lexical_absolute(manifest_path)
    checksum_path = sidecar_path(lexical_manifest)
    for path, role in (
        (lexical_manifest, "manifest"),
        (checksum_path, "manifest sidecar"),
    ):
        if path.is_symlink():
            raise FinalTestEvaluationError(f"{role} must not be a symlink: {path}")
        if not path.is_file():
            raise FinalTestEvaluationError(f"{role} is missing: {path}")

    # This verifies canonical JSON, the detached sidecar, schema, checkpoint
    # metadata, and every frozen selection/runtime/data artifact.
    digest = verify_manifest(lexical_manifest)
    payload = load_json(lexical_manifest)
    payload_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if payload_digest != digest or sha256_file(lexical_manifest) != digest:
        raise FinalTestEvaluationError("manifest changed during verification")
    expected_sidecar = f"{digest}  {lexical_manifest.name}\n"
    try:
        actual_sidecar = checksum_path.read_text(encoding="utf-8")
    except OSError as error:
        raise FinalTestEvaluationError(
            f"cannot re-read manifest sidecar: {checksum_path}"
        ) from error
    if actual_sidecar != expected_sidecar:
        raise FinalTestEvaluationError("manifest sidecar changed during verification")
    return digest, payload


def _select_unique_run(
    payload: Mapping[str, Any], method: str, seed: int
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for arm in payload.get("arms", []):
        if isinstance(arm, Mapping) and arm.get("method") == method:
            for run in arm.get("runs", []):
                if isinstance(run, Mapping) and run.get("seed") == seed:
                    matches.append((arm, run))
    if len(matches) != 1:
        raise FinalTestEvaluationError(
            f"method={method!r}, seed={seed} must match exactly one frozen run; "
            f"found {len(matches)}"
        )
    return matches[0]


def _registered_artifact_paths(
    payload: Mapping[str, Any], project_root: Path
) -> set[Path]:
    selection = payload["selection"]
    entries = list(selection["runtime_files"])
    registered: set[Path] = set()
    for index, entry in enumerate(entries):
        registered.add(
            _resolve_project_path(
                project_root,
                entry["path"],
                f"selection.runtime_files[{index}].path",
            )
        )
    return registered


def _require_runtime_tools_frozen(
    payload: Mapping[str, Any], project_root: Path
) -> None:
    registered = _registered_artifact_paths(payload, project_root)
    missing = [path for path in REQUIRED_RUNTIME_FILES if path not in registered]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FinalTestEvaluationError(
            f"frozen manifest does not register required runtime files: {rendered}"
        )


def _verify_data_manifest_selection(
    *,
    payload: Mapping[str, Any],
    project_root: Path,
    arm: Mapping[str, Any],
) -> tuple[Path, str, Mapping[str, Any], Path, Path, str]:
    """Verify current dataset bytes and bind their semantics to one arm."""

    selection = payload["selection"]
    data_manifest_path = _resolve_project_path(
        project_root,
        selection["data_manifest"]["path"],
        "selection.data_manifest.path",
    )
    declared_digest = selection["data_manifest"]["sha256"]
    try:
        verified_digest = data_manifest_tool.verify_manifest(data_manifest_path)
    except data_manifest_tool.DataManifestError as error:
        raise FinalTestEvaluationError(
            f"dataset manifest verification failed: {error}"
        ) from error
    if verified_digest != declared_digest:
        raise FinalTestEvaluationError(
            "verified dataset manifest digest differs from frozen selection"
        )
    data_payload = manifest_tool.load_json(data_manifest_path)
    if sha256_file(data_manifest_path) != declared_digest:
        raise FinalTestEvaluationError(
            "dataset manifest changed after byte-level verification"
        )

    split = data_payload.get("split")
    if split != "test":
        raise FinalTestEvaluationError(
            f"dataset manifest split must equal 'test'; got {split!r}"
        )
    corruption = data_payload.get("corruption")
    arm_corruption = arm.get("hyperparameters", {}).get("corrupt")
    if not isinstance(corruption, str) or corruption != arm_corruption:
        raise FinalTestEvaluationError(
            "dataset manifest corruption must exactly match "
            "arm.hyperparameters.corrupt"
        )
    class_order = data_payload.get("class_order")
    if class_order != list(RSAR_CLASSES):
        raise FinalTestEvaluationError(
            "dataset manifest class_order mismatch: expected "
            f"{list(RSAR_CLASSES)!r}, got {class_order!r}"
        )
    try:
        annotation_value = data_payload["annotations"]["root"]
        image_value = data_payload["images"]["root"]
    except (KeyError, TypeError) as error:
        raise FinalTestEvaluationError(
            "dataset manifest is missing annotations.root/images.root"
        ) from error
    annotation_root = _resolve_final_dataset_path(annotation_value, None, "ann_file")
    image_root = _resolve_final_dataset_path(image_value, None, "img_prefix")
    return (
        data_manifest_path,
        verified_digest,
        data_payload,
        annotation_root,
        image_root,
        corruption,
    )


def _verify_completed_prediction(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    project_root: Path,
    method: str,
    seed: int,
    run: Mapping[str, Any],
    output_dir: Path,
    predictions_path: Path,
) -> tuple[Path, Mapping[str, Any], str, int]:
    """Bind predictions to the runner's unique verified completion record."""

    registry_path_factory = getattr(runner_tool, "registry_path_for_manifest", None)
    completion_verifier = getattr(runner_tool, "verify_completed_run", None)
    if not callable(registry_path_factory) or not callable(completion_verifier):
        raise FinalTestEvaluationError(
            "runner completion-verification interface is unavailable"
        )
    try:
        registry_path = registry_path_factory(manifest_path)
        record = completion_verifier(manifest_path, method, seed)
    except runner_tool.FinalTestBundleError as error:
        raise FinalTestEvaluationError(
            f"runner completion verification failed: {error}"
        ) from error
    expected_registry_path = (
        _lexical_absolute(manifest_path).parent / runner_tool.REGISTRY_BASENAME
    )
    registry_path = _lexical_absolute(Path(registry_path))
    if registry_path != expected_registry_path:
        raise FinalTestEvaluationError(
            "runner registry path is not the fixed manifest-local registry"
        )
    if not isinstance(record, Mapping):
        raise FinalTestEvaluationError(
            "runner completion verifier returned a non-object record"
        )

    checkpoint = run["checkpoint"]
    checkpoint_path = _resolve_project_path(
        project_root, checkpoint["path"], "selected run.checkpoint.path"
    )
    expected = {
        "event": "completed",
        "manifest_path": str(_lexical_absolute(manifest_path)),
        "manifest_sha256": manifest_sha256,
        "method": method,
        "seed": seed,
        "training_fingerprint": run["training_fingerprint"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint["sha256"],
        "output_dir": str(output_dir),
        "returncode": 0,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise FinalTestEvaluationError(
                f"completed registry {field} differs from frozen run"
            )
    artifacts = record.get("artifacts")
    prediction_identity = (
        artifacts.get(PREDICTIONS_BASENAME) if isinstance(artifacts, Mapping) else None
    )
    if not isinstance(prediction_identity, Mapping):
        raise FinalTestEvaluationError(
            "completed registry lacks predictions.pkl identity"
        )
    declared_path = prediction_identity.get("path")
    declared_digest = prediction_identity.get("sha256")
    declared_size = prediction_identity.get("size_bytes")
    if declared_path != str(predictions_path):
        raise FinalTestEvaluationError(
            "completed registry predictions.pkl path mismatch"
        )
    if (
        not isinstance(declared_digest, str)
        or len(declared_digest) != 64
        or any(character not in "0123456789abcdef" for character in declared_digest)
    ):
        raise FinalTestEvaluationError(
            "completed registry predictions.pkl SHA-256 is invalid"
        )
    if type(declared_size) is not int or declared_size <= 0:
        raise FinalTestEvaluationError(
            "completed registry predictions.pkl size is invalid"
        )
    actual_size = predictions_path.stat().st_size
    actual_digest = sha256_file(predictions_path)
    if actual_size != declared_size or actual_digest != declared_digest:
        raise FinalTestEvaluationError(
            "predictions.pkl differs from the verified completed registry"
        )
    return registry_path, record, actual_digest, actual_size


@contextmanager
def _verified_prediction_handle(
    authorization: FinalTestAuthorization,
) -> Iterator[tuple[BinaryIO, str, int]]:
    """Open, hash, and yield one exact non-symlink prediction descriptor."""

    path = authorization.predictions_path
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise FinalTestEvaluationError(
            "this platform cannot securely open predictions with O_NOFOLLOW"
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise FinalTestEvaluationError(
            f"cannot securely open authorized predictions: {path}: {error}"
        ) from error

    with os.fdopen(descriptor, "rb") as source, tempfile.TemporaryFile(
        mode="w+b"
    ) as handle:
        try:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise FinalTestEvaluationError(
                    f"authorized predictions is no longer a regular file: {path}"
                )
            size = 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                handle.write(chunk)
                size += len(chunk)
            after = os.fstat(source.fileno())
            handle.flush()
            snapshot = os.fstat(handle.fileno())
            handle.seek(0)
            digest_builder = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        except OSError as error:
            raise FinalTestEvaluationError(
                f"cannot verify authorized predictions descriptor: {path}: {error}"
            ) from error
        stable_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        digest = digest_builder.hexdigest()
        if (
            stable_identity != after_identity
            or size != before.st_size
            or snapshot.st_size != size
            or size != authorization.predictions_size_bytes
            or digest != authorization.predictions_sha256
        ):
            raise FinalTestEvaluationError(
                "predictions.pkl changed after runner completion authorization"
            )
        handle.seek(0)
        yield handle, digest, size


def _reverify_prediction_identity(
    authorization: FinalTestAuthorization,
) -> tuple[str, int]:
    """Reverify the exact descriptor without deserializing it."""

    with _verified_prediction_handle(authorization) as (_, digest, size):
        return digest, size


def _load_verified_prediction_pickle(
    authorization: FinalTestAuthorization,
) -> tuple[Any, str, int]:
    """Deserialize from the same descriptor whose bytes passed identity checks."""

    with _verified_prediction_handle(authorization) as (handle, digest, size):
        print(
            "[security] --trust-pickle acknowledged; loading manifest-authorized "
            f"predictions from verified descriptor: {authorization.predictions_path}",
            file=sys.stderr,
            flush=True,
        )
        predictions = pickle.load(  # noqa: S301 - manifest/registry/FD authorized
            handle
        )
        return predictions, digest, size


def authorize_final_test_run(
    *,
    manifest_path: Path,
    method: str,
    seed: int,
    predictions_path: Path,
    output_path: Path,
) -> FinalTestAuthorization:
    """Verify the frozen bundle and authorize one exact generated prediction."""

    if not method or "\x00" in method:
        raise FinalTestEvaluationError("--method must be a non-empty string")
    if seed < 0:
        raise FinalTestEvaluationError("--seed must be non-negative")

    manifest_sha256, payload = _verified_manifest_payload(manifest_path)
    project_root = _resolve_project_path(
        Path("/"), payload["project_root"], "manifest.project_root"
    )
    if not project_root.is_dir():
        raise FinalTestEvaluationError(
            f"manifest.project_root is not a directory: {project_root}"
        )
    _require_runtime_tools_frozen(payload, project_root)
    arm, run = _select_unique_run(payload, method, seed)

    (
        data_manifest_path,
        data_manifest_sha256,
        data_manifest_payload,
        annotation_root,
        image_root,
        corruption,
    ) = _verify_data_manifest_selection(
        payload=payload,
        project_root=project_root,
        arm=arm,
    )

    output_dir = _resolve_declared_output_dir(
        project_root, run["output_dir"], "selected run.output_dir"
    )
    expected_predictions = output_dir / PREDICTIONS_BASENAME
    requested_predictions = _lexical_absolute(predictions_path)
    if requested_predictions != expected_predictions:
        raise FinalTestEvaluationError(
            "--predictions must exactly equal the frozen run path "
            f"{expected_predictions}; got {requested_predictions}"
        )
    if expected_predictions.is_symlink():
        raise FinalTestEvaluationError(
            f"predictions must not be a symlink: {expected_predictions}"
        )
    if not expected_predictions.is_file():
        raise FinalTestEvaluationError(
            f"registered predictions file is missing: {expected_predictions}"
        )

    (
        registry_path,
        completion_record,
        predictions_sha256,
        predictions_size_bytes,
    ) = _verify_completed_prediction(
        manifest_path=_lexical_absolute(manifest_path),
        manifest_sha256=manifest_sha256,
        project_root=project_root,
        method=method,
        seed=seed,
        run=run,
        output_dir=output_dir,
        predictions_path=expected_predictions,
    )

    requested_output = _lexical_absolute(output_path)
    if requested_output.parent != output_dir:
        raise FinalTestEvaluationError(
            f"--output must be a direct child of {output_dir}: {requested_output}"
        )
    if requested_output.suffix.casefold() != ".json":
        raise FinalTestEvaluationError("--output must end with .json")
    if requested_output.name != POSTPROCESS_BASENAME:
        raise FinalTestEvaluationError(
            f"--output basename must equal {POSTPROCESS_BASENAME!r}"
        )
    if requested_output.name == PREDICTIONS_BASENAME:
        raise FinalTestEvaluationError("--output must not replace predictions.pkl")
    if requested_output.is_symlink() or requested_output.exists():
        raise FinalTestEvaluationError(
            f"--output must be a new non-symlink path: {requested_output}"
        )

    selection = payload["selection"]
    config_path = _resolve_project_path(
        project_root, selection["config"]["path"], "selection.config.path"
    )
    return FinalTestAuthorization(
        manifest_path=_lexical_absolute(manifest_path),
        manifest_sha256=manifest_sha256,
        manifest_payload=payload,
        project_root=project_root,
        method=method,
        seed=seed,
        arm=arm,
        run=run,
        output_dir=output_dir,
        predictions_path=expected_predictions,
        output_path=requested_output,
        config_path=config_path,
        data_manifest_path=data_manifest_path,
        data_manifest_sha256=data_manifest_sha256,
        data_manifest_payload=data_manifest_payload,
        annotation_root=annotation_root,
        image_root=image_root,
        corruption=corruption,
        registry_path=registry_path,
        completion_record=completion_record,
        predictions_sha256=predictions_sha256,
        predictions_size_bytes=predictions_size_bytes,
    )


def _contains_test_component(path: Path) -> bool:
    return any(part.casefold() == "test" for part in path.parts)


def _resolve_final_dataset_path(value: Any, data_root: Any, role: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise FinalTestEvaluationError(
            f"data.test.{role} must be one non-empty filesystem path"
        )
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and data_root:
        candidate = Path(str(data_root)).expanduser() / candidate
    if candidate.is_symlink():
        raise FinalTestEvaluationError(
            f"data.test.{role} must not be a symlink: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestEvaluationError(
            f"data.test.{role} does not exist: {candidate}"
        ) from error
    if not resolved.is_dir():
        raise FinalTestEvaluationError(
            f"data.test.{role} is not a directory: {resolved}"
        )
    if not _contains_test_component(resolved):
        raise FinalTestEvaluationError(
            f"final-test {role} must contain an exact 'test' path component: "
            f"{resolved}"
        )
    return resolved


def validate_final_test_dataset_paths(test_cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    data_root = test_cfg.get("data_root")
    ann_file = _resolve_final_dataset_path(
        test_cfg.get("ann_file"), data_root, "ann_file"
    )
    img_prefix = _resolve_final_dataset_path(
        test_cfg.get("img_prefix"), data_root, "img_prefix"
    )
    return ann_file, img_prefix


def _manifest_bound_test_config(
    test_cfg: Mapping[str, Any],
    *,
    annotation_root: Path,
    image_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    """Return a test config whose data identity comes only from the manifest."""

    bound = copy.deepcopy(dict(test_cfg))
    if bound.get("type") != "DOTADataset":
        raise FinalTestEvaluationError(
            "frozen data.test.type must equal 'DOTADataset'; dataset wrappers "
            "cannot be bound safely to one manifest snapshot"
        )
    bound["test_mode"] = True
    bound.pop("data_root", None)
    bound.pop("proposal_file", None)
    bound["ann_file"] = str(annotation_root)
    bound["img_prefix"] = str(image_root)
    bound["classes"] = tuple(RSAR_CLASSES)
    ann_file, img_prefix = validate_final_test_dataset_paths(bound)
    return bound, ann_file, img_prefix


def _declared_artifact_hash(payload: Mapping[str, Any], section: str) -> str:
    return str(payload["selection"][section]["sha256"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Authorize, build the frozen test dataset, then evaluate predictions."""

    cga_environment = force_disable_cga()
    if not args.trust_pickle:
        raise FinalTestEvaluationError(
            "--trust-pickle is required before any pickle bytes are loaded"
        )
    if args.nproc < 1:
        raise FinalTestEvaluationError("--nproc must be at least 1")

    # This complete manifest/artifact authorization intentionally occurs
    # before importing config code or loading any pickle bytes.
    authorization = authorize_final_test_run(
        manifest_path=args.manifest,
        method=args.method,
        seed=args.seed,
        predictions_path=args.predictions,
        output_path=args.output,
    )
    frozen_payload = authorization.manifest_payload

    import mmcv
    import mmdet
    import mmrotate
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.core import eval_rbbox_map
    from mmrotate.datasets import build_dataset
    from sfod.utils import patch_config

    cfg = Config.fromfile(str(authorization.config_path))
    hyperparameters = dict(authorization.arm["hyperparameters"])
    if hyperparameters:
        cfg.merge_from_dict(hyperparameters)
    if "data" not in cfg or "test" not in cfg.data:
        raise FinalTestEvaluationError("frozen config must define data.test")
    if not isinstance(cfg.data.test, Mapping):
        raise FinalTestEvaluationError("frozen data.test must be one dataset mapping")
    cfg = patch_config(cfg)
    if cfg.get("corrupt") != authorization.corruption:
        raise FinalTestEvaluationError(
            "effective config corruption differs from verified dataset manifest"
        )
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    # The byte-verified dataset manifest is authoritative.  The frozen config
    # supplies dataset type/pipeline/version only; it cannot redirect final-test
    # evaluation to a different annotation or image tree.
    test_cfg, ann_file, img_prefix = _manifest_bound_test_config(
        cfg.data.test,
        annotation_root=authorization.annotation_root,
        image_root=authorization.image_root,
    )
    dataset = build_dataset(test_cfg)

    (
        predictions,
        predictions_sha256,
        predictions_size_bytes,
    ) = _load_verified_prediction_pickle(
        authorization,
    )
    result = evaluate_validated_predictions(
        dataset=dataset,
        predictions=predictions,
        eval_rbbox_map_fn=eval_rbbox_map,
        iou_thr=DEFAULT_IOU_THRESHOLD,
        nproc=args.nproc,
    )

    checkpoint = authorization.run["checkpoint"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "authorization": {
            "manifest_path": str(authorization.manifest_path),
            "manifest_sha256": authorization.manifest_sha256,
            "method": authorization.method,
            "seed": authorization.seed,
            "training_fingerprint": authorization.run["training_fingerprint"],
            "checkpoint": _jsonable(checkpoint),
            "method_environment": _jsonable(authorization.arm["method_environment"]),
            "hyperparameters": _jsonable(hyperparameters),
            "registered_output_dir": str(authorization.output_dir),
            "registry_path": str(authorization.registry_path),
            "registry_attempt": authorization.completion_record["attempt"],
        },
        "evaluator": {
            "function": "mmrotate.core.eval_rbbox_map",
            "iou_threshold": DEFAULT_IOU_THRESHOLD,
            "nproc": args.nproc,
            "model_built": False,
            "inference_run": False,
            "cga_environment": cga_environment,
        },
        "result": result,
        "dataset": {
            "type": type(dataset).__name__,
            "test_mode": bool(getattr(dataset, "test_mode", False)),
            "ann_file": str(ann_file),
            "img_prefix": str(img_prefix),
            "classes": list(getattr(dataset, "CLASSES", ())),
            "data_manifest_path": str(authorization.data_manifest_path),
            "data_manifest_sha256": authorization.data_manifest_sha256,
        },
        "provenance": {
            "command": list(sys.argv),
            "python_executable": sys.executable,
            "project_root": str(authorization.project_root),
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "offline_evaluator_path": str(REQUIRED_RUNTIME_FILES[1]),
            "offline_evaluator_sha256": sha256_file(REQUIRED_RUNTIME_FILES[1]),
            "manifest_tool_path": str(REQUIRED_RUNTIME_FILES[2]),
            "manifest_tool_sha256": sha256_file(REQUIRED_RUNTIME_FILES[2]),
            "data_manifest_tool_path": str(REQUIRED_RUNTIME_FILES[3]),
            "data_manifest_tool_sha256": sha256_file(REQUIRED_RUNTIME_FILES[3]),
            "runner_tool_path": str(REQUIRED_RUNTIME_FILES[4]),
            "runner_tool_sha256": sha256_file(REQUIRED_RUNTIME_FILES[4]),
            "config_path": str(authorization.config_path),
            "config_sha256": _declared_artifact_hash(frozen_payload, "config"),
            "effective_config_sha256": sha256_bytes(cfg.pretty_text.encode("utf-8")),
            "predictions_path": str(authorization.predictions_path),
            "predictions_sha256": predictions_sha256,
            "predictions_size_bytes": predictions_size_bytes,
            "output_path": str(authorization.output_path),
            "pickle_trust_acknowledged": True,
            "versions": {
                "mmcv": mmcv.__version__,
                "mmdet": mmdet.__version__,
                "mmrotate": mmrotate.__version__,
                "numpy": np.__version__,
            },
        },
    }
    _atomic_create_json(authorization.output_path, report)
    print(
        json.dumps(
            {
                "manifest_sha256": authorization.manifest_sha256,
                "method": authorization.method,
                "seed": authorization.seed,
                "output": str(authorization.output_path),
                "num_images": result["num_images"],
                "mAP": result["mean_ap"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help=(
            "acknowledge that the exact manifest-authorized predictions.pkl "
            "was produced by the trusted frozen inference run"
        ),
    )
    parser.add_argument("--nproc", type=int, default=4)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    force_disable_cga()
    from iraod_runtime import ensure_iraod_runtime

    ensure_iraod_runtime()
    try:
        run(args)
    except (FinalTestEvaluationError, ManifestError, OfflineEvaluationError) as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
