#!/usr/bin/env python3
"""Aggregate one fully frozen final-test bundle without selecting a method.

The only input is a schema-v2 frozen manifest.  Every run must already have a
runner-verified completion record and a fixed ``per_class_ap.json`` produced by
the manifest-authorized offline evaluator.  The command is descriptive only:
it reports all preregistered arms and paired comparisons to the declared
``paired_comparator`` and never ranks or selects a method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.cga_research import build_data_manifest as data_manifest_tool  # noqa: E402
from tools.cga_research import (  # noqa: E402
    evaluate_final_test_predictions as evaluator_tool,
)
from tools.cga_research import final_test_manifest as manifest_tool  # noqa: E402
from tools.cga_research import run_final_test_bundle as runner_tool  # noqa: E402
from tools.cga_research import statistics as statistics_tool  # noqa: E402


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RSAR_CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
OUTPUT_BASENAMES = (
    "final_test_results.csv",
    "final_test_results.jsonl",
    "final_test_statistical_report.md",
    "final_test_statistical_report.json",
)
REQUIRED_RUNTIME_FILES = (
    Path(__file__).resolve(),
    Path(manifest_tool.__file__).resolve(),
    Path(data_manifest_tool.__file__).resolve(),
    Path(runner_tool.__file__).resolve(),
    Path(evaluator_tool.__file__).resolve(),
    Path(statistics_tool.__file__).resolve(),
)


class FinalTestAggregationError(RuntimeError):
    """Raised when a final-test bundle cannot be aggregated fail-closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FinalTestAggregationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _resolve_project_path(project_root: Path, raw: Any, role: str) -> Path:
    if type(raw) is not str or not raw.strip() or "\x00" in raw:
        raise FinalTestAggregationError(f"{role} must be a non-empty path string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise FinalTestAggregationError(
            f"{role} does not exist: {candidate}"
        ) from error
    return resolved


def _require_exact_object(
    value: Any, required: set[str], role: str
) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise FinalTestAggregationError(f"{role} must be a JSON object")
    actual = set(value)
    if actual != required:
        raise FinalTestAggregationError(
            f"{role} fields differ from schema: "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    return value


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalTestAggregationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise FinalTestAggregationError(f"non-finite JSON value: {value}")


def _load_strict_evaluator_report(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink():
        raise FinalTestAggregationError(
            f"per-class report must not be a symlink: {path}"
        )
    try:
        info = path.lstat()
        content = path.read_bytes()
    except OSError as error:
        raise FinalTestAggregationError(
            f"cannot read fixed per-class report {path}: {error}"
        ) from error
    if not path.is_file() or info.st_size <= 0:
        raise FinalTestAggregationError(
            f"fixed per-class report must be a nonempty regular file: {path}"
        )
    try:
        text = content.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except UnicodeDecodeError as error:
        raise FinalTestAggregationError(
            f"per-class report is not UTF-8: {path}"
        ) from error
    except json.JSONDecodeError as error:
        raise FinalTestAggregationError(
            f"per-class report is invalid JSON: {path}: {error}"
        ) from error
    if type(payload) is not dict:
        raise FinalTestAggregationError("per-class report root must be an object")
    try:
        canonical = (
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise FinalTestAggregationError(
            f"per-class report cannot be serialized safely: {error}"
        ) from error
    if text != canonical:
        raise FinalTestAggregationError(
            f"per-class report is not canonical evaluator JSON: {path}"
        )
    return payload, hashlib.sha256(content).hexdigest()


def _verified_frozen_inputs(
    manifest_path: Path,
) -> tuple[str, dict[str, Any], Path, str, dict[str, Any]]:
    manifest_path = _lexical_absolute(manifest_path)
    try:
        manifest_digest = manifest_tool.verify_manifest(manifest_path)
        payload = manifest_tool.load_json(manifest_path)
    except manifest_tool.ManifestError:
        raise
    if _sha256_file(manifest_path) != manifest_digest:
        raise FinalTestAggregationError("frozen manifest changed after verification")

    project_root = _resolve_project_path(
        Path("/"), payload["project_root"], "manifest.project_root"
    )
    if not project_root.is_dir():
        raise FinalTestAggregationError("manifest.project_root is not a directory")
    registered_runtime = {
        _resolve_project_path(
            project_root,
            artifact["path"],
            f"selection.runtime_files[{index}].path",
        )
        for index, artifact in enumerate(payload["selection"]["runtime_files"])
    }
    missing_runtime = [
        path for path in REQUIRED_RUNTIME_FILES if path not in registered_runtime
    ]
    if missing_runtime:
        raise FinalTestAggregationError(
            "frozen manifest does not register aggregator runtime files: "
            + ", ".join(str(path) for path in missing_runtime)
        )

    selection = payload["selection"]
    data_manifest_path = _resolve_project_path(
        project_root,
        selection["data_manifest"]["path"],
        "selection.data_manifest.path",
    )
    try:
        data_digest = data_manifest_tool.verify_manifest(data_manifest_path)
    except data_manifest_tool.DataManifestError:
        raise
    if data_digest != selection["data_manifest"]["sha256"]:
        raise FinalTestAggregationError(
            "verified data manifest digest differs from frozen selection"
        )
    data_payload = manifest_tool.load_json(data_manifest_path)
    if _sha256_file(data_manifest_path) != data_digest:
        raise FinalTestAggregationError("data manifest changed after verification")
    if data_payload.get("split") != "test":
        raise FinalTestAggregationError("data manifest split must equal 'test'")
    if data_payload.get("class_order") != list(RSAR_CLASSES):
        raise FinalTestAggregationError(
            "data manifest class_order is not fixed RSAR order"
        )
    return manifest_digest, payload, data_manifest_path, data_digest, data_payload


def _finite_metric(value: Any, role: str) -> float:
    if isinstance(value, bool):
        raise FinalTestAggregationError(f"{role} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FinalTestAggregationError(f"{role} must be a finite number") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise FinalTestAggregationError(f"{role} must be in [0, 1]")
    return result


def _nonnegative_int(value: Any, role: str) -> int:
    if type(value) is not int or value < 0:
        raise FinalTestAggregationError(f"{role} must be a nonnegative integer")
    return value


def _sha256(value: Any, role: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise FinalTestAggregationError(f"{role} must be a lowercase SHA-256")
    return value


def _utc_timestamp(value: Any, role: str) -> str:
    if type(value) is not str or not value:
        raise FinalTestAggregationError(f"{role} must be an ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise FinalTestAggregationError(
            f"{role} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise FinalTestAggregationError(f"{role} must include the UTC offset")
    return value


def _validate_per_class_report(
    *,
    report: Mapping[str, Any],
    report_path: Path,
    report_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    data_manifest_path: Path,
    data_manifest_sha256: str,
    data_manifest_payload: Mapping[str, Any],
    project_root: Path,
    selection: Mapping[str, Any],
    arm: Mapping[str, Any],
    run: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> dict[str, Any]:
    _require_exact_object(
        report,
        {
            "schema_version",
            "generated_at_utc",
            "status",
            "authorization",
            "evaluator",
            "result",
            "dataset",
            "provenance",
        },
        "per-class report",
    )
    if report["schema_version"] != 1 or report["status"] != "completed":
        raise FinalTestAggregationError(
            "per-class report must be schema_version=1 and status=completed"
        )
    _utc_timestamp(report["generated_at_utc"], "per-class report.generated_at_utc")
    authorization = _require_exact_object(
        report["authorization"],
        {
            "manifest_path",
            "manifest_sha256",
            "method",
            "seed",
            "training_fingerprint",
            "checkpoint",
            "method_environment",
            "hyperparameters",
            "registered_output_dir",
            "registry_path",
            "registry_attempt",
        },
        "per-class report.authorization",
    )
    expected_output_dir = _resolve_project_path(
        project_root, run["output_dir"], "selected run.output_dir"
    )
    expected_registry = runner_tool.registry_path_for_manifest(manifest_path)
    expected_checkpoint_path = _resolve_project_path(
        project_root, run["checkpoint"]["path"], "selected run.checkpoint.path"
    )
    expected_completion = {
        "event": "completed",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "method": arm["method"],
        "seed": run["seed"],
        "training_fingerprint": run["training_fingerprint"],
        "checkpoint_path": str(expected_checkpoint_path),
        "checkpoint_sha256": run["checkpoint"]["sha256"],
        "output_dir": str(expected_output_dir),
        "returncode": 0,
        "gpu_verified": True,
    }
    for field, expected in expected_completion.items():
        if completion.get(field) != expected:
            raise FinalTestAggregationError(
                f"verified completion record {field} mismatch"
            )
    if type(completion.get("attempt")) is not int or completion["attempt"] < 1:
        raise FinalTestAggregationError(
            "verified completion record attempt must be positive"
        )
    _nonnegative_int(completion.get("gpu_index"), "completion.gpu_index")
    for field in ("gpu_uuid", "gpu_name"):
        if type(completion.get(field)) is not str or not completion[field]:
            raise FinalTestAggregationError(
                f"verified completion record {field} must be a non-empty string"
            )
    _utc_timestamp(completion.get("recorded_at_utc"), "completion.recorded_at_utc")
    _sha256(completion.get("run_identity"), "completion.run_identity")
    expected_authorization = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "method": arm["method"],
        "seed": run["seed"],
        "training_fingerprint": run["training_fingerprint"],
        "checkpoint": run["checkpoint"],
        "method_environment": arm["method_environment"],
        "hyperparameters": arm["hyperparameters"],
        "registered_output_dir": str(expected_output_dir),
        "registry_path": str(expected_registry),
        "registry_attempt": completion["attempt"],
    }
    for field, expected in expected_authorization.items():
        if authorization.get(field) != expected:
            raise FinalTestAggregationError(
                f"per-class report authorization {field} mismatch"
            )

    predictions = completion.get("artifacts", {}).get(runner_tool.PREDICTIONS_BASENAME)
    if type(predictions) is not dict or set(predictions) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise FinalTestAggregationError(
            "verified completed record lacks predictions identity"
        )
    predictions_path = _resolve_project_path(
        project_root, predictions["path"], "completed predictions path"
    )
    if predictions_path != expected_output_dir / runner_tool.PREDICTIONS_BASENAME:
        raise FinalTestAggregationError(
            "verified predictions path differs from fixed run output path"
        )
    _sha256(predictions["sha256"], "completed predictions SHA-256")
    _nonnegative_int(predictions["size_bytes"], "completed predictions size_bytes")
    if (
        predictions["size_bytes"] <= 0
        or predictions_path.stat().st_size != predictions["size_bytes"]
        or _sha256_file(predictions_path) != predictions["sha256"]
    ):
        raise FinalTestAggregationError(
            "predictions.pkl changed after runner completion verification"
        )
    provenance = _require_exact_object(
        report["provenance"],
        {
            "command",
            "python_executable",
            "project_root",
            "tool_path",
            "tool_sha256",
            "offline_evaluator_path",
            "offline_evaluator_sha256",
            "manifest_tool_path",
            "manifest_tool_sha256",
            "data_manifest_tool_path",
            "data_manifest_tool_sha256",
            "runner_tool_path",
            "runner_tool_sha256",
            "config_path",
            "config_sha256",
            "effective_config_sha256",
            "predictions_path",
            "predictions_sha256",
            "predictions_size_bytes",
            "output_path",
            "pickle_trust_acknowledged",
            "versions",
        },
        "per-class report.provenance",
    )
    config_path = _resolve_project_path(
        project_root, selection["config"]["path"], "selection.config.path"
    )
    expected_provenance = {
        "project_root": str(project_root),
        "tool_path": str(Path(evaluator_tool.__file__).resolve()),
        "tool_sha256": _sha256_file(Path(evaluator_tool.__file__).resolve()),
        "offline_evaluator_path": str(evaluator_tool.REQUIRED_RUNTIME_FILES[1]),
        "offline_evaluator_sha256": _sha256_file(
            evaluator_tool.REQUIRED_RUNTIME_FILES[1]
        ),
        "manifest_tool_path": str(evaluator_tool.REQUIRED_RUNTIME_FILES[2]),
        "manifest_tool_sha256": _sha256_file(evaluator_tool.REQUIRED_RUNTIME_FILES[2]),
        "data_manifest_tool_path": str(evaluator_tool.REQUIRED_RUNTIME_FILES[3]),
        "data_manifest_tool_sha256": _sha256_file(
            evaluator_tool.REQUIRED_RUNTIME_FILES[3]
        ),
        "runner_tool_path": str(evaluator_tool.REQUIRED_RUNTIME_FILES[4]),
        "runner_tool_sha256": _sha256_file(evaluator_tool.REQUIRED_RUNTIME_FILES[4]),
        "config_path": str(config_path),
        "config_sha256": selection["config"]["sha256"],
        "predictions_path": predictions["path"],
        "predictions_sha256": predictions["sha256"],
        "predictions_size_bytes": predictions["size_bytes"],
        "output_path": str(report_path),
        "pickle_trust_acknowledged": True,
    }
    for field, expected in expected_provenance.items():
        if provenance.get(field) != expected:
            raise FinalTestAggregationError(
                f"per-class report provenance {field} mismatch"
            )
    _sha256(
        provenance["effective_config_sha256"],
        "per-class report provenance effective_config_sha256",
    )
    if type(provenance["command"]) is not list or not all(
        type(item) is str for item in provenance["command"]
    ):
        raise FinalTestAggregationError("per-class report command must be string array")
    if type(provenance["python_executable"]) is not str:
        raise FinalTestAggregationError(
            "per-class report python_executable must be a string"
        )
    if type(provenance["versions"]) is not dict:
        raise FinalTestAggregationError("per-class report versions must be an object")
    if set(provenance["versions"]) != {"mmcv", "mmdet", "mmrotate", "numpy"} or not all(
        type(value) is str and value for value in provenance["versions"].values()
    ):
        raise FinalTestAggregationError(
            "per-class report versions must name the four evaluator libraries"
        )

    evaluator = _require_exact_object(
        report["evaluator"],
        {
            "function",
            "iou_threshold",
            "nproc",
            "model_built",
            "inference_run",
            "cga_environment",
        },
        "per-class report.evaluator",
    )
    if (
        evaluator["function"] != "mmrotate.core.eval_rbbox_map"
        or evaluator["iou_threshold"] != 0.5
        or type(evaluator["nproc"]) is not int
        or evaluator["nproc"] < 1
        or evaluator["model_built"] is not False
        or evaluator["inference_run"] is not False
        or evaluator["cga_environment"]
        != evaluator_tool.offline_evaluator.DISABLED_CGA_ENV
    ):
        raise FinalTestAggregationError("per-class evaluator contract mismatch")

    dataset = _require_exact_object(
        report["dataset"],
        {
            "type",
            "test_mode",
            "ann_file",
            "img_prefix",
            "classes",
            "data_manifest_path",
            "data_manifest_sha256",
        },
        "per-class report.dataset",
    )
    if (
        dataset["test_mode"] is not True
        or dataset["classes"] != list(RSAR_CLASSES)
        or dataset["data_manifest_path"] != str(data_manifest_path)
        or dataset["data_manifest_sha256"] != data_manifest_sha256
        or dataset["ann_file"] != data_manifest_payload["annotations"]["root"]
        or dataset["img_prefix"] != data_manifest_payload["images"]["root"]
    ):
        raise FinalTestAggregationError("per-class dataset identity mismatch")

    result = _require_exact_object(
        report["result"],
        {
            "metric",
            "iou_threshold",
            "use_07_metric",
            "mean_ap",
            "per_class",
            "num_images",
            "classes",
            "detections_per_class",
            "num_detections",
            "predictions_semantic_sha256",
            "annotations_semantic_sha256",
            "image_order_sha256",
            "first_image_id",
            "last_image_id",
        },
        "per-class report.result",
    )
    if (
        result["metric"] != "rotated_bbox_mAP"
        or result["iou_threshold"] != 0.5
        or result["use_07_metric"] is not True
        or result["classes"] != list(RSAR_CLASSES)
    ):
        raise FinalTestAggregationError("per-class result metric contract mismatch")
    mean_ap = _finite_metric(result["mean_ap"], "result.mean_ap")
    _nonnegative_int(result["num_images"], "result.num_images")
    if result["num_images"] != data_manifest_payload["alignment"]["stem_count"]:
        raise FinalTestAggregationError(
            "result.num_images differs from verified data manifest"
        )
    _nonnegative_int(result["num_detections"], "result.num_detections")
    detections = result["detections_per_class"]
    if type(detections) is not dict or set(detections) != set(RSAR_CLASSES):
        raise FinalTestAggregationError(
            "result.detections_per_class must contain exactly six classes"
        )
    for class_name in RSAR_CLASSES:
        _nonnegative_int(detections[class_name], f"detections_per_class.{class_name}")
    if sum(detections.values()) != result["num_detections"]:
        raise FinalTestAggregationError(
            "result.num_detections differs from per-class detection counts"
        )
    for field in (
        "predictions_semantic_sha256",
        "annotations_semantic_sha256",
        "image_order_sha256",
    ):
        _sha256(result[field], f"result.{field}")

    per_class = result["per_class"]
    if type(per_class) is not list or len(per_class) != len(RSAR_CLASSES):
        raise FinalTestAggregationError("result.per_class must have exactly six rows")
    class_aps: dict[str, float] = {}
    for index, (class_name, item) in enumerate(zip(RSAR_CLASSES, per_class)):
        item = _require_exact_object(
            item,
            {"index", "class", "ap", "num_gts", "num_dets", "final_recall"},
            f"result.per_class[{index}]",
        )
        if item["index"] != index or item["class"] != class_name:
            raise FinalTestAggregationError(
                "result.per_class has duplicate, missing, or reordered classes"
            )
        class_aps[class_name] = _finite_metric(
            item["ap"], f"result.per_class[{index}].ap"
        )
        _nonnegative_int(item["num_gts"], f"result.per_class[{index}].num_gts")
        _nonnegative_int(item["num_dets"], f"result.per_class[{index}].num_dets")
        if item["num_dets"] != detections[class_name]:
            raise FinalTestAggregationError(
                f"result.per_class[{index}].num_dets differs from class count"
            )
        _finite_metric(item["final_recall"], f"result.per_class[{index}].final_recall")

    checkpoint = {
        "path": str(expected_checkpoint_path),
        "declared_path": run["checkpoint"]["path"],
        "sha256": run["checkpoint"]["sha256"],
        "size_bytes": run["checkpoint"]["size_bytes"],
        "meta": run["checkpoint"]["meta"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "method": arm["method"],
        "seed": run["seed"],
        "training_fingerprint": run["training_fingerprint"],
        "mean_ap": mean_ap,
        "per_class_ap": class_aps,
        "checkpoint": checkpoint,
        "gpu": {
            "index": completion["gpu_index"],
            "uuid": completion["gpu_uuid"],
            "name": completion["gpu_name"],
            "verified": completion["gpu_verified"],
        },
        "provenance": {
            "registry_path": str(expected_registry),
            "registry_attempt": completion["attempt"],
            "registry_recorded_at_utc": completion["recorded_at_utc"],
            "run_identity": completion["run_identity"],
            "output_dir": str(expected_output_dir),
            "predictions": dict(predictions),
            "per_class_report_path": str(report_path),
            "per_class_report_sha256": report_sha256,
            "data_manifest_path": str(data_manifest_path),
            "data_manifest_sha256": data_manifest_sha256,
            "evaluator_tool_sha256": provenance["tool_sha256"],
            "config_sha256": provenance["config_sha256"],
            "effective_config_sha256": provenance["effective_config_sha256"],
        },
    }


def _collect_rows(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    payload: Mapping[str, Any],
    data_manifest_path: Path,
    data_manifest_sha256: str,
    data_manifest_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    project_root = _resolve_project_path(
        Path("/"), payload["project_root"], "manifest.project_root"
    )
    required_seeds = set(payload["aggregation"]["required_seed_set"])
    expected_identities = {
        (arm["method"], seed) for arm in payload["arms"] for seed in required_seeds
    }
    seen: set[tuple[str, int]] = set()
    rows: list[dict[str, Any]] = []
    for arm in payload["arms"]:
        method = arm["method"]
        arm_seeds = {run["seed"] for run in arm["runs"]}
        if arm_seeds != required_seeds or len(arm["runs"]) != len(required_seeds):
            raise FinalTestAggregationError(
                f"arm {method!r} does not exactly cover required_seed_set"
            )
        for run in sorted(arm["runs"], key=lambda item: item["seed"]):
            identity = (method, run["seed"])
            if identity in seen:
                raise FinalTestAggregationError(
                    f"duplicate final-test run identity: {identity}"
                )
            seen.add(identity)
            try:
                completion = runner_tool.verify_completed_run(
                    manifest_path, method, run["seed"]
                )
            except runner_tool.FinalTestBundleError as error:
                raise FinalTestAggregationError(
                    f"runner completion verification failed for {identity}: {error}"
                ) from error
            output_dir = _resolve_project_path(
                project_root, run["output_dir"], f"run {identity}.output_dir"
            )
            report_path = output_dir / runner_tool.POSTPROCESS_BASENAME
            report, report_sha256 = _load_strict_evaluator_report(report_path)
            rows.append(
                _validate_per_class_report(
                    report=report,
                    report_path=report_path,
                    report_sha256=report_sha256,
                    manifest_path=manifest_path,
                    manifest_sha256=manifest_sha256,
                    data_manifest_path=data_manifest_path,
                    data_manifest_sha256=data_manifest_sha256,
                    data_manifest_payload=data_manifest_payload,
                    project_root=project_root,
                    selection=payload["selection"],
                    arm=arm,
                    run=run,
                    completion=completion,
                )
            )
    if seen != expected_identities or len(rows) != len(expected_identities):
        missing = sorted(expected_identities - seen)
        extra = sorted(seen - expected_identities)
        raise FinalTestAggregationError(
            f"final-test run coverage mismatch: missing={missing}, extra={extra}"
        )
    return rows


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def _describe(values_by_seed: Mapping[int, float]) -> dict[str, Any]:
    ordered = sorted(values_by_seed.items())
    values = [value for _, value in ordered]
    return {
        "N": len(values),
        "seeds": [seed for seed, _ in ordered],
        "raw_values": [value for _, value in ordered],
        "mean": _mean(values),
        "sample_sd": _sample_sd(values),
    }


def _paired_summary(
    *,
    baseline_method: str,
    candidate_method: str,
    baseline_rows: Mapping[int, Mapping[str, Any]],
    candidate_rows: Mapping[int, Mapping[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if set(baseline_rows) != set(candidate_rows):
        raise FinalTestAggregationError(
            f"seed alignment mismatch for {candidate_method} vs {baseline_method}"
        )
    pairs: list[dict[str, Any]] = []
    differences: list[float] = []
    for seed in sorted(baseline_rows):
        baseline = float(baseline_rows[seed]["mean_ap"])
        candidate = float(candidate_rows[seed]["mean_ap"])
        difference = candidate - baseline
        differences.append(difference)
        pairs.append(
            {
                "seed": seed,
                "baseline": baseline,
                "candidate": candidate,
                "difference": difference,
                "baseline_gpu_uuid": baseline_rows[seed]["gpu"]["uuid"],
                "candidate_gpu_uuid": candidate_rows[seed]["gpu"]["uuid"],
            }
        )
    sample_sd = _sample_sd(differences)
    summary: dict[str, Any] = {
        "baseline_method": baseline_method,
        "candidate_method": candidate_method,
        "N": len(differences),
        "pairs": pairs,
        "differences": differences,
        "mean_difference": _mean(differences),
        "sample_sd_difference": sample_sd,
        "standard_error": (
            sample_sd / math.sqrt(len(differences)) if sample_sd is not None else None
        ),
        "bootstrap_ci_95": None,
        "paired_t": None,
        "paired_t_na_reason": None,
        "exact_sign_flip": None,
        "cohens_dz": None,
        "cohens_dz_na_reason": None,
        "paired_t_p_holm": None,
        "exact_sign_flip_p_holm": None,
    }
    if len(differences) >= 2:
        summary["bootstrap_ci_95"] = list(
            statistics_tool.bootstrap_mean_ci(
                differences,
                resamples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )
        try:
            summary["paired_t"] = statistics_tool.paired_t_test(differences)
        except statistics_tool.StatisticsError as error:
            summary["paired_t_na_reason"] = str(error)
        try:
            summary["cohens_dz"] = statistics_tool.cohens_dz(differences)
        except statistics_tool.StatisticsError as error:
            summary["cohens_dz_na_reason"] = str(error)
    else:
        summary["paired_t_na_reason"] = "paired t-test requires at least two seeds"
        summary["cohens_dz_na_reason"] = "Cohen's dz requires at least two seeds"
    try:
        summary["exact_sign_flip"] = statistics_tool.exact_sign_permutation_test(
            differences
        )
    except statistics_tool.StatisticsError as error:
        raise FinalTestAggregationError(
            f"exact sign-flip test failed for {candidate_method}: {error}"
        ) from error
    return summary


def _statistical_report(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    payload: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_seeds = list(payload["aggregation"]["required_seed_set"])
    baseline = payload["aggregation"]["paired_comparator"]
    by_method: dict[str, dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if row["method"] in by_method and row["seed"] in by_method[row["method"]]:
            raise FinalTestAggregationError(
                "duplicate statistical row for " f"{row['method']!r}/seed_{row['seed']}"
            )
        by_method.setdefault(row["method"], {})[row["seed"]] = row
    method_order = [arm["method"] for arm in payload["arms"]]
    for method in method_order:
        if set(by_method.get(method, {})) != set(required_seeds):
            raise FinalTestAggregationError(
                f"statistical input for {method!r} is not seed-complete"
            )
    method_descriptives: list[dict[str, Any]] = []
    for method in method_order:
        method_rows = by_method[method]
        method_descriptives.append(
            {
                "method": method,
                "mean_ap": _describe(
                    {seed: float(row["mean_ap"]) for seed, row in method_rows.items()}
                ),
                "per_class_ap": {
                    class_name: _describe(
                        {
                            seed: float(row["per_class_ap"][class_name])
                            for seed, row in method_rows.items()
                        }
                    )
                    for class_name in RSAR_CLASSES
                },
            }
        )
    comparisons = [
        _paired_summary(
            baseline_method=baseline,
            candidate_method=method,
            baseline_rows=by_method[baseline],
            candidate_rows=by_method[method],
            bootstrap_samples=payload["aggregation"]["bootstrap_samples"],
            bootstrap_seed=payload["aggregation"]["bootstrap_seed"],
        )
        for method in method_order
        if method != baseline
    ]
    t_p_values = {
        item["candidate_method"]: item["paired_t"]["p"]
        for item in comparisons
        if item["paired_t"] is not None
    }
    sign_p_values = {
        item["candidate_method"]: item["exact_sign_flip"]["p"] for item in comparisons
    }
    adjusted_t = statistics_tool.holm_adjust(t_p_values) if t_p_values else {}
    adjusted_sign = statistics_tool.holm_adjust(sign_p_values)
    for item in comparisons:
        candidate = item["candidate_method"]
        item["paired_t_p_holm"] = adjusted_t.get(candidate)
        item["exact_sign_flip_p_holm"] = adjusted_sign[candidate]
    return {
        "schema_version": SCHEMA_VERSION,
        "report_kind": "preregistered_final_test_descriptive_statistics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "policy": {
            "descriptive_only": True,
            "method_selection_performed": False,
            "ranking_performed": False,
            "difference_direction": "candidate mAP - paired comparator mAP",
            "pairing": "same preregistered seed",
            "test_usage": "single frozen final-test evaluation; no tuning",
        },
        "class_order": list(RSAR_CLASSES),
        "required_seed_set": required_seeds,
        "paired_comparator": baseline,
        "settings": {
            "sample_sd_ddof": 1,
            "bootstrap_samples": payload["aggregation"]["bootstrap_samples"],
            "bootstrap_seed": payload["aggregation"]["bootstrap_seed"],
            "bootstrap_confidence": 0.95,
            "sign_flip": "exact enumeration",
            "holm_family": "all non-comparator arms, separately by test",
        },
        "method_descriptives": method_descriptives,
        "paired_comparisons": comparisons,
    }


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    if type(value) is int:
        return str(value)
    return f"{float(value):.10g}"


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Final-test preregistered statistical report",
        "",
        "> Descriptive reporting only. No method selection, ranking, tuning, or "
        "checkpoint choice was performed from final-test results.",
        "",
        f"- Manifest: `{report['manifest_path']}`",
        f"- Manifest SHA-256: `{report['manifest_sha256']}`",
        f"- Paired comparator: `{report['paired_comparator']}`",
        f"- Required seeds: `{report['required_seed_set']}`",
        "- Pairing: same preregistered seed",
        "- Difference: candidate mAP − comparator mAP",
        "",
        "## Arm descriptions",
        "",
        "| method | N | mean mAP | sample SD |",
        "|---|---:|---:|---:|",
    ]
    for item in report["method_descriptives"]:
        description = item["mean_ap"]
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                item["method"],
                description["N"],
                _format_number(description["mean"]),
                _format_number(description["sample_sd"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-class arm descriptions",
            "",
            "| method | class | mean AP | sample SD |",
            "|---|---|---:|---:|",
        ]
    )
    for item in report["method_descriptives"]:
        for class_name in RSAR_CLASSES:
            description = item["per_class_ap"][class_name]
            lines.append(
                "| `{}` | {} | {} | {} |".format(
                    item["method"],
                    class_name,
                    _format_number(description["mean"]),
                    _format_number(description["sample_sd"]),
                )
            )
    for comparison in report["paired_comparisons"]:
        lines.extend(
            [
                "",
                f"## `{comparison['candidate_method']}` vs "
                f"`{comparison['baseline_method']}`",
                "",
                "| seed | comparator | candidate | difference | comparator GPU | candidate GPU |",
                "|---:|---:|---:|---:|---|---|",
            ]
        )
        for pair in comparison["pairs"]:
            lines.append(
                "| {seed} | {baseline} | {candidate} | {difference} | `{bgpu}` | "
                "`{cgpu}` |".format(
                    seed=pair["seed"],
                    baseline=_format_number(pair["baseline"]),
                    candidate=_format_number(pair["candidate"]),
                    difference=_format_number(pair["difference"]),
                    bgpu=pair["baseline_gpu_uuid"],
                    cgpu=pair["candidate_gpu_uuid"],
                )
            )
        paired_t = comparison["paired_t"] or {}
        sign_flip = comparison["exact_sign_flip"]
        interval = comparison["bootstrap_ci_95"] or [None, None]
        lines.extend(
            [
                "",
                "| statistic | value |",
                "|---|---:|",
                f"| N | {comparison['N']} |",
                f"| Mean difference | {_format_number(comparison['mean_difference'])} |",
                f"| Sample SD | {_format_number(comparison['sample_sd_difference'])} |",
                f"| Standard error | {_format_number(comparison['standard_error'])} |",
                "| Bootstrap 95% CI | [{}, {}] |".format(
                    _format_number(interval[0]), _format_number(interval[1])
                ),
                f"| Paired t | {_format_number(paired_t.get('t'))} |",
                f"| df | {_format_number(paired_t.get('df'))} |",
                f"| Paired t p | {_format_number(paired_t.get('p'))} |",
                f"| Paired t Holm p | {_format_number(comparison['paired_t_p_holm'])} |",
                f"| Exact sign-flip p | {_format_number(sign_flip['p'])} |",
                f"| Exact sign-flip assignments | {sign_flip['assignments']} |",
                f"| Exact sign-flip Holm p | {_format_number(comparison['exact_sign_flip_p_holm'])} |",
                f"| Cohen's dz | {_format_number(comparison['cohens_dz'])} |",
            ]
        )
        if comparison["paired_t_na_reason"]:
            lines.extend(["", f"> Paired t: {comparison['paired_t_na_reason']}"])
        if comparison["cohens_dz_na_reason"]:
            lines.extend(["", f"> Cohen's dz: {comparison['cohens_dz_na_reason']}"])
    return "\n".join(lines).rstrip() + "\n"


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fieldnames = [
        "manifest_sha256",
        "method",
        "seed",
        "training_fingerprint",
        "mean_ap",
        *(f"ap_{class_name}" for class_name in RSAR_CLASSES),
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "gpu_index",
        "gpu_uuid",
        "gpu_name",
        "predictions_path",
        "predictions_sha256",
        "predictions_size_bytes",
        "registry_path",
        "registry_attempt",
        "per_class_report_path",
        "per_class_report_sha256",
        "data_manifest_path",
        "data_manifest_sha256",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        provenance = row["provenance"]
        predictions = provenance["predictions"]
        checkpoint = row["checkpoint"]
        flat = {
            "manifest_sha256": row["manifest_sha256"],
            "method": row["method"],
            "seed": row["seed"],
            "training_fingerprint": row["training_fingerprint"],
            "mean_ap": row["mean_ap"],
            **{
                f"ap_{class_name}": row["per_class_ap"][class_name]
                for class_name in RSAR_CLASSES
            },
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "gpu_index": row["gpu"]["index"],
            "gpu_uuid": row["gpu"]["uuid"],
            "gpu_name": row["gpu"]["name"],
            "predictions_path": predictions["path"],
            "predictions_sha256": predictions["sha256"],
            "predictions_size_bytes": predictions["size_bytes"],
            "registry_path": provenance["registry_path"],
            "registry_attempt": provenance["registry_attempt"],
            "per_class_report_path": provenance["per_class_report_path"],
            "per_class_report_sha256": provenance["per_class_report_sha256"],
            "data_manifest_path": provenance["data_manifest_path"],
            "data_manifest_sha256": provenance["data_manifest_sha256"],
        }
        writer.writerow(flat)
    return stream.getvalue().encode("utf-8")


def _atomic_publish(outputs: Mapping[Path, bytes]) -> None:
    for path in outputs:
        if path.is_symlink() or path.exists():
            raise FinalTestAggregationError(
                f"refusing to replace existing aggregate output: {path}"
            )
    temporaries: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        for target, content in outputs.items():
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(target.parent),
                prefix=f".{target.name}.",
                delete=False,
            ) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                temporaries[target] = Path(handle.name)
        for target, temporary in temporaries.items():
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FinalTestAggregationError(
                    f"aggregate output appeared concurrently: {target}"
                ) from error
            published.append(target)
        directory_fd = os.open(next(iter(outputs)).parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        for path in published:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in temporaries.values():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def aggregate(manifest_path: Path) -> dict[str, Any]:
    manifest_path = _lexical_absolute(manifest_path)
    (
        manifest_sha256,
        payload,
        data_path,
        data_sha256,
        data_payload,
    ) = _verified_frozen_inputs(manifest_path)
    rows = _collect_rows(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        payload=payload,
        data_manifest_path=data_path,
        data_manifest_sha256=data_sha256,
        data_manifest_payload=data_payload,
    )
    report = _statistical_report(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        payload=payload,
        rows=rows,
    )
    output_root = manifest_path.parent
    outputs = {
        output_root / "final_test_results.csv": _csv_bytes(rows),
        output_root / "final_test_results.jsonl": _jsonl_bytes(rows),
        output_root
        / "final_test_statistical_report.md": _render_markdown(report).encode("utf-8"),
        output_root / "final_test_statistical_report.json": _json_bytes(report),
    }
    if set(path.name for path in outputs) != set(OUTPUT_BASENAMES):
        raise FinalTestAggregationError("internal fixed-output contract mismatch")
    _atomic_publish(outputs)
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "rows": len(rows),
        "methods": [arm["method"] for arm in payload["arms"]],
        "required_seed_set": payload["aggregation"]["required_seed_set"],
        "outputs": {path.name: str(path) for path in outputs},
        "descriptive_only": True,
        "method_selection_performed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = aggregate(args.manifest)
    except (
        FinalTestAggregationError,
        manifest_tool.ManifestError,
        data_manifest_tool.DataManifestError,
        runner_tool.FinalTestBundleError,
        statistics_tool.StatisticsError,
        OSError,
    ) as error:
        print(f"final-test aggregation error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
