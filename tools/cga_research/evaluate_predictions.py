#!/usr/bin/env python3
"""Evaluate trusted offline rotated detections without running a model.

The input pickle must have the standard MMRotate shape::

    [image][class] -> ndarray[num_detections, 6]

where each row is ``(cx, cy, width, height, angle, score)``.  Pickle is an
arbitrary-code execution format, so loading requires the explicit
``--trust-pickle`` acknowledgement.  This tool never builds a detector and
forces CGA-related environment variables off before importing project code.
It also rejects paths containing a ``test`` component: this research helper
is intentionally restricted to train/validation protocol checks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RSAR_CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
DETECTION_COLUMNS = 6
DISABLED_CGA_ENV = {
    "CGA_SCORER": "none",
    "CGA_BACKEND": "none",
    "CGA_FILTER_MODE": "none",
}
ANNOTATION_KEYS = (
    "bboxes",
    "labels",
    "bboxes_ignore",
    "labels_ignore",
)


class OfflineEvaluationError(RuntimeError):
    """Raised when an offline-evaluation safety or data invariant fails."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def force_disable_cga(environment: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Force CGA off and return the effective values for provenance."""

    target = os.environ if environment is None else environment
    target.update(DISABLED_CGA_ENV)
    target["PYTHONNOUSERSITE"] = "1"
    return {key: target[key] for key in sorted(DISABLED_CGA_ENV)}


def _contains_test_component(path: Path) -> bool:
    return any(part.casefold() == "test" for part in path.parts)


def reject_test_path(path: Path, role: str) -> Path:
    """Resolve *path* and reject any explicit ``test`` path component."""

    resolved = path.expanduser().resolve()
    if _contains_test_component(resolved):
        raise OfflineEvaluationError(
            f"refusing {role} path containing a 'test' component: {resolved}"
        )
    return resolved


def _resolve_dataset_path(value: Any, data_root: Any, role: str) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise OfflineEvaluationError(
            f"data.test.{role} must be one non-empty filesystem path"
        )
    path = Path(value).expanduser()
    if not path.is_absolute() and data_root:
        path = Path(str(data_root)).expanduser() / path
    return reject_test_path(path, f"data.test.{role}")


def validate_dataset_paths(test_cfg: Mapping[str, Any]) -> tuple[Path, Path]:
    """Validate the effective DOTA annotation and image directories."""

    data_root = test_cfg.get("data_root")
    ann_file = _resolve_dataset_path(test_cfg.get("ann_file"), data_root, "ann_file")
    img_prefix = _resolve_dataset_path(
        test_cfg.get("img_prefix"), data_root, "img_prefix"
    )
    if not ann_file.is_dir():
        raise OfflineEvaluationError(f"annotation directory does not exist: {ann_file}")
    if not img_prefix.is_dir():
        raise OfflineEvaluationError(f"image directory does not exist: {img_prefix}")
    return ann_file, img_prefix


def load_trusted_pickle(path: Path, trust_pickle: bool) -> Any:
    """Load a pickle only after an explicit trust acknowledgement."""

    if not trust_pickle:
        raise OfflineEvaluationError(
            "refusing to load pickle without --trust-pickle; pickle can execute "
            "arbitrary code and must come from a trusted local inference run"
        )
    if path.suffix.casefold() not in {".pkl", ".pickle"}:
        raise OfflineEvaluationError("predictions must use a .pkl or .pickle suffix")
    if not path.is_file():
        raise OfflineEvaluationError(f"predictions file does not exist: {path}")
    print(
        "[security] --trust-pickle acknowledged; loading trusted local "
        f"predictions: {path}",
        file=sys.stderr,
        flush=True,
    )
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - guarded by explicit acknowledgement


def _update_array_digest(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


def validate_predictions(
    predictions: Any,
    *,
    num_images: int,
    classes: Sequence[str],
) -> tuple[list[list[np.ndarray]], list[int], str]:
    """Validate and normalize the MMRotate per-image/per-class result shape."""

    actual_classes = tuple(str(item) for item in classes)
    if actual_classes != RSAR_CLASSES:
        raise OfflineEvaluationError(
            "dataset class order mismatch: expected "
            f"{RSAR_CLASSES!r}, got {actual_classes!r}"
        )
    if not isinstance(predictions, (list, tuple)):
        raise OfflineEvaluationError("predictions root must be a list or tuple")
    if len(predictions) != num_images:
        raise OfflineEvaluationError(
            f"prediction/image count mismatch: {len(predictions)} != {num_images}"
        )

    normalized: list[list[np.ndarray]] = []
    class_counts = [0 for _ in RSAR_CLASSES]
    digest = hashlib.sha256()
    digest.update(json.dumps(list(RSAR_CLASSES)).encode("utf-8"))
    digest.update(str(num_images).encode("ascii"))

    for image_index, image_result in enumerate(predictions):
        if not isinstance(image_result, (list, tuple)):
            raise OfflineEvaluationError(
                f"prediction[{image_index}] must be a six-class list or tuple"
            )
        if len(image_result) != len(RSAR_CLASSES):
            raise OfflineEvaluationError(
                f"prediction[{image_index}] has {len(image_result)} classes; "
                f"expected {len(RSAR_CLASSES)} in order {RSAR_CLASSES!r}"
            )

        normalized_image: list[np.ndarray] = []
        for class_index, class_result in enumerate(image_result):
            if not isinstance(class_result, np.ndarray):
                raise OfflineEvaluationError(
                    f"prediction[{image_index}][{class_index}] must be a numpy array"
                )
            if class_result.ndim != 2 or class_result.shape[1] != DETECTION_COLUMNS:
                raise OfflineEvaluationError(
                    f"prediction[{image_index}][{class_index}] has shape "
                    f"{class_result.shape}; expected (N, {DETECTION_COLUMNS})"
                )
            if not np.issubdtype(class_result.dtype, np.number):
                raise OfflineEvaluationError(
                    f"prediction[{image_index}][{class_index}] must be numeric"
                )
            array = np.asarray(class_result, dtype=np.float32)
            if not np.isfinite(array).all():
                raise OfflineEvaluationError(
                    f"prediction[{image_index}][{class_index}] contains NaN or Inf"
                )
            if array.shape[0] and np.any(array[:, 2:4] <= 0):
                raise OfflineEvaluationError(
                    f"prediction[{image_index}][{class_index}] contains "
                    "non-positive width or height"
                )
            if array.shape[0] and (
                np.any(array[:, 5] < 0) or np.any(array[:, 5] > 1)
            ):
                raise OfflineEvaluationError(
                    f"prediction[{image_index}][{class_index}] has scores "
                    "outside [0, 1]"
                )
            array = np.ascontiguousarray(array)
            normalized_image.append(array)
            class_counts[class_index] += int(array.shape[0])
            digest.update(image_index.to_bytes(8, "little", signed=False))
            digest.update(class_index.to_bytes(4, "little", signed=False))
            _update_array_digest(digest, array)
        normalized.append(normalized_image)

    return normalized, class_counts, digest.hexdigest()


def collect_annotations(dataset: Any) -> tuple[list[Mapping[str, Any]], str]:
    """Collect annotations in dataset order and hash their evaluation semantics."""

    annotations: list[Mapping[str, Any]] = []
    digest = hashlib.sha256()
    for image_index in range(len(dataset)):
        annotation = dataset.get_ann_info(image_index)
        if not isinstance(annotation, Mapping):
            raise OfflineEvaluationError(
                f"annotation[{image_index}] must be a mapping"
            )
        annotations.append(annotation)
        digest.update(image_index.to_bytes(8, "little", signed=False))
        for key in ANNOTATION_KEYS:
            digest.update(key.encode("ascii"))
            value = annotation.get(key)
            if value is None:
                digest.update(b"<none>")
                continue
            array = np.asarray(value)
            if key.endswith("labels") or key == "labels_ignore":
                array = np.asarray(array, dtype=np.int64)
            else:
                array = np.asarray(array, dtype=np.float32)
            _update_array_digest(digest, array)
    return annotations, digest.hexdigest()


def dataset_image_order(dataset: Any) -> tuple[list[str], str]:
    """Return image identifiers in evaluation order and their SHA-256 digest."""

    identifiers: list[str] = []
    data_infos = getattr(dataset, "data_infos", None)
    image_ids = getattr(dataset, "img_ids", None)
    for index in range(len(dataset)):
        identifier: Any = None
        if isinstance(data_infos, Sequence) and index < len(data_infos):
            info = data_infos[index]
            if isinstance(info, Mapping):
                identifier = info.get("filename")
        if (
            identifier is None
            and isinstance(image_ids, Sequence)
            and index < len(image_ids)
        ):
            identifier = image_ids[index]
        if identifier is None:
            identifier = str(index)
        identifiers.append(str(identifier))
    encoded = json.dumps(identifiers, ensure_ascii=False, separators=(",", ":"))
    return identifiers, sha256_bytes(encoded.encode("utf-8"))


def summarize_class_metrics(
    eval_results: Sequence[Mapping[str, Any]], classes: Sequence[str]
) -> list[dict[str, Any]]:
    if len(eval_results) != len(classes):
        raise OfflineEvaluationError(
            f"eval_rbbox_map returned {len(eval_results)} class results; "
            f"expected {len(classes)}"
        )
    summaries: list[dict[str, Any]] = []
    for index, (class_name, result) in enumerate(zip(classes, eval_results)):
        ap = float(np.asarray(result["ap"]).reshape(-1)[0])
        num_gts = int(np.asarray(result["num_gts"]).sum())
        num_dets = int(result["num_dets"])
        recall_array = np.asarray(result.get("recall", []), dtype=np.float64)
        final_recall = float(recall_array.reshape(-1)[-1]) if recall_array.size else 0.0
        if not all(math.isfinite(item) for item in (ap, final_recall)):
            raise OfflineEvaluationError(
                f"non-finite metric returned for class {class_name}"
            )
        summaries.append(
            {
                "index": index,
                "class": str(class_name),
                "ap": ap,
                "num_gts": num_gts,
                "num_dets": num_dets,
                "final_recall": final_recall,
            }
        )
    return summaries


def evaluate_validated_predictions(
    *,
    dataset: Any,
    predictions: Any,
    eval_rbbox_map_fn: Callable[..., tuple[Any, Sequence[Mapping[str, Any]]]],
    iou_thr: float,
    nproc: int,
) -> dict[str, Any]:
    """Validate inputs, call the official evaluator, and summarize metrics."""

    classes = tuple(str(item) for item in getattr(dataset, "CLASSES", ()))
    normalized, class_counts, semantic_hash = validate_predictions(
        predictions, num_images=len(dataset), classes=classes
    )
    annotations, annotations_hash = collect_annotations(dataset)
    image_identifiers, image_order_hash = dataset_image_order(dataset)

    mean_ap, class_results = eval_rbbox_map_fn(
        normalized,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        use_07_metric=True,
        dataset=classes,
        logger="silent",
        nproc=nproc,
    )
    mean_ap = float(mean_ap)
    if not math.isfinite(mean_ap):
        raise OfflineEvaluationError("eval_rbbox_map returned a non-finite mAP")
    per_class = summarize_class_metrics(class_results, classes)
    return {
        "metric": "rotated_bbox_mAP",
        "iou_threshold": float(iou_thr),
        "use_07_metric": True,
        "mean_ap": mean_ap,
        "per_class": per_class,
        "num_images": len(dataset),
        "classes": list(classes),
        "detections_per_class": {
            class_name: class_counts[index]
            for index, class_name in enumerate(classes)
        },
        "num_detections": int(sum(class_counts)),
        "predictions_semantic_sha256": semantic_hash,
        "annotations_semantic_sha256": annotations_hash,
        "image_order_sha256": image_order_hash,
        "first_image_id": image_identifiers[0] if image_identifiers else None,
        "last_image_id": image_identifiers[-1] if image_identifiers else None,
    }


def _git_head(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_parser(dict_action: type) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trusted MMRotate predictions pickle offline on an "
            "explicit non-test RSAR split; no detector or CGA inference is run."
        )
    )
    parser.add_argument("config", type=Path, help="trusted MMRotate config")
    parser.add_argument("predictions", type=Path, help="trusted .pkl predictions")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help=(
            "acknowledge that pickle can execute arbitrary code and that this "
            "file was produced by a trusted local inference run"
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=dict_action,
        default={},
        help="MMCV key=value config overrides, for example corrupt=chaff",
    )
    parser.add_argument(
        "--ann-file",
        type=Path,
        help="override data.test.ann_file (must be paired with --img-prefix)",
    )
    parser.add_argument(
        "--img-prefix",
        type=Path,
        help="override data.test.img_prefix (must be paired with --ann-file)",
    )
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output JSON"
    )
    return parser


def _validate_cli_args(args: argparse.Namespace) -> None:
    if (args.ann_file is None) != (args.img_prefix is None):
        raise OfflineEvaluationError(
            "--ann-file and --img-prefix must be supplied together"
        )
    if not 0 < args.iou_thr <= 1:
        raise OfflineEvaluationError("--iou-thr must be in (0, 1]")
    if args.nproc < 1:
        raise OfflineEvaluationError("--nproc must be at least 1")
    config = args.config.expanduser().resolve()
    predictions = args.predictions.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not config.is_file():
        raise OfflineEvaluationError(f"config does not exist: {config}")
    if not predictions.is_file():
        raise OfflineEvaluationError(f"predictions do not exist: {predictions}")
    if output.suffix.casefold() != ".json":
        raise OfflineEvaluationError("--output must end with .json")
    if output in {config, predictions}:
        raise OfflineEvaluationError("output path must differ from both input paths")
    if output.exists() and not args.overwrite:
        raise OfflineEvaluationError(
            f"output already exists (use --overwrite explicitly): {output}"
        )
    if not args.trust_pickle:
        raise OfflineEvaluationError(
            "--trust-pickle is required before any pickle bytes are loaded"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build the configured test-mode dataset and perform offline evaluation."""

    cga_environment = force_disable_cga()
    _validate_cli_args(args)
    config_path = args.config.expanduser().resolve()
    predictions_path = args.predictions.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    # Heavy project imports remain here so pure validation helpers and unit
    # tests do not initialize CUDA/MMCV.  No model-building API is imported.
    import mmcv
    import mmdet
    import mmrotate
    from mmcv import Config
    from mmcv.utils import import_modules_from_strings
    from mmrotate.core import eval_rbbox_map
    from mmrotate.datasets import build_dataset
    from sfod.utils import patch_config

    cfg = Config.fromfile(str(config_path))
    cfg_options = dict(args.cfg_options or {})
    if cfg_options:
        cfg.merge_from_dict(cfg_options)
    if "data" not in cfg or "test" not in cfg.data:
        raise OfflineEvaluationError("config must define data.test")
    if not isinstance(cfg.data.test, Mapping):
        raise OfflineEvaluationError("data.test must be one dataset mapping")
    if args.ann_file is not None:
        cfg.data.test.ann_file = str(args.ann_file.expanduser().resolve())
        cfg.data.test.img_prefix = str(args.img_prefix.expanduser().resolve())

    cfg = patch_config(cfg)
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg.custom_imports)
    test_cfg = copy.deepcopy(cfg.data.test)
    test_cfg["test_mode"] = True
    ann_file, img_prefix = validate_dataset_paths(test_cfg)

    predictions_sha256 = sha256_file(predictions_path)
    predictions = load_trusted_pickle(predictions_path, args.trust_pickle)
    dataset = build_dataset(test_cfg)
    result = evaluate_validated_predictions(
        dataset=dataset,
        predictions=predictions,
        eval_rbbox_map_fn=eval_rbbox_map,
        iou_thr=args.iou_thr,
        nproc=args.nproc,
    )

    project_root = PROJECT_ROOT
    effective_config_text = cfg.pretty_text
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "evaluator": {
            "function": "mmrotate.core.eval_rbbox_map",
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
        },
        "provenance": {
            "command": list(sys.argv),
            "python_executable": sys.executable,
            "project_root": str(project_root),
            "git_head": _git_head(project_root),
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "effective_config_sha256": sha256_bytes(
                effective_config_text.encode("utf-8")
            ),
            "cfg_options": _jsonable(cfg_options),
            "predictions_path": str(predictions_path),
            "predictions_sha256": predictions_sha256,
            "predictions_size_bytes": predictions_path.stat().st_size,
            "output_path": str(output_path),
            "versions": {
                "mmcv": mmcv.__version__,
                "mmdet": mmdet.__version__,
                "mmrotate": mmrotate.__version__,
                "numpy": np.__version__,
            },
            "pickle_trust_acknowledged": True,
        },
    }
    _atomic_write_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "num_images": result["num_images"],
                "mAP": result["mean_ap"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    force_disable_cga()
    # Match the project's normal launch behavior so the Conda libstdc++ is
    # selected before MMCV/OpenCV imports.  This may re-exec once.
    from iraod_runtime import ensure_iraod_runtime

    ensure_iraod_runtime()
    from mmcv import DictAction

    args = build_parser(DictAction).parse_args(argv)
    try:
        run(args)
    except OfflineEvaluationError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
