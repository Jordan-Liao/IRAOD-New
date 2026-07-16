#!/usr/bin/env python3
"""Audit raw teacher predictions with rotated-IoU and independent SARCLIP scores.

The detector is always run without CGA.  SARCLIP is then called explicitly on
the retained raw detections, in bounded chunks, so the audit cannot accidentally
rescore a prediction twice.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fix the conda runtime before importing numpy/mmcv when this file is executed.
# Imports from unit tests intentionally skip the exec and exercise pure helpers.
if __name__ == "__main__":
    from iraod_runtime import ensure_iraod_runtime

    ensure_iraod_runtime()

import numpy as np

_CGA_SPEC = importlib.util.spec_from_file_location(
    "iraod_box_audit_cga", REPO_ROOT / "sfod" / "cga.py")
if _CGA_SPEC is None or _CGA_SPEC.loader is None:
    raise ImportError("unable to load sfod/cga.py for shuffled control")
_CGA_MODULE = importlib.util.module_from_spec(_CGA_SPEC)
_CGA_SPEC.loader.exec_module(_CGA_MODULE)
CGA_SHUFFLE_SCORE_BINS = _CGA_MODULE.CGA_SHUFFLE_SCORE_BINS
stratified_shuffle_probability_vectors = (
    _CGA_MODULE.stratified_shuffle_probability_vectors)


RSAR_CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
CLASS_TO_ID = {name: index for index, name in enumerate(RSAR_CLASSES)}
DEFAULT_CONFUSION_GROUPS = (("ship", "aircraft", "car", "tank"),)
CONTEXT_CLASSES = ("bridge", "harbor")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
TRAIN_POOL_SEED = 20260714
TRAIN_POOL_SIZE = 8467
MATCH_CATEGORIES = ("TP", "wrong-class", "localization-error", "pure-FP")

DEFAULT_CONFIG = "configs/baseline/oriented_rcnn_orthonet_rsar.py"
DEFAULT_CHECKPOINT = "work_dirs/oriented_rcnn_orthonet_rsar/epoch_100.pth"

PROBABILITY_FIELDS = tuple(f"p_clip_{name}" for name in RSAR_CLASSES)
SHUFFLED_PROBABILITY_FIELDS = tuple(
    f"shuffled_p_clip_{name}" for name in RSAR_CLASSES)
BOX_FIELDNAMES = (
    "run_signature",
    "image",
    "image_stem",
    "image_path",
    "split",
    "corruption",
    "detection_index",
    "detector_label_id",
    "detector_label",
    "matched_gt_label_id",
    "matched_gt_label",
    "match_category",
    "detector_score",
    "max_rotated_iou",
    "box_cx",
    "box_cy",
    "box_width",
    "box_height",
    "box_area",
    "aspect_ratio",
    "angle_radians",
    *PROBABILITY_FIELDS,
    "p_clip_detector_label",
    "sarclip_top1_label_id",
    "sarclip_top1_label",
    "top1_probability",
    "top2_probability",
    "margin",
    "normalized_entropy",
    "agreement",
    "legacy_score",
    "adaptive_score",
    "evidence_veto_score",
    "evidence_veto_triggered",
    "fixed_penalty_score",
    "disagreement_threshold_score",
    *SHUFFLED_PROBABILITY_FIELDS,
    "shuffled_p_clip_detector_label",
    "shuffled_sarclip_top1_label_id",
    "shuffled_sarclip_top1_label",
    "shuffled_agreement",
    "shuffled_moved",
    "shuffled_legacy_score",
    "before_score_thr",
    "after_score_thr",
    "legacy_after_score_thr",
    "adaptive_after_score_thr",
    "evidence_after_score_thr",
    "fixed_penalty_after_score_thr",
    "disagreement_threshold_after_score_thr",
    "shuffled_legacy_after_score_thr",
    "entered_student",
    "enters_student_legacy",
    "enters_student_adaptive",
    "enters_student_evidence_veto",
    "enters_student_fixed_penalty",
    "enters_student_disagreement_threshold",
    "enters_student_shuffled_legacy",
)

METHOD_SCORE_COLUMNS = {
    "legacy": "legacy_score",
    "adaptive_blend": "adaptive_score",
    "evidence_veto": "evidence_veto_score",
    "fixed_disagreement_penalty": "fixed_penalty_score",
    "disagreement_threshold": "disagreement_threshold_score",
    "shuffled_legacy": "shuffled_legacy_score",
}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return float(default if value is None or not value.strip() else value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclasses.dataclass(frozen=True)
class ScoreSettings:
    """Parameters mirroring the implemented modes in :mod:`sfod.cga`."""

    score_thr: float = 0.9
    legacy_detector_weight: float = 0.7
    adaptive_weight_min: float = 0.3
    adaptive_weight_max: float = 0.95
    fixed_penalty_delta: float = 0.1
    disagreement_score_thr: float = 0.95
    disagreement_drop_score: float = 0.0
    evidence_pred_hi: float = 0.90
    evidence_label_lo: float = 0.05
    evidence_margin: float = 0.60
    evidence_entropy: float = 0.35
    evidence_penalty: float = 0.0
    evidence_skip_context: bool = False

    @classmethod
    def from_environment(cls, score_thr: float) -> "ScoreSettings":
        return cls(
            score_thr=float(score_thr),
            legacy_detector_weight=_env_float("CGA_BLEND_DET_WEIGHT", 0.7),
            adaptive_weight_min=_env_float("CGA_ADAPT_W_MIN", 0.3),
            adaptive_weight_max=_env_float("CGA_ADAPT_W_MAX", 0.95),
            fixed_penalty_delta=_env_float("CGA_DISAGREE_DELTA", 0.1),
            disagreement_score_thr=_env_float("CGA_DISAGREE_SCORE_THR", 0.95),
            disagreement_drop_score=_env_float("CGA_DROP_SCORE", 0.0),
            evidence_pred_hi=_env_float("CGA_VETO_PRED_HI", 0.90),
            evidence_label_lo=_env_float("CGA_VETO_LABEL_LO", 0.05),
            evidence_margin=_env_float("CGA_VETO_MARGIN", 0.60),
            evidence_entropy=_env_float("CGA_VETO_ENTROPY", 0.35),
            evidence_penalty=_env_float("CGA_VETO_PENALTY", 0.0),
            evidence_skip_context=_env_bool("CGA_VETO_SKIP_CONTEXT", False),
        )


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy divided by log(K), matching ``sfod.cga``."""

    prob = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    if prob.ndim != 1 or not len(prob):
        raise ValueError("probabilities must be a non-empty vector")
    entropy = -float(np.sum(prob * np.log(prob)))
    return entropy / math.log(len(prob)) if len(prob) > 1 else 0.0


def _group_map(
    classes: Sequence[str], groups: Sequence[Sequence[str]]
) -> dict[int, int]:
    name_to_id = {name: index for index, name in enumerate(classes)}
    result: dict[int, int] = {}
    for group_id, group in enumerate(groups):
        for name in group:
            if name in name_to_id:
                result[name_to_id[name]] = group_id
    return result


def probability_features(probabilities: Sequence[float]) -> dict[str, Any]:
    prob = np.asarray(probabilities, dtype=np.float64)
    if prob.shape != (len(RSAR_CLASSES),):
        raise ValueError(
            f"expected {len(RSAR_CLASSES)} SARCLIP probabilities, got {prob.shape}")
    if not np.all(np.isfinite(prob)):
        raise ValueError("SARCLIP probabilities contain NaN or infinity")
    order = np.argsort(-prob, kind="stable")
    top1_id = int(order[0])
    top2_id = int(order[1])
    top1 = float(prob[top1_id])
    top2 = float(prob[top2_id])
    return {
        "top1_id": top1_id,
        "top2_id": top2_id,
        "top1_probability": top1,
        "top2_probability": top2,
        "margin": top1 - top2,
        "normalized_entropy": normalized_entropy(prob),
    }


def evidence_veto_trigger(
    probabilities: Sequence[float],
    detector_label: int,
    settings: ScoreSettings,
    *,
    classes: Sequence[str] = RSAR_CLASSES,
    confusion_groups: Sequence[Sequence[str]] = DEFAULT_CONFUSION_GROUPS,
    context_classes: Sequence[str] = CONTEXT_CLASSES,
) -> bool:
    """Apply the exact inclusive evidence-veto thresholds from ``sfod.cga``."""

    prob = np.asarray(probabilities, dtype=np.float64)
    features = probability_features(prob)
    prediction = int(features["top1_id"])
    detector_label = int(detector_label)
    if prediction == detector_label:
        return False

    groups = _group_map(classes, confusion_groups)
    same_group = (
        prediction in groups
        and detector_label in groups
        and groups[prediction] == groups[detector_label]
    )
    context_ids = {classes.index(name) for name in context_classes if name in classes}
    skip_context = settings.evidence_skip_context and (
        prediction in context_ids or detector_label in context_ids)
    return bool(
        prob[prediction] >= settings.evidence_pred_hi
        and prob[detector_label] <= settings.evidence_label_lo
        and features["margin"] >= settings.evidence_margin
        and features["normalized_entropy"] <= settings.evidence_entropy
        and not same_group
        and not skip_context
    )


def score_box_variants(
    detector_score: float,
    detector_label: int,
    probabilities: Sequence[float],
    settings: ScoreSettings,
) -> dict[str, Any]:
    """Compute all non-shuffled counterfactual scores for one detection."""

    score = float(detector_score)
    label = int(detector_label)
    prob = np.asarray(probabilities, dtype=np.float64)
    features = probability_features(prob)
    prediction = int(features["top1_id"])
    agreement = prediction == label
    label_probability = float(prob[label])

    legacy = score
    adaptive = score
    fixed = score
    threshold = score
    if not agreement:
        weight = settings.legacy_detector_weight
        legacy = score * weight + label_probability * (1.0 - weight)

        adaptive_weight = (
            settings.adaptive_weight_min
            + (settings.adaptive_weight_max - settings.adaptive_weight_min) * score
        )
        adaptive_weight = min(1.0, max(0.0, adaptive_weight))
        adaptive = score * adaptive_weight + label_probability * (1.0 - adaptive_weight)

        fixed = max(0.0, score - settings.fixed_penalty_delta)
        if score < settings.disagreement_score_thr:
            threshold = settings.disagreement_drop_score

    evidence_triggered = evidence_veto_trigger(prob, label, settings)
    evidence = score * settings.evidence_penalty if evidence_triggered else score
    return {
        **features,
        "agreement": agreement,
        "p_clip_detector_label": label_probability,
        "legacy_score": float(legacy),
        "adaptive_score": float(adaptive),
        "evidence_veto_score": float(evidence),
        "evidence_veto_triggered": evidence_triggered,
        "fixed_penalty_score": float(fixed),
        "disagreement_threshold_score": float(threshold),
    }


def shuffled_legacy_scores(
    detector_scores: Sequence[float],
    detector_labels: Sequence[int],
    probability_vectors: Sequence[Sequence[float]],
    identities: Sequence[str],
    *,
    seed: int,
    detector_weight: float,
    exclude_ids: Sequence[int] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the production stratified full-vector shuffled legacy control."""

    scores = np.asarray(detector_scores, dtype=np.float64)
    labels = np.asarray(detector_labels, dtype=np.int64)
    shuffled, source_indices = stratified_shuffle_probability_vectors(
        probability_vectors,
        scores,
        labels,
        identities,
        seed=seed,
        exclude_ids=exclude_ids,
    )
    predictions = np.argmax(shuffled, axis=1)
    label_probabilities = shuffled[np.arange(len(labels)), labels]
    disagreement = predictions != labels
    if exclude_ids:
        disagreement &= ~np.isin(labels, np.asarray(exclude_ids, dtype=np.int64))
    rescored = scores.copy()
    rescored[disagreement] = (
        scores[disagreement] * detector_weight
        + label_probabilities[disagreement] * (1.0 - detector_weight)
    )
    return shuffled, label_probabilities, rescored, source_indices


def classify_match(max_iou: float, detector_label: int, matched_gt_label: int) -> str:
    """Classify one detection with explicit boundary ownership.

    * IoU >= 0.5 belongs to TP/wrong-class (0.5 is included).
    * 0.1 < IoU < 0.5 belongs to localization-error.
    * IoU <= 0.1 belongs to pure-FP (0.1 is included).
    """

    iou = float(max_iou)
    if iou >= 0.5:
        return "TP" if int(detector_label) == int(matched_gt_label) else "wrong-class"
    if iou > 0.1:
        return "localization-error"
    return "pure-FP"


def match_detections_to_gt(
    detection_boxes: np.ndarray,
    detection_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match every detection to its maximum rotated-IoU ground truth."""

    detection_boxes = np.asarray(detection_boxes, dtype=np.float32).reshape(-1, 5)
    detection_labels = np.asarray(detection_labels, dtype=np.int64).reshape(-1)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 5)
    gt_labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    count = len(detection_boxes)
    if count == 0:
        return (
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            [],
        )
    if len(gt_boxes) == 0:
        max_ious = np.zeros(count, dtype=np.float32)
        matched_labels = np.full(count, -1, dtype=np.int64)
    else:
        import torch
        from mmrotate.core.bbox import rbbox_overlaps

        overlaps = rbbox_overlaps(
            torch.from_numpy(detection_boxes), torch.from_numpy(gt_boxes))
        overlaps_np = overlaps.detach().cpu().numpy()
        matched_indices = np.argmax(overlaps_np, axis=1)
        max_ious = overlaps_np[np.arange(count), matched_indices].astype(np.float32)
        matched_labels = gt_labels[matched_indices].astype(np.int64)
    categories = [
        classify_match(iou, det_label, gt_label)
        for iou, det_label, gt_label in zip(
            max_ious, detection_labels, matched_labels)
    ]
    return max_ious, matched_labels, categories


def parse_dota_annotation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse DOTA polygons and convert them to le90 OBBs."""

    from mmrotate.core.bbox import poly2obb_np

    boxes: list[np.ndarray] = []
    labels: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.strip().split()
            if not parts:
                continue
            if len(parts) < 10:
                raise ValueError(f"malformed DOTA row {path}:{line_number}")
            class_name = parts[8]
            if class_name not in CLASS_TO_ID:
                raise ValueError(
                    f"unknown class {class_name!r} at {path}:{line_number}")
            polygon = np.asarray(parts[:8], dtype=np.float32)
            obb = poly2obb_np(polygon, "le90")
            if obb is None or not np.all(np.isfinite(obb)):
                raise ValueError(f"invalid polygon at {path}:{line_number}")
            boxes.append(np.asarray(obb, dtype=np.float32))
            labels.append(CLASS_TO_ID[class_name])
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(boxes), np.asarray(labels, dtype=np.int64)


def obb_to_aabb(boxes: np.ndarray) -> np.ndarray:
    """Convert le90 ``cx,cy,w,h,a`` boxes to enclosing ``x1,y1,x2,y2``."""

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 5)
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cosine = np.abs(np.cos(boxes[:, 4]))
    sine = np.abs(np.sin(boxes[:, 4]))
    width = cosine * boxes[:, 2] + sine * boxes[:, 3]
    height = sine * boxes[:, 2] + cosine * boxes[:, 3]
    return np.stack(
        (
            boxes[:, 0] - width / 2.0,
            boxes[:, 1] - height / 2.0,
            boxes[:, 0] + width / 2.0,
            boxes[:, 1] + height / 2.0,
        ),
        axis=1,
    ).astype(np.float32)


def flatten_detector_results(
    result: Any,
    *,
    min_score: float,
    max_boxes: int,
    num_classes: int = len(RSAR_CLASSES),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-class mmrotate results, filter, then cap by raw score."""

    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, (list, tuple)) or len(result) != num_classes:
        raise ValueError(
            f"expected {num_classes} per-class detector arrays, got "
            f"{type(result).__name__} length={getattr(result, '__len__', lambda: '?')()}")
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for class_id, class_result in enumerate(result):
        array = np.asarray(class_result, dtype=np.float32)
        if array.size == 0:
            continue
        if array.ndim != 2 or array.shape[1] < 6:
            raise ValueError(
                f"class {class_id} detector result must be Nx6, got {array.shape}")
        keep = np.isfinite(array[:, 5]) & (array[:, 5] >= float(min_score))
        array = array[keep]
        if not len(array):
            continue
        boxes.append(array[:, :5])
        scores.append(array[:, 5])
        labels.append(np.full(len(array), class_id, dtype=np.int64))
    if not boxes:
        return (
            np.zeros((0, 5), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.int64),
        )
    all_boxes = np.concatenate(boxes)
    all_scores = np.concatenate(scores)
    all_labels = np.concatenate(labels)
    order = np.argsort(-all_scores, kind="stable")[: int(max_boxes)]
    return all_boxes[order], all_scores[order], all_labels[order]


def inference_without_cga(
    model: Any,
    image_path: str,
    inference_fn: Callable[..., Any],
) -> Any:
    """Call ``inference_detector`` while making raw-CGA behavior explicit."""

    parameters = inspect.signature(inference_fn).parameters.values()
    accepts_keyword = any(
        parameter.name == "with_cga"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if accepts_keyword:
        return inference_fn(model, image_path, with_cga=False)

    simple_test = getattr(model, "simple_test", None)
    if simple_test is not None:
        simple_parameters = inspect.signature(simple_test).parameters
        with_cga = simple_parameters.get("with_cga")
        if with_cga is not None and with_cga.default is not False:
            raise RuntimeError(
                "inference_detector cannot pass with_cga=False and model.simple_test "
                "does not default it to False")
    # Standard mmdet inference_detector accepts no extra kwargs.  Its forward
    # path calls this project's simple_test, whose with_cga default is False.
    return inference_fn(model, image_path)


def build_cga_from_environment() -> Any:
    """Construct SARCLIP CGA once, using the LoRA path from the environment."""

    from sfod.cga import CGA

    os.environ.setdefault("CGA_SCORER", "sarclip")
    os.environ.setdefault("CGA_BACKEND", "sarclip")
    lora = os.environ.get("SARCLIP_LORA", "").strip()
    if not lora:
        raise RuntimeError(
            "SARCLIP_LORA must be set for the teacher prediction box audit")
    lora_path = Path(lora).expanduser()
    if not lora_path.is_absolute():
        lora_path = (REPO_ROOT / lora_path).resolve()
    if not lora_path.is_file():
        raise FileNotFoundError(f"SARCLIP_LORA does not exist: {lora_path}")
    os.environ["SARCLIP_LORA"] = str(lora_path)

    templates = (
        os.environ.get("CGA_TEMPLATES")
        or "A SAR image of a {};This SAR patch shows a {}"
    ).split(";")
    cga = CGA(
        list(RSAR_CLASSES),
        model=os.environ.get("SARCLIP_MODEL", "ViT-B-32"),
        pretrained=os.environ.get(
            "SARCLIP_PRETRAINED",
            "/home/storageSDA1/Dataset/SARCLIP/ViT-B-32/"
            "vit_b_32_model.safetensors",
        ),
        cache_dir=os.environ.get(
            "SARCLIP_CACHE_DIR",
            "/home/storageSDA1/Dataset/SARCLIP/ViT-B-32",
        ),
        precision=os.environ.get("SARCLIP_PRECISION", "fp32"),
        templates=templates,
        tau=_env_float("CGA_TAU", 100.0),
        expand_ratio=_env_float("CGA_EXPAND_RATIO", 0.4),
        force_grayscale=_env_bool("CGA_FORCE_GRAYSCALE", False),
        backend="sarclip",
    )
    return cga


def run_cga_in_chunks(
    cga: Any,
    image_path: Path,
    detection_boxes: np.ndarray,
    detection_scores: np.ndarray,
    detection_labels: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    aabbs = obb_to_aabb(detection_boxes)
    chunks: list[np.ndarray] = []
    for start in range(0, len(aabbs), int(chunk_size)):
        stop = min(len(aabbs), start + int(chunk_size))
        probability, _ = cga(
            str(image_path),
            aabbs[start:stop],
            detection_scores[start:stop],
            detection_labels[start:stop],
        )
        probability = np.asarray(probability, dtype=np.float64)
        expected = (stop - start, len(RSAR_CLASSES))
        if probability.shape != expected:
            raise ValueError(
                f"CGA returned {probability.shape}, expected {expected}")
        chunks.append(probability)
    if not chunks:
        return np.zeros((0, len(RSAR_CLASSES)), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def build_image_records(
    *,
    run_signature: str,
    image_path: Path,
    split: str,
    corruption: str,
    detection_boxes: np.ndarray,
    detection_scores: np.ndarray,
    detection_labels: np.ndarray,
    max_ious: np.ndarray,
    matched_gt_labels: np.ndarray,
    categories: Sequence[str],
    probabilities: np.ndarray,
    settings: ScoreSettings,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    variants = [
        score_box_variants(score, label, probability, settings)
        for score, label, probability in zip(
            detection_scores, detection_labels, probabilities)
    ]
    # Shuffled fields are initialized to the real-vector legacy result here.
    # They are rebuilt globally per split after all resumable part CSVs merge;
    # doing the control at image scope would make singleton images degenerate.
    del shuffle_seed

    records: list[dict[str, Any]] = []
    for index, (
        box,
        detector_score,
        detector_label,
        max_iou,
        matched_label,
        category,
        probability,
        variant,
    ) in enumerate(
        zip(
            detection_boxes,
            detection_scores,
            detection_labels,
            max_ious,
            matched_gt_labels,
            categories,
            probabilities,
            variants,
        )
    ):
        width = float(box[2])
        height = float(box[3])
        shorter = max(min(abs(width), abs(height)), 1e-12)
        before = float(detector_score) >= settings.score_thr
        after = {
            method: float(variant[column]) >= settings.score_thr
            for method, column in METHOD_SCORE_COLUMNS.items()
            if method != "shuffled_legacy"
        }
        after["shuffled_legacy"] = (
            float(variant["legacy_score"]) >= settings.score_thr)
        record: dict[str, Any] = {
            "run_signature": run_signature,
            "image": image_path.name,
            "image_stem": image_path.stem,
            "image_path": str(image_path),
            "split": split,
            "corruption": corruption,
            "detection_index": index,
            "detector_label_id": int(detector_label),
            "detector_label": RSAR_CLASSES[int(detector_label)],
            "matched_gt_label_id": int(matched_label),
            "matched_gt_label": (
                RSAR_CLASSES[int(matched_label)] if int(matched_label) >= 0 else ""
            ),
            "match_category": category,
            "detector_score": float(detector_score),
            "max_rotated_iou": float(max_iou),
            "box_cx": float(box[0]),
            "box_cy": float(box[1]),
            "box_width": width,
            "box_height": height,
            "box_area": abs(width * height),
            "aspect_ratio": max(abs(width), abs(height)) / shorter,
            "angle_radians": float(box[4]),
            "p_clip_detector_label": variant["p_clip_detector_label"],
            "sarclip_top1_label_id": int(variant["top1_id"]),
            "sarclip_top1_label": RSAR_CLASSES[int(variant["top1_id"])],
            "top1_probability": variant["top1_probability"],
            "top2_probability": variant["top2_probability"],
            "margin": variant["margin"],
            "normalized_entropy": variant["normalized_entropy"],
            "agreement": int(bool(variant["agreement"])),
            "legacy_score": variant["legacy_score"],
            "adaptive_score": variant["adaptive_score"],
            "evidence_veto_score": variant["evidence_veto_score"],
            "evidence_veto_triggered": int(
                bool(variant["evidence_veto_triggered"])),
            "fixed_penalty_score": variant["fixed_penalty_score"],
            "disagreement_threshold_score": variant[
                "disagreement_threshold_score"],
            "shuffled_p_clip_detector_label": variant[
                "p_clip_detector_label"],
            "shuffled_sarclip_top1_label_id": int(variant["top1_id"]),
            "shuffled_sarclip_top1_label": RSAR_CLASSES[int(variant["top1_id"])],
            "shuffled_agreement": int(bool(variant["agreement"])),
            "shuffled_moved": 0,
            "shuffled_legacy_score": variant["legacy_score"],
            "before_score_thr": int(before),
            # Compatibility alias: "after" and "entered_student" refer to
            # legacy, the historical deployed CGA mode.
            "after_score_thr": int(after["legacy"]),
            "legacy_after_score_thr": int(after["legacy"]),
            "adaptive_after_score_thr": int(after["adaptive_blend"]),
            "evidence_after_score_thr": int(after["evidence_veto"]),
            "fixed_penalty_after_score_thr": int(
                after["fixed_disagreement_penalty"]),
            "disagreement_threshold_after_score_thr": int(
                after["disagreement_threshold"]),
            "shuffled_legacy_after_score_thr": int(after["shuffled_legacy"]),
            "entered_student": int(after["legacy"]),
            "enters_student_legacy": int(after["legacy"]),
            "enters_student_adaptive": int(after["adaptive_blend"]),
            "enters_student_evidence_veto": int(after["evidence_veto"]),
            "enters_student_fixed_penalty": int(
                after["fixed_disagreement_penalty"]),
            "enters_student_disagreement_threshold": int(
                after["disagreement_threshold"]),
            "enters_student_shuffled_legacy": int(after["shuffled_legacy"]),
        }
        record.update(
            {
                field: float(probability[class_id])
                for class_id, field in enumerate(PROBABILITY_FIELDS)
            }
        )
        record.update(
            {
                field: float(probability[class_id])
                for class_id, field in enumerate(
                    SHUFFLED_PROBABILITY_FIELDS)
            }
        )
        records.append(record)
    return records


def apply_global_shuffled_control(
    records: Sequence[dict[str, Any]],
    settings: ScoreSettings,
    *,
    seed: int,
) -> dict[str, Any]:
    """Rebuild shuffled fields over every box in each audit split.

    The online control operates over the current inference batch.  The audit
    intentionally uses a larger pool -- all merged rows in one split -- so that
    the causal diagnostic does not collapse to per-image singletons.  Splits are
    never mixed.
    """

    diagnostics: dict[str, Any] = {
        "scope": "all merged boxes within each split",
        "seed": int(seed),
        "score_bins": list(CGA_SHUFFLE_SCORE_BINS),
        "moved": 0,
        "unmoved": 0,
        "real_agree": 0,
        "operative_agree": 0,
        "trigger_changed": 0,
        "splits": {},
    }
    split_names = sorted({str(record.get("split", "")) for record in records})
    for split in split_names:
        row_indices = [
            index for index, record in enumerate(records)
            if str(record.get("split", "")) == split
        ]
        if not row_indices:
            continue
        scores = np.asarray(
            [_as_float(records[index], "detector_score") for index in row_indices],
            dtype=np.float64,
        )
        labels = np.asarray(
            [int(records[index]["detector_label_id"]) for index in row_indices],
            dtype=np.int64,
        )
        probabilities = np.asarray(
            [
                [_as_float(records[index], field) for field in PROBABILITY_FIELDS]
                for index in row_indices
            ],
            dtype=np.float64,
        )
        identities = [
            "\0".join(
                (
                    split,
                    str(records[index].get("image_stem", "")),
                    str(records[index].get("detection_index", "")),
                )
            )
            for index in row_indices
        ]
        shuffled, label_probabilities, rescored, source_indices = (
            shuffled_legacy_scores(
                scores,
                labels,
                probabilities,
                identities,
                seed=seed,
                detector_weight=settings.legacy_detector_weight,
            )
        )
        moved = source_indices != np.arange(len(source_indices))
        real_agreement = np.argmax(probabilities, axis=1) == labels
        predictions = np.argmax(shuffled, axis=1)
        operative_agreement = predictions == labels
        if int(real_agreement.sum()) != int(operative_agreement.sum()):
            raise RuntimeError(
                f"shuffle agreement count changed inside split {split}"
            )

        bin_indices = np.searchsorted(
            np.asarray(CGA_SHUFFLE_SCORE_BINS), scores, side="right") - 1
        bin_indices = np.clip(
            bin_indices, 0, len(CGA_SHUFFLE_SCORE_BINS) - 2)
        stratum_sizes: dict[tuple[int, int], int] = {}
        for label, bin_index in zip(labels, bin_indices):
            key = (int(label), int(bin_index))
            stratum_sizes[key] = stratum_sizes.get(key, 0) + 1

        split_diagnostic = {
            "total": len(row_indices),
            "strata": len(stratum_sizes),
            "singleton_strata": sum(
                int(size == 1) for size in stratum_sizes.values()),
            "moved": int(moved.sum()),
            "unmoved": int((~moved).sum()),
            "real_agree": int(real_agreement.sum()),
            "operative_agree": int(operative_agreement.sum()),
            "trigger_changed": int(
                np.count_nonzero(real_agreement != operative_agreement)),
        }
        diagnostics["splits"][split] = split_diagnostic
        for key in (
            "moved", "unmoved", "real_agree", "operative_agree",
            "trigger_changed",
        ):
            diagnostics[key] += split_diagnostic[key]

        for local_index, record_index in enumerate(row_indices):
            record = records[record_index]
            for class_id, field in enumerate(SHUFFLED_PROBABILITY_FIELDS):
                record[field] = float(shuffled[local_index, class_id])
            top1_id = int(predictions[local_index])
            record["shuffled_p_clip_detector_label"] = float(
                label_probabilities[local_index])
            record["shuffled_sarclip_top1_label_id"] = top1_id
            record["shuffled_sarclip_top1_label"] = RSAR_CLASSES[top1_id]
            record["shuffled_agreement"] = int(
                operative_agreement[local_index])
            record["shuffled_moved"] = int(moved[local_index])
            record["shuffled_legacy_score"] = float(rescored[local_index])
            enters_student = int(
                float(rescored[local_index]) >= settings.score_thr)
            record["shuffled_legacy_after_score_thr"] = enters_student
            record["enters_student_shuffled_legacy"] = enters_student
    return diagnostics


def fixed_train_research_pool(
    stems: Sequence[str], pool_size: int = TRAIN_POOL_SIZE
) -> list[str]:
    """Build the protocol-mandated train pool independently of CLI seed."""

    ordered = sorted(str(stem) for stem in stems)
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate stems are not allowed")
    if len(ordered) < int(pool_size):
        raise ValueError(
            f"need at least {pool_size} train stems, found {len(ordered)}")
    return random.Random(TRAIN_POOL_SEED).sample(ordered, int(pool_size))


def select_split_stems(
    stems: Sequence[str],
    *,
    split: str,
    limit: int,
    seed: int,
    train_pool_size: int = TRAIN_POOL_SIZE,
) -> list[str]:
    if limit < 0:
        raise ValueError("sample limit must be non-negative")
    if split == "train":
        pool = fixed_train_research_pool(stems, train_pool_size)
        if limit > len(pool):
            raise ValueError(f"--num-train={limit} exceeds pool size {len(pool)}")
        return pool[:limit]
    if split == "val":
        ordered = sorted(str(stem) for stem in stems)
        if limit > len(ordered):
            raise ValueError(f"--num-val={limit} exceeds val size {len(ordered)}")
        return random.Random(int(seed)).sample(ordered, limit)
    raise ValueError(f"unsupported audit split: {split}")


def index_images(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise ValueError(f"duplicate image stem {path.stem!r} in {directory}")
        result[path.stem] = path
    return result


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
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def ensure_csv_header(path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or ())
            if existing_fields == list(fieldnames):
                return
            existing_rows = [dict(row) for row in reader]
        # Derived shuffled columns can be added safely to an interrupted audit.
        # The final split-global rebuild overwrites their placeholder values.
        write_csv_records_atomic(path, existing_rows, fieldnames)
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        handle.flush()
        os.fsync(handle.fileno())


def append_csv_rows(
    path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]
) -> int:
    ensure_csv_header(path, fieldnames)
    materialized = list(rows)
    if not materialized:
        return 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writerows(materialized)
        handle.flush()
        os.fsync(handle.fileno())
    return len(materialized)


def write_csv_records_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed_images(
    status_path: Path, expected_signature: Optional[str] = None
) -> set[str]:
    """Return stems whose latest valid status event is ``completed``."""

    if not status_path.is_file():
        return set()
    latest: dict[str, str] = {}
    with status_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                # A power loss can leave one partial trailing JSONL record.  It is
                # safe to retry that image because final CSV merge is deduplicated.
                continue
            if expected_signature and payload.get("run_signature") != expected_signature:
                continue
            stem = str(payload.get("image_stem", ""))
            if stem:
                latest[stem] = str(payload.get("status", ""))
    return {stem for stem, status in latest.items() if status == "completed"}


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("split", "")),
        str(row.get("image_stem", row.get("image", ""))),
        str(row.get("detection_index", "")),
    )


def merge_part_csvs(
    part_paths: Sequence[Path],
    output_path: Path,
    fieldnames: Optional[Sequence[str]] = None,
) -> int:
    """Merge parts atomically and keep the last duplicate detection record."""

    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    inferred: list[str] = []
    for part_path in part_paths:
        if not part_path.is_file() or not part_path.stat().st_size:
            continue
        with part_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for name in reader.fieldnames:
                    if name not in inferred:
                        inferred.append(name)
            for row in reader:
                rows[_row_identity(row)] = dict(row)
    columns = list(fieldnames or inferred)
    if not columns:
        columns = list(BOX_FIELDNAMES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        def identity_sort_key(key: tuple[str, str, str]) -> tuple[Any, ...]:
            detection_index: tuple[int, Any]
            if key[2].isdigit():
                detection_index = (0, int(key[2]))
            else:
                detection_index = (1, key[2])
            return key[0], key[1], detection_index

        for identity in sorted(rows, key=identity_sort_key):
            writer.writerow(rows[identity])
        temporary = Path(handle.name)
    os.replace(temporary, output_path)
    return len(rows)


def read_csv_records(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _as_float(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key, 0.0)
    return float(value) if value not in (None, "") else 0.0


def _as_bool(record: Mapping[str, Any], key: str) -> bool:
    value = record.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_method(
    records: Sequence[Mapping[str, Any]],
    score_column: str,
    score_thr: float,
    *,
    trigger_column: Optional[str] = None,
) -> dict[str, Any]:
    """Compute crossing-based removal metrics with zero-safe denominators."""

    baseline = [
        record for record in records
        if _as_float(record, "detector_score") >= float(score_thr)
    ]
    removed = [
        record for record in baseline
        if _as_float(record, score_column) < float(score_thr)
    ]
    baseline_by_category = {
        category: sum(record.get("match_category") == category for record in baseline)
        for category in MATCH_CATEGORIES
    }
    removed_by_category = {
        category: sum(record.get("match_category") == category for record in removed)
        for category in MATCH_CATEGORIES
    }
    baseline_fp = len(baseline) - baseline_by_category["TP"]
    removed_fp = len(removed) - removed_by_category["TP"]
    modified_not_crossed = sum(
        abs(_as_float(record, score_column) - _as_float(record, "detector_score"))
        > 1e-12
        and (
            (_as_float(record, "detector_score") >= score_thr)
            == (_as_float(record, score_column) >= score_thr)
        )
        for record in records
    )
    metrics: dict[str, Any] = {
        "candidates": len(records),
        "baseline_kept": len(baseline),
        "removed_total": len(removed),
        "baseline_fp": baseline_fp,
        "removed_fp": removed_fp,
        "removed_tp": removed_by_category["TP"],
        "fp_removal_precision": _safe_ratio(removed_fp, len(removed)),
        "fp_removal_recall": _safe_ratio(removed_fp, baseline_fp),
        "tp_damage_rate": _safe_ratio(
            removed_by_category["TP"], baseline_by_category["TP"]),
        "wrong_class_removal_rate": _safe_ratio(
            removed_by_category["wrong-class"],
            baseline_by_category["wrong-class"],
        ),
        "localization_error_removal_rate": _safe_ratio(
            removed_by_category["localization-error"],
            baseline_by_category["localization-error"],
        ),
        "pure_fp_removal_rate": _safe_ratio(
            removed_by_category["pure-FP"], baseline_by_category["pure-FP"]),
        "modified_not_crossed": modified_not_crossed,
        "fp_crossed_and_filtered": removed_fp,
        "tp_crossed_and_filtered": removed_by_category["TP"],
    }
    for category in MATCH_CATEGORIES:
        key = category.lower().replace("-", "_")
        metrics[f"baseline_{key}"] = baseline_by_category[category]
        metrics[f"removed_{key}"] = removed_by_category[category]

    if trigger_column:
        triggered = [record for record in baseline if _as_bool(record, trigger_column)]
        triggered_fp = sum(record.get("match_category") != "TP" for record in triggered)
        triggered_tp = len(triggered) - triggered_fp
        metrics.update(
            {
                "triggered_baseline": len(triggered),
                "triggered_fp": triggered_fp,
                "triggered_tp": triggered_tp,
                "trigger_fp_precision": _safe_ratio(triggered_fp, len(triggered)),
                "trigger_fp_recall": _safe_ratio(triggered_fp, baseline_fp),
                "trigger_tp_damage_rate": _safe_ratio(
                    triggered_tp, baseline_by_category["TP"]),
            }
        )
    return metrics


def _format_rate(value: float) -> str:
    return f"{100.0 * float(value):.3f}%"


def _metrics_table(
    rows: Sequence[tuple[str, str, Mapping[str, Any]]]
) -> list[str]:
    lines = [
        "| group | method | N | kept | removed | FP precision | FP recall | "
        "TP damage | evidence trigger FP recall | wrong-class removal | "
        "localization removal | pure-FP removal | modified/no crossing | "
        "FP crossed | TP crossed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, method, metrics in rows:
        lines.append(
            "| {group} | {method} | {candidates} | {baseline_kept} | "
            "{removed_total} | {fp_precision} | {fp_recall} | {tp_damage} | "
            "{trigger_recall} | {wrong} | {localization} | {pure} | {modified} | "
            "{fp_crossed} | {tp_crossed} |".format(
                group=group,
                method=method,
                fp_precision=_format_rate(metrics["fp_removal_precision"]),
                fp_recall=_format_rate(metrics["fp_removal_recall"]),
                tp_damage=_format_rate(metrics["tp_damage_rate"]),
                trigger_recall=(
                    _format_rate(metrics["trigger_fp_recall"])
                    if "trigger_fp_recall" in metrics
                    else "—"
                ),
                wrong=_format_rate(metrics["wrong_class_removal_rate"]),
                localization=_format_rate(
                    metrics["localization_error_removal_rate"]),
                pure=_format_rate(metrics["pure_fp_removal_rate"]),
                modified=metrics["modified_not_crossed"],
                fp_crossed=metrics["fp_crossed_and_filtered"],
                tp_crossed=metrics["tp_crossed_and_filtered"],
                **metrics,
            )
        )
    return lines


def _bin_label(value: float, boundaries: Sequence[float]) -> str:
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if lower <= value < upper:
            return f"[{lower:g},{upper:g})"
    if value < boundaries[0]:
        return f"<{boundaries[0]:g}"
    return f">={boundaries[-1]:g}"


def _group_records(
    records: Sequence[Mapping[str, Any]],
    key_fn: Callable[[Mapping[str, Any]], str],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(key_fn(record), []).append(record)
    return sorted(grouped.items())


def render_summary(
    records: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    settings: ScoreSettings,
    *,
    parquet_error: Optional[str] = None,
) -> str:
    shuffle_diagnostic = metadata.get("shuffled_control", {})
    lines = [
        "# Teacher prediction box-level audit",
        "",
        "## Run metadata",
        "",
        f"- Run signature: `{metadata.get('signature', '')}`",
        f"- Config: `{metadata.get('config', '')}`",
        f"- Checkpoint: `{metadata.get('checkpoint', '')}`",
        f"- Corruption: `{metadata.get('corruption', '')}`",
        f"- Splits: `{','.join(metadata.get('splits', []))}`",
        f"- Completed images: {metadata.get('completed_images', 0)}",
        f"- Failed images: {metadata.get('failed_images', 0)}",
        f"- Candidate boxes: {len(records)}",
        f"- Detector candidate floor: {metadata.get('min_det_score')}",
        f"- Student score threshold: {settings.score_thr}",
        f"- SARCLIP LoRA: `{metadata.get('sarclip_lora', '')}`",
        "- Detector inference: raw teacher checkpoint, `with_cga=False`; "
        "SARCLIP was applied once afterward.",
        "- Shuffled audit scope: all merged boxes within each split, grouped by "
        "detector label and fixed detector-score bin. Train and val never mix.",
        "- Online shuffled scope differs intentionally: production can only "
        "shuffle across all images in the current inference batch.",
        "- Shuffled score bins: `" + ",".join(
            f"{value:g}" for value in CGA_SHUFFLE_SCORE_BINS) + "`.",
        "",
    ]
    if parquet_error:
        lines.extend(
            [
                "- Parquet status: **ERROR** — CSV and this summary are valid, "
                f"but Parquet was not written: `{parquet_error}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Definitions",
            "",
            "- `TP`: rotated IoU >= 0.5 and detector class equals matched GT class.",
            "- `wrong-class`: rotated IoU >= 0.5 and classes differ.",
            "- `localization-error`: 0.1 < rotated IoU < 0.5.",
            "- `pure-FP`: rotated IoU <= 0.1.",
            "- FP removal treats wrong-class, localization-error and pure-FP as FP.",
            "- A removal is counted only when a box is above the baseline student "
            "threshold and crosses below it after the simulated method.",
            "- `entered_student`/`after_score_thr` are compatibility aliases for "
            "legacy; every simulated method also has its own field.",
            "- Shuffled-SARCLIP moves the complete six-class probability vector "
            "with a deterministic no-fixed-point cyclic permutation for every "
            "non-singleton `(detector label, score bin)` stratum. Operative "
            "argmax and `p(y_det)` both come from the moved vector.",
            "",
            "## Shuffled-SARCLIP assignment diagnostic",
            "",
            "| split | total | strata | singleton strata | moved | unmoved | "
            "real agree | operative agree | trigger membership changed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, diagnostic in sorted(
            shuffle_diagnostic.get("splits", {}).items()):
        lines.append(
            "| {split} | {total} | {strata} | {singleton_strata} | {moved} | "
            "{unmoved} | {real_agree} | {operative_agree} | "
            "{trigger_changed} |".format(split=split, **diagnostic)
        )
    if not shuffle_diagnostic.get("splits"):
        lines.append("| unavailable | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
    lines.extend(
        [
            "",
            "The real and operative agreement totals must match within every "
            "split. `trigger membership changed` measures whether the preserved "
            "trigger count was reassigned to different instances.",
            "",
            "## Overall crossing metrics",
            "",
        ]
    )
    overall_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    overall_metrics: dict[str, Mapping[str, Any]] = {}
    for method, score_column in METHOD_SCORE_COLUMNS.items():
        metrics = summarize_method(
            records,
            score_column,
            settings.score_thr,
            trigger_column=(
                "evidence_veto_triggered" if method == "evidence_veto" else None),
        )
        overall_metrics[method] = metrics
        overall_rows.append(("all", method, metrics))
    lines.extend(_metrics_table(overall_rows))

    evidence = overall_metrics["evidence_veto"]
    lines.extend(
        [
            "",
            "## Evidence-veto trigger diagnostic",
            "",
            "Evidence trigger recall is reported separately from threshold-crossing "
            "recall so a low-penalty veto cannot hide a low trigger rate.",
            "",
            "| baseline FP | triggered | triggered FP | triggered TP | trigger FP "
            "precision | trigger FP recall | trigger TP damage | crossing FP recall |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| {baseline_fp} | {triggered_baseline} | {triggered_fp} | "
            "{triggered_tp} | {precision} | {recall} | {damage} | {crossing} |".format(
                precision=_format_rate(evidence["trigger_fp_precision"]),
                recall=_format_rate(evidence["trigger_fp_recall"]),
                damage=_format_rate(evidence["trigger_tp_damage_rate"]),
                crossing=_format_rate(evidence["fp_removal_recall"]),
                **evidence,
            ),
        ]
    )

    group_specs = [
        (
            "Per detector class",
            [
                (
                    class_name,
                    [
                        row for row in records
                        if str(row.get("detector_label", "")) == class_name
                    ],
                )
                for class_name in RSAR_CLASSES
            ],
        ),
        (
            "Detector-score bins",
            _group_records(
                records,
                lambda row: _bin_label(
                    _as_float(row, "detector_score"),
                    (0.5, 0.7, 0.8, 0.9, 0.95, 1.000001),
                ),
            ),
        ),
        (
            "Normalized-entropy bins",
            _group_records(
                records,
                lambda row: _bin_label(
                    _as_float(row, "normalized_entropy"),
                    (0.0, 0.25, 0.5, 0.75, 1.000001),
                ),
            ),
        ),
        (
            "SARCLIP-margin bins",
            _group_records(
                records,
                lambda row: _bin_label(
                    _as_float(row, "margin"),
                    (0.0, 0.1, 0.3, 0.6, 1.000001),
                ),
            ),
        ),
    ]
    for title, groups in group_specs:
        rows: list[tuple[str, str, Mapping[str, Any]]] = []
        for group, subset in groups:
            for method, score_column in METHOD_SCORE_COLUMNS.items():
                rows.append(
                    (
                        group,
                        method,
                        summarize_method(
                            subset,
                            score_column,
                            settings.score_thr,
                            trigger_column=(
                                "evidence_veto_triggered"
                                if method == "evidence_veto"
                                else None
                            ),
                        ),
                    )
                )
        lines.extend(["", f"## {title}", ""])
        lines.extend(_metrics_table(rows))

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "A low TP-damage rate is not sufficient evidence of a useful veto. "
            "Inspect evidence trigger FP recall and crossing FP recall together; "
            "near-zero values indicate a safe but inactive filter.",
            "",
        ]
    )
    return "\n".join(lines)


def attempt_parquet(csv_path: Path, parquet_path: Path) -> Optional[str]:
    """Write Parquet or return a clear, non-destructive dependency error."""

    try:
        import pandas as pd

        frame = pd.read_csv(csv_path)
        frame.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pandas reports optional-engine errors at runtime.
        return f"{type(exc).__name__}: {exc}"
    return None


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_splits(value: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in value.split(",") if item.strip())
    if not splits or any(split not in {"train", "val"} for split in splits):
        raise argparse.ArgumentTypeError("--splits must contain only train,val")
    if len(set(splits)) != len(splits):
        raise argparse.ArgumentTypeError("--splits contains duplicates")
    return splits


def default_output_root() -> Path:
    research_root = os.environ.get("AUTO_RESEARCH_ROOT", "").strip()
    if research_root:
        return Path(research_root).expanduser() / "box_audit"
    return REPO_ROOT / "work_dirs" / "auto_research" / "box_audit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--corruption", default="chaff")
    parser.add_argument("--splits", type=_parse_splits, default=("train", "val"))
    parser.add_argument("--num-train", type=int, default=1500)
    parser.add_argument("--num-val", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--min-det-score", type=float, default=0.5)
    parser.add_argument("--score-thr", type=float, default=0.9)
    parser.add_argument("--max-boxes-per-image", type=int, default=512)
    parser.add_argument("--cga-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rebuild-shuffled-only",
        action="store_true",
        help=(
            "rebuild shuffled columns and summary from an existing merged CSV; "
            "does not initialize the detector or SARCLIP"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_train < 0 or args.num_val < 0:
        raise ValueError("sample counts must be non-negative")
    if not 0.0 <= args.min_det_score <= 1.0:
        raise ValueError("--min-det-score must be in [0,1]")
    if not 0.0 <= args.score_thr <= 1.0:
        raise ValueError("--score-thr must be in [0,1]")
    if args.max_boxes_per_image <= 0:
        raise ValueError("--max-boxes-per-image must be positive")
    if args.cga_chunk_size <= 0:
        raise ValueError("--cga-chunk-size must be positive")
    if not args.corruption or any(part in {"..", ""} for part in Path(args.corruption).parts):
        raise ValueError("invalid corruption name")


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _signature(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prepare_manifest(
    output_root: Path,
    parameters: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    signature = _signature(parameters)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError(
                f"output already initialized at {output_root}; use --resume or "
                "choose another --output-root")
        if existing.get("signature") != signature:
            raise RuntimeError(
                "resume manifest mismatch; refusing to mix samples/settings")
        return existing
    if resume:
        raise RuntimeError(f"--resume requested but {manifest_path} is missing")
    existing_files = (
        [path for path in output_root.rglob("*") if path.is_file()]
        if output_root.exists()
        else []
    )
    if existing_files:
        raise RuntimeError(
            f"output contains files but has no manifest: {output_root}; "
            "choose another --output-root")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "signature": signature,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parameters": parameters,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def rebuild_existing_shuffled_outputs(output_root: Path) -> int:
    """Rebuild the causal-control columns without any GPU inference."""

    output_root = output_root.expanduser().resolve()
    manifest_path = output_root / "manifest.json"
    metadata_path = output_root / "run_metadata.json"
    csv_path = output_root / "prediction_boxes.csv"
    for required in (manifest_path, metadata_path, csv_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parameters = manifest.get("parameters", {})
    score_payload = parameters.get("score_settings")
    if not isinstance(score_payload, dict):
        raise ValueError("manifest is missing parameters.score_settings")
    settings = ScoreSettings(**score_payload)
    seed = int(parameters.get("seed", 0))
    records = read_csv_records(csv_path)
    diagnostic = apply_global_shuffled_control(
        records, settings, seed=seed)
    write_csv_records_atomic(csv_path, records, BOX_FIELDNAMES)

    parquet_path = output_root / "prediction_boxes.parquet"
    parquet_error = attempt_parquet(csv_path, parquet_path)
    parquet_error_path = output_root / "parquet_error.txt"
    if parquet_error:
        _atomic_write_text(
            parquet_error_path,
            "Parquet generation failed; prediction_boxes.csv and summary.md "
            f"remain valid.\n{parquet_error}\n",
        )
    elif parquet_error_path.exists():
        _atomic_write_text(
            parquet_error_path,
            "Resolved: prediction_boxes.parquet was generated successfully.\n",
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["shuffled_control"] = diagnostic
    _atomic_write_text(
        output_root / "summary.md",
        render_summary(
            records, metadata, settings, parquet_error=parquet_error),
    )
    _atomic_write_json(metadata_path, metadata)
    print(
        f"[box-audit] rebuilt shuffled control boxes={len(records)} "
        f"csv={csv_path}",
        flush=True,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    if args.rebuild_shuffled_only:
        return rebuild_existing_shuffled_outputs(args.output_root)

    from mmcv import Config
    from mmdet.apis import inference_detector, init_detector

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    cfg = Config.fromfile(str(config_path))
    data_root = (
        _resolve_project_path(args.data_root)
        if args.data_root
        else Path(str(cfg.get("data_root"))).expanduser().resolve()
    )
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)

    selected: dict[str, list[str]] = {}
    image_indices: dict[str, dict[str, Path]] = {}
    annotation_dirs: dict[str, Path] = {}
    for split in args.splits:
        annotation_dir = data_root / split / "annfiles"
        image_dir = data_root / "corruptions" / args.corruption / split / "images"
        if not annotation_dir.is_dir():
            raise FileNotFoundError(annotation_dir)
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        stems = sorted(path.stem for path in annotation_dir.glob("*.txt"))
        limit = args.num_train if split == "train" else args.num_val
        selected[split] = select_split_stems(
            stems, split=split, limit=limit, seed=args.seed)
        image_indices[split] = index_images(image_dir)
        missing = [stem for stem in selected[split] if stem not in image_indices[split]]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} sampled {split} images are missing; first={missing[0]}")
        annotation_dirs[split] = annotation_dir

    settings = ScoreSettings.from_environment(args.score_thr)
    parameters: dict[str, Any] = {
        "config": str(config_path),
        "checkpoint": _checkpoint_identity(checkpoint_path),
        "data_root": str(data_root),
        "corruption": args.corruption,
        "splits": list(args.splits),
        "selected_stems": selected,
        "seed": args.seed,
        "train_pool_seed": TRAIN_POOL_SEED,
        "train_pool_size": TRAIN_POOL_SIZE,
        "min_det_score": args.min_det_score,
        "max_boxes_per_image": args.max_boxes_per_image,
        "cga_chunk_size": args.cga_chunk_size,
        "score_settings": dataclasses.asdict(settings),
        "shuffled_control": {
            "audit_scope": "all merged boxes within each split",
            "online_scope": "all images in the current inference batch",
            "strata": "detector_label+fixed_detector_score_bin",
            "score_bins": list(CGA_SHUFFLE_SCORE_BINS),
            "complete_probability_vector": True,
        },
        "sarclip_lora": os.environ.get("SARCLIP_LORA", ""),
    }
    output_root = args.output_root.expanduser().resolve()
    manifest = _prepare_manifest(output_root, parameters, resume=args.resume)
    run_signature = str(manifest["signature"])

    print(f"[box-audit] output={output_root}", flush=True)
    print(f"[box-audit] signature={run_signature}", flush=True)
    print(
        f"[box-audit] init raw teacher config={config_path} "
        f"checkpoint={checkpoint_path} device={args.device}",
        flush=True,
    )
    model = init_detector(str(config_path), str(checkpoint_path), device=args.device)
    model.CLASSES = RSAR_CLASSES
    cga = build_cga_from_environment()

    part_dir = output_root / "parts"
    status_dir = output_root / "status"
    completed_count = 0
    failed_count = 0
    part_paths: list[Path] = []
    for split in args.splits:
        part_path = part_dir / f"{split}.csv"
        status_path = status_dir / f"{split}.jsonl"
        part_paths.append(part_path)
        ensure_csv_header(part_path, BOX_FIELDNAMES)
        completed = load_completed_images(status_path, run_signature) if args.resume else set()
        total = len(selected[split])
        for position, stem in enumerate(selected[split], start=1):
            if stem in completed:
                completed_count += 1
                continue
            image_path = image_indices[split][stem]
            annotation_path = annotation_dirs[split] / f"{stem}.txt"
            started = time.time()
            try:
                raw_result = inference_without_cga(
                    model, str(image_path), inference_detector)
                boxes, scores, labels = flatten_detector_results(
                    raw_result,
                    min_score=args.min_det_score,
                    max_boxes=args.max_boxes_per_image,
                )
                gt_boxes, gt_labels = parse_dota_annotation(annotation_path)
                max_ious, matched_labels, categories = match_detections_to_gt(
                    boxes, labels, gt_boxes, gt_labels)
                probabilities = run_cga_in_chunks(
                    cga,
                    image_path,
                    boxes,
                    scores,
                    labels,
                    args.cga_chunk_size,
                )
                records = build_image_records(
                    run_signature=run_signature,
                    image_path=image_path,
                    split=split,
                    corruption=args.corruption,
                    detection_boxes=boxes,
                    detection_scores=scores,
                    detection_labels=labels,
                    max_ious=max_ious,
                    matched_gt_labels=matched_labels,
                    categories=categories,
                    probabilities=probabilities,
                    settings=settings,
                    shuffle_seed=args.seed,
                )
                rows_written = append_csv_rows(part_path, records, BOX_FIELDNAMES)
                append_status(
                    status_path,
                    {
                        "run_signature": run_signature,
                        "status": "completed",
                        "image_stem": stem,
                        "image": image_path.name,
                        "rows": rows_written,
                        "elapsed_seconds": time.time() - started,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                completed_count += 1
                if position == 1 or position % 25 == 0 or position == total:
                    print(
                        f"[box-audit] split={split} image={position}/{total} "
                        f"boxes={len(boxes)} completed={completed_count} "
                        f"failed={failed_count}",
                        flush=True,
                    )
            except Exception as exc:
                failed_count += 1
                append_status(
                    status_path,
                    {
                        "run_signature": run_signature,
                        "status": "failed",
                        "image_stem": stem,
                        "image": image_path.name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                        "elapsed_seconds": time.time() - started,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    },
                )
                print(
                    f"[box-audit][ERROR] split={split} image={image_path.name}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    final_csv = output_root / "prediction_boxes.csv"
    merged_count = merge_part_csvs(part_paths, final_csv, BOX_FIELDNAMES)
    records = read_csv_records(final_csv)
    if merged_count != len(records):
        raise RuntimeError("merged CSV row-count mismatch")
    shuffle_diagnostic = apply_global_shuffled_control(
        records, settings, seed=args.seed)
    write_csv_records_atomic(final_csv, records, BOX_FIELDNAMES)

    parquet_path = output_root / "prediction_boxes.parquet"
    parquet_error = attempt_parquet(final_csv, parquet_path)
    parquet_error_path = output_root / "parquet_error.txt"
    if parquet_error:
        message = (
            "Parquet generation failed; prediction_boxes.csv and summary.md remain "
            f"valid.\n{parquet_error}\n")
        _atomic_write_text(parquet_error_path, message)
        print(f"[box-audit][ERROR] {message.strip()}", file=sys.stderr, flush=True)
    elif parquet_error_path.exists():
        _atomic_write_text(
            parquet_error_path,
            "Resolved: prediction_boxes.parquet was generated successfully.\n",
        )

    metadata = {
        "signature": run_signature,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "corruption": args.corruption,
        "splits": list(args.splits),
        "completed_images": completed_count,
        "failed_images": failed_count,
        "min_det_score": args.min_det_score,
        "sarclip_lora": os.environ.get("SARCLIP_LORA", ""),
        "shuffled_control": shuffle_diagnostic,
    }
    _atomic_write_text(
        output_root / "summary.md",
        render_summary(
            records, metadata, settings, parquet_error=parquet_error),
    )
    _atomic_write_json(output_root / "run_metadata.json", metadata)
    print(
        f"[box-audit] finished images={completed_count} failed={failed_count} "
        f"boxes={merged_count} csv={final_csv}",
        flush=True,
    )
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
