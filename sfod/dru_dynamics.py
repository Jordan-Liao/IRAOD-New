"""DRU (ECCV 2024) training dynamics on Oriented R-CNN.

Official DRU is Dynamic Retraining-Updating + Historical Student Loss on
Deformable DETR. This module is the OBB port of those two mechanisms. It is
not Mean Teacher / method B: B uses a fixed EMA momentum and no historical
student term and never copies the teacher back onto the student.
"""
from __future__ import annotations

from math import isfinite
from typing import Any, Iterable


def adaptive_ema_momentum(
    base_momentum: float,
    student_loss: float,
    reference_loss: float,
    *,
    step: float = 0.02,
) -> float:
    """Dynamic updating: better student -> lower momentum (more teacher update)."""
    if not isfinite(student_loss) or not isfinite(reference_loss):
        return float(base_momentum)
    if reference_loss <= 0:
        return float(base_momentum)
    momentum = float(base_momentum)
    if student_loss < reference_loss:
        momentum = max(0.0, momentum - step)
    elif student_loss > reference_loss * 1.05:
        momentum = min(0.999, momentum + step)
    return momentum


def should_retrain_student(
    student_loss: float,
    reference_loss: float,
    *,
    ratio: float = 1.25,
) -> bool:
    """Dynamic retraining: student clearly worse than the teacher reference."""
    if not isfinite(student_loss) or not isfinite(reference_loss):
        return False
    if reference_loss <= 0:
        return False
    return bool(student_loss > reference_loss * ratio)


def _as_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (int, float)):
        return [float(values)]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        return [float(v) for v in values]
    return [float(values)]


def historical_student_consistency(current: Any, historical: Any) -> float:
    """MSE between current student scores and stored historical student scores."""
    cur = _as_float_list(current)
    hist = _as_float_list(historical)
    if not cur or not hist or len(cur) != len(hist):
        return 0.0
    return sum((a - b) ** 2 for a, b in zip(cur, hist)) / len(cur)


def unlabeled_loss_scalar(losses: dict) -> float:
    """Mean of unlabeled classification-like entries in a parsed loss dict."""
    values = []
    for key, val in losses.items():
        if "loss" not in key:
            continue
        if "unlabeled" not in key:
            continue
        if "bbox" in key:
            continue
        try:
            values.append(float(val.detach().item() if hasattr(val, "detach") else val))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return sum(values) / len(values)
