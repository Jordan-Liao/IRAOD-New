#!/usr/bin/env python3
"""Paired statistical analysis for CGA research experiments.

The experimental unit is a seed block.  A candidate result is paired with the
baseline only when both runs use the same *actual* seed and the same GPU UUID.
Ambiguous duplicate runs and GPU-mismatched runs are reported and excluded;
they are never counted as additional observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats


DEFAULT_BOOTSTRAP_RESAMPLES = 100_000
DEFAULT_MONTE_CARLO_SAMPLES = 200_000
DEFAULT_RANDOM_SEED = 20260714


class StatisticsError(RuntimeError):
    """Base error for an invalid statistical input or operation."""


class PairingError(StatisticsError):
    """Raised when experiment rows cannot be interpreted for pairing."""


@dataclass(frozen=True)
class PairObservation:
    seed: int
    baseline: float
    candidate: float
    difference: float
    gpu_uuid: str
    baseline_work_dir: str
    candidate_work_dir: str


@dataclass(frozen=True)
class PairedSample:
    baseline_method: str
    candidate_method: str
    pairs: tuple[PairObservation, ...]
    missing_candidate_seeds: tuple[int, ...] = ()
    missing_baseline_seeds: tuple[int, ...] = ()
    duplicate_candidate_seeds: tuple[int, ...] = ()
    duplicate_baseline_seeds: tuple[int, ...] = ()
    gpu_mismatch_seeds: tuple[int, ...] = ()


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    return None


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise StatisticsError("Mean requires at least one value")
    return math.fsum(float(value) for value in values) / len(values)


def _sample_stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise StatisticsError(
            "Sample standard deviation requires at least two values"
        )
    mean = _mean(values)
    squared_deviations = math.fsum(
        (float(value) - mean) ** 2 for value in values
    )
    return math.sqrt(squared_deviations / (len(values) - 1))


def _row_seed(row: Mapping[str, Any]) -> int | None:
    seed = _as_int(row.get("actual_seed"))
    return seed if seed is not None else _as_int(row.get("seed"))


def _basic_admission_reasons(row: Mapping[str, Any]) -> list[str]:
    """Return reasons why a parsed experiment row is not statistically usable."""

    reasons: list[str] = []
    if _as_float(row.get("final_map")) is None:
        reasons.append("missing_or_invalid_final_map")
    status = str(
        row.get("completion_status") or row.get("status") or ""
    ).strip().lower()
    if status and status not in {"complete", "completed", "success", "succeeded"}:
        reasons.append("row_status_not_successful")
    if _as_bool(row.get("complete")) is False:
        reasons.append("row_marked_incomplete")
    if _as_bool(row.get("final_ema_found")) is False:
        reasons.append("final_ema_not_found")
    if not str(row.get("method", "")).strip():
        reasons.append("missing_method")
    if _row_seed(row) is None:
        reasons.append("missing_or_invalid_seed")
    return reasons


def _success_sentinel_reasons(row: Mapping[str, Any]) -> list[str]:
    """Validate the runner's successful ``run_result.json`` sentinel."""

    work_dir = str(row.get("work_dir", "")).strip()
    if not work_dir:
        return ["missing_work_dir_for_success_sentinel"]
    sentinel_path = Path(work_dir) / "run_result.json"
    try:
        payload = json.loads(sentinel_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ["missing_success_sentinel"]
    except (OSError, json.JSONDecodeError):
        return ["unreadable_or_invalid_success_sentinel"]
    if not isinstance(payload, dict):
        return ["invalid_success_sentinel_object"]

    reasons: list[str] = []
    if _as_bool(payload.get("success")) is not True:
        reasons.append("sentinel_success_not_true")
    if str(payload.get("status", "")).strip().lower() not in {
        "complete",
        "completed",
        "success",
        "succeeded",
    }:
        reasons.append("sentinel_status_not_successful")
    if _as_bool(payload.get("full_run")) is not True:
        reasons.append("sentinel_full_run_not_true")
    if payload.get("failure_kind") not in (None, ""):
        reasons.append("sentinel_has_failure_kind")

    row_seed = _row_seed(row)
    sentinel_seed = _as_int(payload.get("actual_seed"))
    if row_seed is None or sentinel_seed != row_seed:
        reasons.append("sentinel_seed_mismatch")
    if str(payload.get("method", "")).strip() != str(
        row.get("method", "")
    ).strip():
        reasons.append("sentinel_method_mismatch")
    if str(payload.get("gpu_uuid", "")).strip() != str(
        row.get("gpu_uuid", "")
    ).strip():
        reasons.append("sentinel_gpu_uuid_mismatch")
    return reasons


def audit_experiment_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_success_sentinel: bool = True,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Apply the default admission policy and return an auditable summary.

    Successful runner sentinels are required by default.  The opt-out exists
    for explicit legacy or pure-numeric analyses and is always recorded in the
    returned policy summary.
    """

    materialized = list(rows)
    admitted: list[Mapping[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    for index, row in enumerate(materialized):
        reasons = _basic_admission_reasons(row)
        if not reasons and require_success_sentinel:
            reasons.extend(_success_sentinel_reasons(row))
        if not reasons:
            admitted.append(row)
            continue
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        rejections.append(
            {
                "row_index": index,
                "method": str(row.get("method", "")).strip(),
                "seed": _row_seed(row),
                "work_dir": str(row.get("work_dir", "")).strip(),
                "reasons": reasons,
            }
        )

    policy = (
        "eligible parsed row plus matching successful run_result.json sentinel"
        if require_success_sentinel
        else "eligible parsed row only; success sentinel audit explicitly disabled"
    )
    return admitted, {
        "success_sentinel_required": require_success_sentinel,
        "policy": policy,
        "input_rows": len(materialized),
        "admitted_rows": len(admitted),
        "rejected_rows": len(rejections),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "rejections": rejections,
    }


def load_experiments(path: str | Path) -> list[dict[str, Any]]:
    """Load experiments from CSV or JSON Lines.

    When a directory is supplied, ``experiments.jsonl`` is preferred because
    it preserves native JSON types; ``experiments.csv`` is the fallback.
    """

    source = Path(path)
    if source.is_dir():
        jsonl = source / "experiments.jsonl"
        csv_path = source / "experiments.csv"
        if jsonl.is_file():
            source = jsonl
        elif csv_path.is_file():
            source = csv_path
        else:
            raise FileNotFoundError(
                f"No experiments.jsonl or experiments.csv in {source}"
            )
    if not source.is_file():
        raise FileNotFoundError(source)

    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PairingError(
                    f"Invalid JSON at {source}:{line_number}: {exc}"
                ) from exc
            if not isinstance(item, dict):
                raise PairingError(
                    f"Expected an object at {source}:{line_number}"
                )
            rows.append(item)
    return rows


def _eligible_row(row: Mapping[str, Any]) -> bool:
    return not _basic_admission_reasons(row)


def _normalized_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not _eligible_row(row):
        return None
    method = str(row.get("method", "")).strip()
    seed = _row_seed(row)
    if not method or seed is None:
        return None
    final_map = _as_float(row.get("final_map"))
    assert final_map is not None
    return {
        "method": method,
        "seed": seed,
        "final_map": final_map,
        "gpu_uuid": str(row.get("gpu_uuid", "")).strip(),
        "work_dir": str(row.get("work_dir", "")).strip(),
        "ended_at": str(row.get("ended_at", "")).strip(),
    }


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    work_dir = str(row.get("work_dir", ""))
    if work_dir:
        return ("work_dir", work_dir)
    return (
        "content",
        row.get("final_map"),
        row.get("gpu_uuid"),
        row.get("ended_at"),
    )


def _collapse_method_rows(
    rows: Iterable[Mapping[str, Any]], method: str
) -> tuple[dict[int, dict[str, Any]], tuple[int, ...]]:
    grouped: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
    conflicting_identities: set[int] = set()
    for raw in rows:
        row = _normalized_row(raw)
        if row is None or row["method"] != method:
            continue
        identity = _row_identity(row)
        by_identity = grouped.setdefault(row["seed"], {})
        previous = by_identity.get(identity)
        if previous is not None and (
            previous["final_map"] != row["final_map"]
            or previous["gpu_uuid"] != row["gpu_uuid"]
        ):
            conflicting_identities.add(row["seed"])
        by_identity[identity] = row

    unique: dict[int, dict[str, Any]] = {}
    duplicates: list[int] = []
    for seed, identities in grouped.items():
        values = list(identities.values())
        if len(values) == 1 and seed not in conflicting_identities:
            unique[seed] = values[0]
        else:
            duplicates.append(seed)
    return unique, tuple(sorted(duplicates))


def align_pairs(
    rows: Iterable[Mapping[str, Any]],
    candidate_method: str,
    baseline_method: str,
    *,
    require_success_sentinel: bool = True,
) -> PairedSample:
    """Align candidate and baseline by actual seed and GPU UUID.

    Exact duplicate records for the same work directory collapse to one row.
    Multiple distinct completed work directories for a method/seed are
    ambiguous and therefore excluded instead of being pseudo-replicated.
    """

    materialized = list(rows)
    if require_success_sentinel:
        materialized, _ = audit_experiment_rows(
            materialized, require_success_sentinel=True
        )
    baseline, duplicate_baseline = _collapse_method_rows(
        materialized, baseline_method
    )
    candidate, duplicate_candidate = _collapse_method_rows(
        materialized, candidate_method
    )

    baseline_seeds = set(baseline)
    candidate_seeds = set(candidate)
    common = baseline_seeds & candidate_seeds
    missing_candidate = sorted(
        baseline_seeds - candidate_seeds - set(duplicate_candidate)
    )
    missing_baseline = sorted(
        candidate_seeds - baseline_seeds - set(duplicate_baseline)
    )

    observations: list[PairObservation] = []
    gpu_mismatch: list[int] = []
    for seed in sorted(common):
        baseline_row = baseline[seed]
        candidate_row = candidate[seed]
        baseline_gpu = baseline_row["gpu_uuid"]
        candidate_gpu = candidate_row["gpu_uuid"]
        if not baseline_gpu or baseline_gpu != candidate_gpu:
            gpu_mismatch.append(seed)
            continue
        baseline_map = float(baseline_row["final_map"])
        candidate_map = float(candidate_row["final_map"])
        observations.append(
            PairObservation(
                seed=seed,
                baseline=baseline_map,
                candidate=candidate_map,
                difference=candidate_map - baseline_map,
                gpu_uuid=baseline_gpu,
                baseline_work_dir=baseline_row["work_dir"],
                candidate_work_dir=candidate_row["work_dir"],
            )
        )

    return PairedSample(
        baseline_method=baseline_method,
        candidate_method=candidate_method,
        pairs=tuple(observations),
        missing_candidate_seeds=tuple(missing_candidate),
        missing_baseline_seeds=tuple(missing_baseline),
        duplicate_candidate_seeds=duplicate_candidate,
        duplicate_baseline_seeds=duplicate_baseline,
        gpu_mismatch_seeds=tuple(gpu_mismatch),
    )


def bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise StatisticsError("Bootstrap CI requires at least two differences")
    if not np.all(np.isfinite(values)):
        raise StatisticsError("Bootstrap input contains a non-finite value")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if resamples <= 0:
        raise ValueError("resamples must be positive")

    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk_size = min(20_000, resamples)
    offset = 0
    while offset < resamples:
        size = min(chunk_size, resamples - offset)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(axis=1)
        offset += size
    alpha = 1.0 - confidence
    low, high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def paired_t_test(differences: Sequence[float]) -> dict[str, float | int]:
    """Two-sided paired t-test represented as a one-sample test of differences."""

    values = np.asarray(differences, dtype=np.float64)
    if len(values) < 2:
        raise StatisticsError("Paired t-test requires at least two pairs")
    if not np.all(np.isfinite(values)):
        raise StatisticsError("Paired t-test input contains a non-finite value")
    sample_std = float(np.std(values, ddof=1))
    if math.isclose(sample_std, 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise StatisticsError("Paired t-test is undefined for zero variance")
    result = stats.ttest_1samp(values, popmean=0.0, alternative="two-sided")
    return {
        "t": float(result.statistic),
        "df": int(len(values) - 1),
        "p": float(result.pvalue),
    }


def sign_flip_test(
    differences: Sequence[float],
    *,
    seed: int = DEFAULT_RANDOM_SEED,
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
) -> dict[str, Any]:
    """Two-sided paired sign-flip permutation test.

    All ``2**N`` sign assignments are enumerated for ``N <= 20``.  Larger
    samples use a deterministic Monte Carlo estimate with the standard
    plus-one correction.
    """

    values = np.asarray(differences, dtype=np.float64)
    n = len(values)
    if n < 1:
        raise StatisticsError("Sign-flip test requires at least one pair")
    if not np.all(np.isfinite(values)):
        raise StatisticsError("Permutation input contains a non-finite value")
    observed = abs(float(values.mean()))
    tolerance = max(1e-15, observed * 1e-12)

    if n <= 20:
        exceed = 0
        total = 1 << n
        for signs in product((-1.0, 1.0), repeat=n):
            statistic = abs(
                sum(value * sign for value, sign in zip(values, signs)) / n
            )
            if statistic >= observed - tolerance:
                exceed += 1
        return {
            "p": exceed / total,
            "mode": "exact",
            "assignments": total,
            "extreme_assignments": exceed,
        }

    if monte_carlo_samples <= 0:
        raise ValueError("monte_carlo_samples must be positive")
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = monte_carlo_samples
    while remaining:
        size = min(20_000, remaining)
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8)
        signs = signs * 2 - 1
        statistics_values = np.abs((signs * values).mean(axis=1))
        exceed += int(np.count_nonzero(statistics_values >= observed - tolerance))
        remaining -= size
    return {
        "p": (exceed + 1) / (monte_carlo_samples + 1),
        "mode": "monte_carlo",
        "assignments": monte_carlo_samples,
        "extreme_assignments": exceed,
        "seed": seed,
    }


def exact_sign_permutation_test(
    differences: Sequence[float],
) -> dict[str, Any]:
    """Compatibility wrapper that requires the exact ``N <= 20`` procedure."""

    if len(differences) > 20:
        raise StatisticsError("Exact sign-flip enumeration is limited to N <= 20")
    return sign_flip_test(differences)


def cohens_dz(differences: Sequence[float]) -> float:
    values = [float(value) for value in differences]
    if len(values) < 2:
        raise StatisticsError("Cohen's dz requires at least two pairs")
    if not all(math.isfinite(value) for value in values):
        raise StatisticsError("Cohen's dz input contains a non-finite value")
    sample_std = _sample_stdev(values)
    if math.isclose(sample_std, 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise StatisticsError("Cohen's dz is undefined for zero variance")
    return _mean(values) / sample_std


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise error correction."""

    valid: list[tuple[str, float]] = []
    for label, raw in p_values.items():
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for {label!r}: {raw!r}")
        valid.append((label, value))
    ordered = sorted(valid, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for index, (label, value) in enumerate(ordered):
        running_max = max(running_max, (count - index) * value)
        adjusted[label] = min(1.0, running_max)
    return adjusted


def _paired_method_descriptive(
    method: str,
    values: Sequence[float],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Describe one method on exactly the seeds retained in a paired sample."""

    return {
        "method": method,
        "N": len(values),
        "seeds": list(seeds),
        "mean": _mean(values) if values else None,
        "sample_sd": _sample_stdev(values) if len(values) >= 2 else None,
    }


def summarize_comparison(
    paired: PairedSample,
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_RANDOM_SEED,
    permutation_seed: int = DEFAULT_RANDOM_SEED,
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
) -> dict[str, Any]:
    differences = [pair.difference for pair in paired.pairs]
    paired_seeds = [pair.seed for pair in paired.pairs]
    baseline_values = [pair.baseline for pair in paired.pairs]
    candidate_values = [pair.candidate for pair in paired.pairs]
    summary: dict[str, Any] = {
        "baseline_method": paired.baseline_method,
        "candidate_method": paired.candidate_method,
        "N": len(paired.pairs),
        "pairs": [asdict(pair) for pair in paired.pairs],
        "paired_seeds": paired_seeds,
        "baseline_descriptive": _paired_method_descriptive(
            paired.baseline_method, baseline_values, paired_seeds
        ),
        "candidate_descriptive": _paired_method_descriptive(
            paired.candidate_method, candidate_values, paired_seeds
        ),
        "differences": differences,
        "missing_candidate_seeds": list(paired.missing_candidate_seeds),
        "missing_baseline_seeds": list(paired.missing_baseline_seeds),
        "duplicate_candidate_seeds": list(paired.duplicate_candidate_seeds),
        "duplicate_baseline_seeds": list(paired.duplicate_baseline_seeds),
        "gpu_mismatch_seeds": list(paired.gpu_mismatch_seeds),
        "mean_difference": None,
        "sample_std_difference": None,
        "standard_error": None,
        "bootstrap_ci_95": None,
        "paired_t": None,
        "paired_t_na_reason": None,
        "permutation_test": None,
        "cohens_dz": None,
        "cohens_dz_na_reason": None,
        "paired_t_p_holm": None,
        "permutation_p_holm": None,
    }

    if differences:
        summary["mean_difference"] = _mean(differences)
    if len(differences) < 2:
        summary["status"] = "insufficient_n"
        summary["reason"] = "At least two valid same-seed, same-GPU pairs are required"
        summary["paired_t_na_reason"] = "Paired t-test requires at least two pairs"
        summary["cohens_dz_na_reason"] = "Cohen's dz requires at least two pairs"
        return summary

    mean_difference = _mean(differences)
    sample_std = _sample_stdev(differences)
    summary["mean_difference"] = mean_difference
    summary["sample_std_difference"] = sample_std
    summary["standard_error"] = sample_std / math.sqrt(len(differences))
    summary["bootstrap_ci_95"] = list(
        bootstrap_mean_ci(
            differences,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    )
    summary["permutation_test"] = sign_flip_test(
        differences,
        seed=permutation_seed,
        monte_carlo_samples=monte_carlo_samples,
    )
    if math.isclose(sample_std, 0.0, rel_tol=0.0, abs_tol=1e-15):
        summary["sample_std_difference"] = 0.0
        summary["standard_error"] = 0.0
        summary["status"] = "zero_variance"
        summary["paired_t_na_reason"] = (
            "Paired t-test is undefined because paired differences have zero "
            "sample variance"
        )
        summary["cohens_dz_na_reason"] = (
            "Cohen's dz is undefined because paired differences have zero "
            "sample variance"
        )
        summary["reason"] = (
            "Paired t-test and Cohen's dz are NA for zero variance; the paired "
            "bootstrap CI and exact sign-flip test remain defined"
        )
        return summary

    summary["paired_t"] = paired_t_test(differences)
    summary["cohens_dz"] = cohens_dz(differences)
    summary["status"] = "ok"
    return summary


def analyze_comparisons(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline_method: str,
    candidate_methods: Sequence[str],
    require_success_sentinel: bool = True,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_RANDOM_SEED,
    permutation_seed: int = DEFAULT_RANDOM_SEED,
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
) -> dict[str, Any]:
    materialized = list(rows)
    admitted_rows, admission_audit = audit_experiment_rows(
        materialized,
        require_success_sentinel=require_success_sentinel,
    )
    candidates = list(dict.fromkeys(candidate_methods))
    if baseline_method in candidates:
        candidates.remove(baseline_method)
    comparisons = [
        summarize_comparison(
            align_pairs(
                admitted_rows,
                candidate,
                baseline_method,
                require_success_sentinel=False,
            ),
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            permutation_seed=permutation_seed,
            monte_carlo_samples=monte_carlo_samples,
        )
        for candidate in candidates
    ]

    t_values = {
        item["candidate_method"]: item["paired_t"]["p"]
        for item in comparisons
        if item["paired_t"] is not None
    }
    permutation_values = {
        item["candidate_method"]: item["permutation_test"]["p"]
        for item in comparisons
        if item["permutation_test"] is not None
    }
    adjusted_t = holm_adjust(t_values) if t_values else {}
    adjusted_permutation = (
        holm_adjust(permutation_values) if permutation_values else {}
    )
    for item in comparisons:
        candidate = item["candidate_method"]
        item["paired_t_p_holm"] = adjusted_t.get(candidate)
        item["permutation_p_holm"] = adjusted_permutation.get(candidate)

    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_method": baseline_method,
        "candidate_methods": candidates,
        "settings": {
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed": bootstrap_seed,
            "permutation_seed": permutation_seed,
            "monte_carlo_samples": monte_carlo_samples,
            "pairing": "same actual_seed and identical non-empty gpu_uuid",
            "difference": "candidate final_map - baseline final_map",
            "holm_family": (
                "all candidate methods supplied in this analyze_comparisons call, "
                "separately for each available test statistic"
            ),
        },
        "admission_audit": admission_audit,
        "comparisons": comparisons,
    }


def _format_number(value: Any, digits: int = 8) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}g}"


def render_markdown(report: Mapping[str, Any]) -> str:
    admission = report.get("admission_audit", {})
    lines = [
        "# CGA paired statistical report",
        "",
        f"- Baseline: `{report['baseline_method']}`",
        "- Pairing unit: identical actual seed and GPU UUID",
        "- Difference direction: candidate − baseline",
        "- 95% CI: paired percentile bootstrap with a fixed RNG seed",
        "- Permutation test: exact sign flips for N ≤ 20; deterministic Monte Carlo otherwise",
        (
            "- Success-sentinel admission audit: "
            + ("required" if admission.get("success_sentinel_required") else "disabled")
        ),
        (
            "- Admission rows: "
            f"{admission.get('admitted_rows', 'NA')}/{admission.get('input_rows', 'NA')} "
            f"admitted; {admission.get('rejected_rows', 'NA')} rejected"
        ),
        "",
    ]
    rejections = admission.get("rejections", [])
    if rejections:
        lines.extend(
            [
                "## Admission audit exclusions",
                "",
                "| row | method | seed | reasons | work directory |",
                "|---:|---|---:|---|---|",
            ]
        )
        for rejection in rejections:
            lines.append(
                "| {row} | `{method}` | {seed} | {reasons} | `{work_dir}` |".format(
                    row=rejection["row_index"],
                    method=rejection["method"],
                    seed=_format_number(rejection["seed"]),
                    reasons=", ".join(rejection["reasons"]),
                    work_dir=rejection["work_dir"],
                )
            )
        lines.append("")
    for comparison in report["comparisons"]:
        lines.extend(
            [
                f"## {comparison['candidate_method']} vs {comparison['baseline_method']}",
                "",
                f"- Status: `{comparison['status']}`",
                f"- N: {comparison['N']}",
                f"- Missing candidate seeds: {comparison['missing_candidate_seeds'] or 'none'}",
                f"- Missing baseline seeds: {comparison['missing_baseline_seeds'] or 'none'}",
                f"- Ambiguous candidate duplicates: {comparison['duplicate_candidate_seeds'] or 'none'}",
                f"- Ambiguous baseline duplicates: {comparison['duplicate_baseline_seeds'] or 'none'}",
                f"- GPU mismatches/exclusions: {comparison['gpu_mismatch_seeds'] or 'none'}",
                "",
                "| method (paired seeds only) | N | mean | sample SD |",
                "|---|---:|---:|---:|",
                "| `{method}` | {n} | {mean} | {sample_sd} |".format(
                    method=comparison["baseline_descriptive"]["method"],
                    n=comparison["baseline_descriptive"]["N"],
                    mean=_format_number(comparison["baseline_descriptive"]["mean"]),
                    sample_sd=_format_number(
                        comparison["baseline_descriptive"]["sample_sd"]
                    ),
                ),
                "| `{method}` | {n} | {mean} | {sample_sd} |".format(
                    method=comparison["candidate_descriptive"]["method"],
                    n=comparison["candidate_descriptive"]["N"],
                    mean=_format_number(comparison["candidate_descriptive"]["mean"]),
                    sample_sd=_format_number(
                        comparison["candidate_descriptive"]["sample_sd"]
                    ),
                ),
                "",
                "| seed | GPU UUID | baseline | candidate | difference |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for pair in comparison["pairs"]:
            lines.append(
                "| {seed} | `{gpu}` | {baseline} | {candidate} | {difference} |".format(
                    seed=pair["seed"],
                    gpu=pair["gpu_uuid"],
                    baseline=_format_number(pair["baseline"]),
                    candidate=_format_number(pair["candidate"]),
                    difference=_format_number(pair["difference"]),
                )
            )
        paired_t = comparison["paired_t"] or {}
        permutation = comparison["permutation_test"] or {}
        interval = comparison["bootstrap_ci_95"] or [None, None]
        lines.extend(
            [
                "",
                "| statistic | value |",
                "|---|---:|",
                f"| Mean paired difference | {_format_number(comparison['mean_difference'])} |",
                f"| Sample SD of differences | {_format_number(comparison['sample_std_difference'])} |",
                f"| Standard error | {_format_number(comparison['standard_error'])} |",
                f"| Bootstrap 95% CI | [{_format_number(interval[0])}, {_format_number(interval[1])}] |",
                f"| Paired t | {_format_number(paired_t.get('t'))} |",
                f"| df | {_format_number(paired_t.get('df'))} |",
                f"| Paired t p | {_format_number(paired_t.get('p'))} |",
                f"| Paired t Holm p | {_format_number(comparison['paired_t_p_holm'])} |",
                f"| Sign-flip p | {_format_number(permutation.get('p'))} |",
                f"| Sign-flip mode | {permutation.get('mode', 'NA')} |",
                f"| Sign-flip Holm p | {_format_number(comparison['permutation_p_holm'])} |",
                f"| Cohen's dz | {_format_number(comparison['cohens_dz'])} |",
                "",
            ]
        )
        if comparison.get("reason"):
            lines.extend([f"> {comparison['reason']}", ""])
        if comparison.get("paired_t_na_reason"):
            lines.extend(
                [f"> Paired t: NA — {comparison['paired_t_na_reason']}", ""]
            )
        if comparison.get("cohens_dz_na_reason"):
            lines.extend(
                [f"> Cohen's dz: NA — {comparison['cohens_dz_na_reason']}", ""]
            )
    return "\n".join(lines).rstrip() + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_report(
    report: Mapping[str, Any],
    markdown_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    _atomic_write(Path(markdown_path), render_markdown(report))
    if json_path is not None:
        _atomic_write(
            Path(json_path),
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


def _discover_methods(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    methods = {
        str(row.get("method", "")).strip()
        for row in rows
        if _normalized_row(row) is not None
    }
    methods.discard("")
    return sorted(methods)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute paired CGA experiment statistics"
    )
    parser.add_argument(
        "experiments", help="experiments.csv, experiments.jsonl, or research root"
    )
    parser.add_argument("--baseline", default="no_cga", help="Baseline method")
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate method; repeat for multiple comparisons",
    )
    parser.add_argument("--output", help="Markdown report output path")
    parser.add_argument("--json-output", help="Optional machine-readable JSON output")
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--permutation-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--monte-carlo-samples", type=int, default=DEFAULT_MONTE_CARLO_SAMPLES
    )
    parser.add_argument(
        "--no-success-sentinel-audit",
        action="store_true",
        help=(
            "Explicitly allow eligible table rows without verifying a matching "
            "successful work_dir/run_result.json sentinel"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    rows = load_experiments(args.experiments)
    candidates = args.candidate or [
        method for method in _discover_methods(rows) if method != args.baseline
    ]
    report = analyze_comparisons(
        rows,
        baseline_method=args.baseline,
        candidate_methods=candidates,
        require_success_sentinel=not args.no_success_sentinel_audit,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        permutation_seed=args.permutation_seed,
        monte_carlo_samples=args.monte_carlo_samples,
    )

    source = Path(args.experiments)
    output = Path(args.output) if args.output else (
        source if source.is_dir() else source.parent
    ) / "statistical_report.md"
    write_report(report, output, args.json_output)
    print(
        json.dumps(
            {
                "output": str(output),
                "json_output": args.json_output,
                "comparisons": len(report["comparisons"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
