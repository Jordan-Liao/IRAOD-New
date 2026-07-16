#!/usr/bin/env python3
"""Parse completed CGA research runs into stable experiment tables.

The parser treats the final ``Epoch(val)`` record in timestamp logs as the EMA
evaluation.  When the companion ``.log.json`` contains a final ``mode=val``
record, its higher precision mAP is used after checking it against the text log.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


CLASSES = ("ship", "aircraft", "car", "tank", "bridge", "harbor")
TIMESTAMP_LOG_RE = re.compile(r"^\d{8}_\d{6}[^/]*\.log$")
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
VAL_RE = re.compile(
    rf"Epoch\(val\)\s*\[(?P<epoch>\d+)\]\[(?P<iteration>\d+)\]"
    rf".*?mAP:\s*(?P<map>{FLOAT_PATTERN})"
)
SEED_RE = re.compile(
    r"Set random seed to\s+(?P<seed>-?\d+)\s*,\s*"
    r"deterministic:\s*(?P<deterministic>True|False)"
)
PSEUDO_RE = re.compile(
    rf"Epoch\s*\[(?P<epoch>\d+)\]\s*"
    rf"\[(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\].*?"
    rf"pseudo_num:\s*(?P<pseudo>{FLOAT_PATTERN}).*?"
    rf"pseudo_num\(acc\):\s*(?P<accuracy>{FLOAT_PATTERN})"
)
LOG_TIMESTAMP_RE = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3,6})"
)
TEXT_JSON_MAP_TOLERANCE = 5e-5 + 1e-12
EXPECTED_FINAL_VAL_ITERATION = 4234
EXPECTED_CHECKPOINT_ITERATION = 4235
PSEUDO_PHASE_QUANTILES = {
    "early": 1.0 / 6.0,
    "middle": 1.0 / 2.0,
    "late": 5.0 / 6.0,
}
PSEUDO_PHASE_RULE = (
    "records_in_log_order after removing consecutive identical logger copies; "
    "nearest observed record to sequence quantiles 1/6,1/2,5/6 "
    "(centres of early/middle/late thirds)"
)
CGA_COUNT_FIELDS = (
    "calls",
    "total",
    "agree",
    "dropped",
    "blended",
    "boosted",
    "multiplied",
    "penalized",
    "threshold_dropped",
    "shuffled",
    "moved",
    "unmoved",
    "real_agree",
    "operative_agree",
)
CGA_OPTIONAL_DIAGNOSTIC_FIELDS = (
    "moved",
    "unmoved",
    "real_agree",
    "operative_agree",
)
CGA_OPERATION_FIELDS = (
    "dropped",
    "blended",
    "boosted",
    "multiplied",
    "penalized",
    "threshold_dropped",
    "shuffled",
)

EXPERIMENT_FIELDS = [
    "experiment_id",
    "run_id",
    "work_dir",
    "method",
    "requested_seed",
    "actual_seed",
    "seed",
    "seed_verified",
    "deterministic",
    "gpu_uuid",
    "physical_gpu_index",
    "gpu_name",
    "status",
    "complete",
    "final_map",
    "text_final_map",
    "final_epoch",
    "final_iteration",
    "final_eval_source",
    "final_log",
    "final_json_log",
    *[f"ap_{class_name}" for class_name in CLASSES],
    "pseudo_rule",
    "pseudo_record_count",
    "pseudo_early",
    "pseudo_acc_early",
    "pseudo_early_epoch",
    "pseudo_early_iteration",
    "pseudo_middle",
    "pseudo_acc_middle",
    "pseudo_middle_epoch",
    "pseudo_middle_iteration",
    "pseudo_late",
    "pseudo_acc_late",
    "pseudo_late_epoch",
    "pseudo_late_iteration",
    *[f"cga_{field}" for field in CGA_COUNT_FIELDS],
    "cga_window_count",
    "cga_coverage_scope",
    "cga_last_logged_call",
    "cga_mean_label_prob",
    "cga_argmax_json",
    "cga_label_prob_percentiles_json",
    "cga_per_class_json",
    "training_seconds",
    "gpu_seconds",
    "started_at",
    "ended_at",
    "environment_json",
]


class ResultParseError(ValueError):
    """A completed run is internally inconsistent."""


class IncompleteRunError(ResultParseError):
    """A work directory has no final EMA validation record."""


@dataclasses.dataclass(frozen=True)
class LogPair:
    text_path: Path
    json_path: Optional[Path]
    mtime_ns: int


@dataclasses.dataclass(frozen=True)
class ValMetric:
    epoch: int
    iteration: int
    text_map: float
    final_map: float
    source: str
    text_path: Path
    json_path: Optional[Path]


@dataclasses.dataclass(frozen=True)
class PseudoRecord:
    epoch: int
    iteration: int
    total_iterations: int
    pseudo_num: float
    pseudo_accuracy: float


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


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


def _optional_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ResultParseError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResultParseError(f"expected a JSON object in {path}")
    return payload


def discover_log_pairs(work_dir: Path) -> list[LogPair]:
    """Recursively find timestamp logs, excluding the tee log."""

    work_dir = Path(work_dir)
    pairs: list[LogPair] = []
    for text_path in work_dir.rglob("*.log"):
        if not text_path.is_file() or text_path.name == "run_train.log":
            continue
        if not TIMESTAMP_LOG_RE.fullmatch(text_path.name):
            continue
        json_path = Path(f"{text_path}.json")
        pairs.append(
            LogPair(
                text_path=text_path,
                json_path=json_path if json_path.is_file() else None,
                mtime_ns=text_path.stat().st_mtime_ns,
            )
        )
    return sorted(
        pairs,
        key=lambda pair: (
            pair.mtime_ns,
            pair.text_path.relative_to(work_dir).as_posix(),
        ),
    )


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(_read_text(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResultParseError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise ResultParseError(
                f"expected JSON object at {path}:{line_number}"
            )
        records.append(record)
    return records


def parse_final_val(text: str) -> Optional[tuple[int, int, float]]:
    matches = list(VAL_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return (
        int(match.group("epoch")),
        int(match.group("iteration")),
        float(match.group("map")),
    )


def _last_json_val(path: Path) -> Optional[dict[str, Any]]:
    records = parse_jsonl(path)
    values = [record for record in records if record.get("mode") == "val"]
    if not values:
        return None
    record = values[-1]
    if "mAP" not in record:
        raise ResultParseError(f"final val JSON record has no mAP: {path}")
    return record


def select_final_ema_val(pairs: Sequence[LogPair]) -> ValMetric:
    selected_pair: Optional[LogPair] = None
    selected_text_val: Optional[tuple[int, int, float]] = None
    for pair in pairs:
        parsed = parse_final_val(_read_text(pair.text_path))
        if parsed is not None:
            selected_pair = pair
            selected_text_val = parsed
    if selected_pair is None or selected_text_val is None:
        raise IncompleteRunError("no final EMA Epoch(val) in timestamp logs")

    epoch, iteration, text_map = selected_text_val
    final_map = text_map
    source = "text"
    json_path = selected_pair.json_path
    if json_path is not None:
        json_val = _last_json_val(json_path)
        if json_val is not None:
            json_map = float(json_val["mAP"])
            json_epoch = json_val.get("epoch")
            if json_epoch is not None and int(json_epoch) != epoch:
                raise ResultParseError(
                    f"final val epoch mismatch: text={epoch}, json={json_epoch}"
                )
            json_iteration = json_val.get("iter", json_val.get("iteration"))
            if json_iteration is not None and int(json_iteration) != iteration:
                raise ResultParseError(
                    "final val iteration mismatch: "
                    f"text={iteration}, json={json_iteration}"
                )
            if abs(json_map - text_map) > TEXT_JSON_MAP_TOLERANCE:
                raise ResultParseError(
                    "final mAP mismatch between text and JSON: "
                    f"text={text_map}, json={json_map}, log={selected_pair.text_path}"
                )
            final_map = json_map
            source = "jsonl"

    return ValMetric(
        epoch=epoch,
        iteration=iteration,
        text_map=text_map,
        final_map=final_map,
        source=source,
        text_path=selected_pair.text_path,
        json_path=json_path if source == "jsonl" else None,
    )


def parse_final_class_table(
    text: str, classes: Sequence[str] = CLASSES
) -> dict[str, float]:
    """Return the last complete class table in the requested class order."""

    expected = tuple(classes)
    current: list[tuple[str, float]] = []
    completed: list[dict[str, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 5 or columns[0] not in expected:
            continue
        try:
            ap = float(columns[4])
        except ValueError:
            continue
        class_name = columns[0]
        expected_index = len(current)
        if expected_index >= len(expected) or class_name != expected[expected_index]:
            current = []
            expected_index = 0
        if class_name == expected[expected_index]:
            current.append((class_name, ap))
        if len(current) == len(expected):
            completed.append(dict(current))
            current = []
    return completed[-1] if completed else {}


def parse_pseudo_records(text: str) -> list[PseudoRecord]:
    records: list[PseudoRecord] = []
    for match in PSEUDO_RE.finditer(text):
        record = PseudoRecord(
            epoch=int(match.group("epoch")),
            iteration=int(match.group("iteration")),
            total_iterations=int(match.group("total")),
            pseudo_num=float(match.group("pseudo")),
            pseudo_accuracy=float(match.group("accuracy")),
        )
        if records and record == records[-1]:
            continue
        records.append(record)
    return records


def summarize_pseudo(records: Sequence[PseudoRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rule": PSEUDO_PHASE_RULE,
        "record_count": len(records),
    }
    for phase, quantile in PSEUDO_PHASE_QUANTILES.items():
        if not records:
            record = None
        else:
            index = int(math.floor((len(records) - 1) * quantile + 0.5))
            record = records[index]
        summary[phase] = (
            None
            if record is None
            else {
                "pseudo_num": record.pseudo_num,
                "pseudo_accuracy": record.pseudo_accuracy,
                "epoch": record.epoch,
                "iteration": record.iteration,
                "total_iterations": record.total_iterations,
            }
        )
    return summary


def _numeric_diag_value(
    line: str, name: str, integer: bool = True
) -> Optional[Any]:
    pattern = rf"(?:^|[,\s]){re.escape(name)}=({FLOAT_PATTERN})"
    match = re.search(pattern, line)
    if match is None:
        return None
    return int(float(match.group(1))) if integer else float(match.group(1))


def _canonical_cga_mode(environment: Mapping[str, Any]) -> str:
    mode = str(environment.get("CGA_FILTER_MODE", "") or "").strip().lower()
    if mode in ("blend", "rescore"):
        return "legacy"
    return mode


def _excluded_class_ids(environment: Mapping[str, Any]) -> set[int]:
    value = environment.get("CGA_EXCLUDE_IDS")
    if value is None or str(value).strip() == "":
        return set()
    try:
        return {
            int(item.strip())
            for item in str(value).split(",")
            if item.strip()
        }
    except ValueError as error:
        raise ResultParseError(
            f"invalid CGA_EXCLUDE_IDS={value!r} in recorded environment"
        ) from error


def _validate_cga_aggregate(
    totals: Mapping[str, Any],
    raw_per_class: Mapping[str, Mapping[str, int]],
    argmax_counts: Mapping[str, int],
) -> None:
    total = int(totals["total"])
    agree = int(totals["agree"])
    dropped = int(totals["dropped"])
    if total < 0 or agree < 0 or agree > total:
        raise ResultParseError(
            f"invalid CGA aggregate counts: total={total}, agree={agree}"
        )
    for field in CGA_COUNT_FIELDS:
        value = totals[field]
        if value is not None and int(value) < 0:
            raise ResultParseError(
                f"negative CGA aggregate {field}={value}"
            )

    class_total = sum(bucket["total"] for bucket in raw_per_class.values())
    class_agree = sum(bucket["agree"] for bucket in raw_per_class.values())
    class_drop = sum(bucket["hard_drop_count"] for bucket in raw_per_class.values())
    if class_total != total:
        raise ResultParseError(
            "CGA detector-class total invariant failed: "
            f"per_class={class_total}, aggregate={total}"
        )
    if class_agree != agree:
        raise ResultParseError(
            "CGA detector-class agreement invariant failed: "
            f"per_class={class_agree}, aggregate={agree}"
        )
    if class_drop != dropped:
        raise ResultParseError(
            "CGA detector-class hard-drop invariant failed: "
            f"per_class={class_drop}, aggregate={dropped}"
        )
    if sum(argmax_counts.values()) != total:
        raise ResultParseError(
            "CGA argmax total invariant failed: "
            f"argmax={sum(argmax_counts.values())}, aggregate={total}"
        )

    operative_agree = totals.get("operative_agree")
    if operative_agree is not None and int(operative_agree) != agree:
        raise ResultParseError(
            "CGA operative agreement invariant failed: "
            f"operative={operative_agree}, aggregate agree={agree}"
        )


def _derive_cga_per_class(
    totals: Mapping[str, Any],
    raw_per_class: Mapping[str, Mapping[str, int]],
    environment: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    mode = _canonical_cga_mode(environment)
    excluded_ids = _excluded_class_ids(environment)

    moved = totals.get("moved")
    unmoved = totals.get("unmoved")
    real_agree = totals.get("real_agree")
    operative_agree = totals.get("operative_agree")
    if mode == "shuffled_legacy":
        if moved is not None and unmoved is not None:
            if int(moved) + int(unmoved) != int(totals["total"]):
                raise ResultParseError(
                    "shuffled CGA movement invariant failed: "
                    "moved + unmoved != total"
                )
        if real_agree is not None and operative_agree is not None:
            if int(real_agree) != int(operative_agree):
                raise ResultParseError(
                    "shuffled CGA agreement-preservation invariant failed: "
                    "real_agree != operative_agree"
                )
    else:
        if moved not in (None, 0) or unmoved not in (None, 0):
            raise ResultParseError(
                f"non-shuffled CGA mode {mode!r} logged movement counts"
            )
        if real_agree is not None and int(real_agree) != int(totals["agree"]):
            raise ResultParseError(
                f"non-shuffled CGA mode {mode!r} has real_agree != agree"
            )

    allowed_fields: set[str]
    strategy: str
    operation_kind: str
    aggregate_operation_count: int
    recoverable = True
    if mode in ("legacy", "adaptive_blend"):
        strategy = "disagreement"
        operation_kind = "blend"
        allowed_fields = {"blended"}
        aggregate_operation_count = int(totals["blended"])
    elif mode == "shuffled_legacy":
        strategy = "disagreement"
        operation_kind = "blend_shuffled_control"
        allowed_fields = {"blended", "shuffled"}
        aggregate_operation_count = int(totals["blended"])
        if int(totals["shuffled"]) != aggregate_operation_count:
            raise ResultParseError(
                "shuffled legacy invariant failed: shuffled != blended"
            )
    elif mode == "fixed_disagreement_penalty":
        strategy = "disagreement"
        operation_kind = "fixed_penalty"
        allowed_fields = {"penalized"}
        aggregate_operation_count = int(totals["penalized"])
    elif mode == "disagreement_threshold":
        strategy = "hard_drop"
        operation_kind = "hard_drop_below_disagreement_threshold"
        allowed_fields = {"dropped", "threshold_dropped"}
        aggregate_operation_count = int(totals["dropped"])
        if int(totals["threshold_dropped"]) != aggregate_operation_count:
            raise ResultParseError(
                "disagreement-threshold invariant failed: "
                "threshold_dropped != dropped"
            )
    elif mode in ("multiply", "prob_multiply"):
        strategy = "all_candidates"
        operation_kind = "multiply"
        allowed_fields = {"multiplied"}
        aggregate_operation_count = int(totals["multiplied"])
    elif mode in ("disagree_gate", "gate", "veto_soft"):
        strategy = "disagreement"
        operation_kind = "hard_drop_or_blend"
        allowed_fields = {"dropped", "blended"}
        aggregate_operation_count = int(totals["dropped"]) + int(
            totals["blended"]
        )
    elif mode in ("agree_gate", "strict_gate", "prob_gate", "label_prob_gate"):
        strategy = "hard_drop"
        operation_kind = "hard_drop"
        allowed_fields = {"dropped"}
        aggregate_operation_count = int(totals["dropped"])
    elif mode == "evidence_veto":
        allowed_fields = {"dropped", "blended"}
        aggregate_operation_count = int(totals["dropped"]) + int(
            totals["blended"]
        )
        if int(totals["blended"]) == 0:
            strategy = "hard_drop"
            operation_kind = "hard_drop"
        else:
            strategy = "unrecoverable"
            operation_kind = "hard_drop_or_soft_penalty_unattributed"
            recoverable = False
    elif mode == "consensus_boost":
        allowed_fields = {"boosted", "blended"}
        aggregate_operation_count = int(totals["boosted"]) + int(
            totals["blended"]
        )
        if aggregate_operation_count == 0:
            strategy = "zero"
            operation_kind = "boost_or_blend"
        else:
            strategy = "unrecoverable"
            operation_kind = "boost_or_blend_unattributed"
            recoverable = False
    else:
        raise ResultParseError(
            "cannot derive per-class CGA operations without a supported "
            f"CGA_FILTER_MODE; got {mode!r}"
        )

    for field in CGA_OPERATION_FIELDS:
        if field not in allowed_fields and int(totals[field]) != 0:
            raise ResultParseError(
                f"CGA mode {mode!r} logged unexpected {field}={totals[field]}"
            )

    result: dict[str, dict[str, Any]] = {}
    recoverable_sum = 0
    for class_id, class_name in enumerate(CLASSES):
        raw = raw_per_class[class_name]
        total = int(raw["total"])
        agree = int(raw["agree"])
        hard_drop = int(raw["hard_drop_count"])
        if agree < 0 or agree > total or hard_drop < 0 or hard_drop > total:
            raise ResultParseError(
                f"invalid CGA per-class counts for {class_name}: {raw}"
            )
        disagreement = total - agree
        excluded = class_id in excluded_ids
        class_recoverable = recoverable
        class_operation_kind = operation_kind
        if excluded:
            if hard_drop != 0:
                raise ResultParseError(
                    f"excluded CGA class {class_name} has hard_drop_count={hard_drop}"
                )
            actual_operation_count: Optional[int] = 0
            class_operation_kind = "excluded"
            class_recoverable = True
        elif strategy == "disagreement":
            actual_operation_count = disagreement
        elif strategy == "all_candidates":
            actual_operation_count = total
        elif strategy == "hard_drop":
            actual_operation_count = hard_drop
        elif strategy == "zero":
            actual_operation_count = 0
        else:
            actual_operation_count = None

        if actual_operation_count is not None:
            recoverable_sum += actual_operation_count
        result[class_name] = {
            "total": total,
            "agree": agree,
            "disagreement_candidates": disagreement,
            "hard_drop_count": hard_drop,
            "actual_operation_count": actual_operation_count,
            "operation_kind": class_operation_kind,
            "excluded_from_cga": excluded,
            "actual_operation_recoverable": class_recoverable,
        }

    if recoverable and recoverable_sum != aggregate_operation_count:
        raise ResultParseError(
            f"CGA mode {mode!r} operation invariant failed: "
            f"per_class={recoverable_sum}, aggregate={aggregate_operation_count}"
        )
    return result


def parse_cga_windows(
    text: str,
    environment: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    totals = {field: 0 for field in CGA_COUNT_FIELDS}
    field_presence = {field: 0 for field in CGA_COUNT_FIELDS}
    weighted_label_prob = 0.0
    weighted_count = 0
    window_count = 0
    per_class: dict[str, dict[str, int]] = {}
    argmax_counts: dict[str, int] = {}
    percentile_windows: list[dict[str, float]] = []
    last_filter_call: Optional[int] = None

    detector_re = re.compile(
        r"(?P<class>[^:;,\s]+):n=(?P<total>\d+),"
        r"agree=(?P<agree>\d+),drop=(?P<drop>\d+)"
    )
    for raw_line in text.splitlines():
        if "[CGA] filter " in raw_line:
            filter_line = raw_line.split("[CGA] filter ", 1)[1]
            filter_call = _numeric_diag_value(filter_line, "calls")
            if filter_call is not None:
                last_filter_call = int(filter_call)
        if "[CGA] diag_window" not in raw_line:
            continue
        line = raw_line.split("[CGA] diag_window", 1)[1]
        window_count += 1
        values: dict[str, int] = {}
        for field in CGA_COUNT_FIELDS:
            value = _numeric_diag_value(line, field)
            if value is None:
                values[field] = 0
            else:
                values[field] = int(value)
                field_presence[field] += 1
        for field, value in values.items():
            totals[field] += value
        mean_label_prob = _numeric_diag_value(
            line, "mean_label_prob", integer=False
        )
        if mean_label_prob is not None and values["total"] > 0:
            weighted_label_prob += mean_label_prob * values["total"]
            weighted_count += values["total"]

        percentile_match = re.search(
            rf"label_prob_pct=min=(?P<min>{FLOAT_PATTERN}),"
            rf"p25=(?P<p25>{FLOAT_PATTERN}),"
            rf"p50=(?P<p50>{FLOAT_PATTERN}),"
            rf"p75=(?P<p75>{FLOAT_PATTERN}),"
            rf"max=(?P<max>{FLOAT_PATTERN})",
            line,
        )
        if percentile_match is not None:
            percentile_windows.append(
                {
                    name: float(percentile_match.group(name))
                    for name in ("min", "p25", "p50", "p75", "max")
                }
            )

        argmax_text = ""
        if "argmax=" in line:
            argmax_text = line.split("argmax=", 1)[1].split("detector=", 1)[0]
        for class_name, count in re.findall(r"([^:;,\s]+):(\d+)", argmax_text):
            argmax_counts[class_name] = argmax_counts.get(class_name, 0) + int(count)

        detector_text = line.split("detector=", 1)[1] if "detector=" in line else ""
        for match in detector_re.finditer(detector_text):
            class_name = match.group("class")
            if class_name not in CLASSES:
                raise ResultParseError(
                    f"unknown detector class in CGA diagnostics: {class_name!r}"
                )
            bucket = per_class.setdefault(
                class_name,
                {"total": 0, "agree": 0, "hard_drop_count": 0},
            )
            class_total = int(match.group("total"))
            class_agree = int(match.group("agree"))
            class_drop = int(match.group("drop"))
            bucket["total"] += class_total
            bucket["agree"] += class_agree
            bucket["hard_drop_count"] += class_drop

    if window_count:
        required_fields = set(CGA_COUNT_FIELDS) - set(
            CGA_OPTIONAL_DIAGNOSTIC_FIELDS
        )
        for field in required_fields:
            if field_presence[field] != window_count:
                raise ResultParseError(
                    f"CGA diag_window field {field!r} is present in "
                    f"{field_presence[field]}/{window_count} windows"
                )
        for field in CGA_OPTIONAL_DIAGNOSTIC_FIELDS:
            if field_presence[field] not in (0, window_count):
                raise ResultParseError(
                    f"optional CGA field {field!r} is present in "
                    f"{field_presence[field]}/{window_count} windows"
                )
            if field_presence[field] == 0:
                totals[field] = None

        for class_name in CLASSES:
            per_class.setdefault(
                class_name,
                {"total": 0, "agree": 0, "hard_drop_count": 0},
            )
        unknown_argmax = sorted(set(argmax_counts) - set(CLASSES))
        if unknown_argmax:
            raise ResultParseError(
                f"unknown argmax classes in CGA diagnostics: {unknown_argmax}"
            )
        _validate_cga_aggregate(totals, per_class, argmax_counts)
        per_class_result = _derive_cga_per_class(
            totals, per_class, environment or {}
        )
        if (
            last_filter_call is not None
            and last_filter_call != totals["calls"]
        ):
            raise ResultParseError(
                "CGA logged-prefix call invariant failed: "
                f"last filter call={last_filter_call}, "
                f"summed window calls={totals['calls']}"
            )
    else:
        for field in CGA_OPTIONAL_DIAGNOSTIC_FIELDS:
            totals[field] = None
        per_class_result = {}

    return {
        "window_count": window_count,
        **totals,
        "coverage_scope": "logged_prefix" if window_count else "not_logged",
        "last_logged_call": (
            last_filter_call
            if last_filter_call is not None
            else (totals["calls"] if window_count else None)
        ),
        "mean_label_prob": (
            weighted_label_prob / weighted_count if weighted_count else None
        ),
        "argmax": argmax_counts,
        "label_prob_percentile_windows": percentile_windows,
        "per_class": per_class_result,
    }


def parse_actual_seed(text: str) -> Optional[tuple[int, bool]]:
    matches = list(SEED_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group("seed")), match.group("deterministic") == "True"


def _parse_process_environment(path: Path) -> dict[str, str]:
    try:
        lines = _read_text(path).splitlines()
    except FileNotFoundError:
        return {}
    environment: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            environment[key] = value
    return environment


def _decode_json_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value) != "":
            return value
    return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None or str(value) == "":
        return None
    return int(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or str(value) == "":
        return None
    return float(value)


def _infer_seed_from_path(work_dir: Path) -> Optional[int]:
    for part in reversed(work_dir.parts):
        match = re.fullmatch(r"seed_(-?\d+)", part)
        if match:
            return int(match.group(1))
    return None


def _infer_method_from_path(work_dir: Path) -> str:
    parts = list(work_dir.parts)
    for index, part in enumerate(parts):
        if re.fullmatch(r"seed_-?\d+", part) and index > 0:
            return parts[index - 1]
    return work_dir.name


def _log_duration_seconds(text: str) -> Optional[float]:
    timestamps = []
    for match in LOG_TIMESTAMP_RE.finditer(text):
        value = match.group("timestamp")
        fmt = "%Y-%m-%d %H:%M:%S,%f"
        try:
            timestamps.append(dt.datetime.strptime(value, fmt))
        except ValueError:
            continue
    if len(timestamps) < 2:
        return None
    return max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())


def _experiment_id(work_dir: Path) -> str:
    digest = hashlib.sha256(str(work_dir.resolve()).encode("utf-8")).hexdigest()
    return digest[:20]


def _validate_completion_sentinel(
    work_dir: Path,
    run_result: Mapping[str, Any],
    final_val: ValMetric,
    actual_seed: int,
) -> None:
    if run_result.get("status") != "completed":
        raise IncompleteRunError(
            f"run_result status is not completed in {work_dir}: "
            f"{run_result.get('status')!r}"
        )
    if run_result.get("success") is not True:
        raise ResultParseError(
            f"run_result success is not true in {work_dir}"
        )
    exit_code = run_result.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code != 0:
        raise ResultParseError(
            f"run_result exit_code is not integer zero in {work_dir}: {exit_code!r}"
        )

    if "actual_seed" not in run_result:
        raise ResultParseError(
            f"run_result has no actual_seed in {work_dir}"
        )
    manifest_actual_seed = _optional_int(run_result.get("actual_seed"))
    if manifest_actual_seed != actual_seed:
        raise ResultParseError(
            f"run_result actual_seed={manifest_actual_seed} disagrees with "
            f"log actual_seed={actual_seed}"
        )

    if "final_val_iteration" not in run_result:
        raise ResultParseError(
            f"run_result has no final_val_iteration in {work_dir}"
        )
    manifest_iteration = _optional_int(run_result.get("final_val_iteration"))
    if manifest_iteration != EXPECTED_FINAL_VAL_ITERATION:
        raise ResultParseError(
            "run_result final_val_iteration is not the required full validation "
            f"iteration {EXPECTED_FINAL_VAL_ITERATION}: {manifest_iteration}"
        )
    if final_val.iteration != EXPECTED_FINAL_VAL_ITERATION:
        raise ResultParseError(
            "selected final validation is not at the required iteration "
            f"{EXPECTED_FINAL_VAL_ITERATION}: {final_val.iteration}"
        )

    progress = run_result.get("progress")
    if not isinstance(progress, dict):
        raise ResultParseError(
            f"run_result has no progress object in {work_dir}"
        )
    progress_total = _optional_int(progress.get("total"))
    if progress_total != EXPECTED_FINAL_VAL_ITERATION:
        raise ResultParseError(
            "run_result progress.total is not the required total "
            f"{EXPECTED_FINAL_VAL_ITERATION}: {progress_total}"
        )

    manifest_epoch = _optional_int(run_result.get("final_val_epoch"))
    if manifest_epoch is not None and manifest_epoch != final_val.epoch:
        raise ResultParseError(
            "run_result final_val_epoch disagrees with the selected log: "
            f"manifest={manifest_epoch}, log={final_val.epoch}"
        )
    manifest_log = run_result.get("final_val_log")
    if manifest_log:
        manifest_log_path = Path(str(manifest_log)).expanduser()
        if not manifest_log_path.is_absolute():
            manifest_log_path = work_dir / manifest_log_path
        if manifest_log_path.resolve() != final_val.text_path.resolve():
            raise ResultParseError(
                "run_result final_val_log disagrees with selected timestamp log: "
                f"manifest={manifest_log_path}, selected={final_val.text_path}"
            )

    for filename in (
        f"iter_{EXPECTED_CHECKPOINT_ITERATION}.pth",
        f"iter_{EXPECTED_CHECKPOINT_ITERATION}_ema.pth",
    ):
        if not (work_dir / filename).is_file():
            raise IncompleteRunError(
                f"missing required completion checkpoint {work_dir / filename}"
            )


def parse_work_dir(
    work_dir: Path,
    registry_row: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    work_dir = Path(work_dir).resolve()
    pairs = discover_log_pairs(work_dir)
    if not pairs:
        raise IncompleteRunError(f"no timestamp logs under {work_dir}")
    final_val = select_final_ema_val(pairs)
    selected_text = _read_text(final_val.text_path)
    run_train_path = work_dir / "run_train.log"
    run_train_text = _read_text(run_train_path) if run_train_path.is_file() else ""

    run_result_path = work_dir / "run_result.json"
    if not run_result_path.is_file():
        raise IncompleteRunError(
            f"no run_result.json completion sentinel in {work_dir}"
        )
    run_result = _optional_json(run_result_path)
    registry = dict(registry_row or {})
    requested_seed = _optional_int(
        _first_nonempty(
            registry.get("requested_seed"),
            registry.get("seed"),
            run_result.get("seed"),
            _infer_seed_from_path(work_dir),
        )
    )
    actual_seed_record = (
        parse_actual_seed(selected_text)
        or parse_actual_seed(run_train_text)
    )
    if actual_seed_record is None:
        raise ResultParseError(f"actual seed is not logged in {work_dir}")
    actual_seed, deterministic = actual_seed_record
    if requested_seed is not None and requested_seed != actual_seed:
        raise ResultParseError(
            f"seed mismatch for {work_dir}: requested={requested_seed}, "
            f"actual={actual_seed}"
        )
    _validate_completion_sentinel(
        work_dir, run_result, final_val, actual_seed
    )

    class_ap = parse_final_class_table(selected_text)
    if tuple(class_ap) != CLASSES:
        raise ResultParseError(
            f"no complete final six-class AP table in {final_val.text_path}"
        )
    pseudo = summarize_pseudo(parse_pseudo_records(run_train_text))

    environment: dict[str, Any] = {}
    environment.update(_decode_json_mapping(registry.get("environment_json")))
    environment.update(_decode_json_mapping(run_result.get("environment")))
    environment.update(_optional_json(work_dir / "launch_environment.json"))
    environment.update(_parse_process_environment(work_dir / "process_environment.txt"))
    cga = parse_cga_windows(selected_text, environment)

    started_at = _optional_float(
        _first_nonempty(registry.get("started_at"), run_result.get("started_at"))
    )
    ended_at = _optional_float(
        _first_nonempty(registry.get("ended_at"), run_result.get("ended_at"))
    )
    gpu_seconds = _optional_float(
        _first_nonempty(registry.get("gpu_seconds"), run_result.get("gpu_seconds"))
    )
    duration = (
        max(0.0, ended_at - started_at)
        if started_at is not None and ended_at is not None
        else _log_duration_seconds(selected_text or run_train_text)
    )

    method = str(
        _first_nonempty(
            registry.get("method"), run_result.get("method"),
            _infer_method_from_path(work_dir),
        )
    )
    row: dict[str, Any] = {
        "experiment_id": _experiment_id(work_dir),
        "run_id": str(_first_nonempty(registry.get("run_id"), "") or ""),
        "work_dir": str(work_dir),
        "method": method,
        "requested_seed": requested_seed,
        "actual_seed": actual_seed,
        "seed": actual_seed,
        "seed_verified": True,
        "deterministic": deterministic,
        "gpu_uuid": str(
            _first_nonempty(registry.get("gpu_uuid"), run_result.get("gpu_uuid"), "")
            or ""
        ),
        "physical_gpu_index": _optional_int(
            _first_nonempty(
                registry.get("physical_gpu_index"), run_result.get("gpu_index")
            )
        ),
        "gpu_name": str(registry.get("gpu_name", "") or ""),
        "status": "completed",
        "complete": True,
        "final_map": final_val.final_map,
        "text_final_map": final_val.text_map,
        "final_epoch": final_val.epoch,
        "final_iteration": final_val.iteration,
        "final_eval_source": final_val.source,
        "final_log": str(final_val.text_path),
        "final_json_log": str(final_val.json_path or ""),
        "pseudo_rule": pseudo["rule"],
        "pseudo_record_count": pseudo["record_count"],
        "cga_window_count": cga["window_count"],
        "cga_coverage_scope": cga["coverage_scope"],
        "cga_last_logged_call": cga["last_logged_call"],
        "cga_mean_label_prob": cga["mean_label_prob"],
        "cga_argmax_json": json.dumps(
            cga["argmax"], ensure_ascii=False, sort_keys=True
        ),
        "cga_label_prob_percentiles_json": json.dumps(
            cga["label_prob_percentile_windows"],
            ensure_ascii=False,
            sort_keys=True,
        ),
        "cga_per_class_json": json.dumps(
            cga["per_class"], ensure_ascii=False, sort_keys=True
        ),
        "training_seconds": duration,
        "gpu_seconds": gpu_seconds,
        "started_at": started_at,
        "ended_at": ended_at,
        "environment_json": json.dumps(
            environment, ensure_ascii=False, sort_keys=True
        ),
    }
    for class_name in CLASSES:
        row[f"ap_{class_name}"] = class_ap.get(class_name)
    for phase in PSEUDO_PHASE_QUANTILES:
        record = pseudo[phase]
        row[f"pseudo_{phase}"] = None if record is None else record["pseudo_num"]
        row[f"pseudo_acc_{phase}"] = (
            None if record is None else record["pseudo_accuracy"]
        )
        row[f"pseudo_{phase}_epoch"] = None if record is None else record["epoch"]
        row[f"pseudo_{phase}_iteration"] = (
            None if record is None else record["iteration"]
        )
    for field in CGA_COUNT_FIELDS:
        row[f"cga_{field}"] = cga[field]
    return {field: row.get(field) for field in EXPERIMENT_FIELDS}


def load_registry(path: Optional[Path]) -> dict[str, dict[str, str]]:
    if path is None or not Path(path).is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            work_dir = row.get("work_dir", "")
            if not work_dir:
                continue
            key = str(Path(work_dir).expanduser().resolve())
            previous = rows.get(key)
            if previous is None or int(row.get("attempt", "0") or 0) >= int(
                previous.get("attempt", "0") or 0
            ):
                rows[key] = dict(row)
    return rows


def discover_run_dirs(research_root: Path) -> list[Path]:
    research_root = Path(research_root).resolve()
    runs_root = research_root / "runs"
    if not runs_root.is_dir():
        return []
    runs_root = runs_root.resolve()
    directories = {path.parent.resolve() for path in runs_root.rglob("run_result.json")}
    for path in runs_root.rglob("*.log"):
        if path.name == "run_train.log" or not TIMESTAMP_LOG_RE.fullmatch(path.name):
            continue
        owner = path.parent.resolve()
        for candidate in (owner, *owner.parents):
            if candidate == runs_root.parent:
                break
            if ((candidate / "run_result.json").is_file()
                    or (candidate / "run_train.log").is_file()):
                owner = candidate
                break
            if candidate == runs_root:
                break
        directories.add(owner)
    return sorted(directories, key=lambda path: path.as_posix())


def _load_existing_rows(csv_path: Path, jsonl_path: Path) -> list[dict[str, Any]]:
    if jsonl_path.is_file():
        return parse_jsonl(jsonl_path)
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in EXPERIMENT_FIELDS}


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return value


def upsert_experiments(
    output_root: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    replace_existing: bool = False,
) -> int:
    """Atomically write experiment rows keyed by resolved work_dir.

    A full research-root scan uses ``replace_existing`` so rows that are no
    longer in the formal ``runs/`` tree (for example smoke training) cannot
    survive from an older, broader scan.  A single-work-dir parse keeps normal
    upsert semantics.
    """

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "experiments.csv"
    jsonl_path = output_root / "experiments.jsonl"
    lock_path = output_root / ".experiments.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = [] if replace_existing else _load_existing_rows(csv_path, jsonl_path)
        by_work_dir: dict[str, dict[str, Any]] = {}
        for row in existing:
            work_dir = str(row.get("work_dir", "") or "")
            if work_dir:
                by_work_dir[str(Path(work_dir).resolve())] = _normalize_row(row)
        for row in rows:
            normalized = _normalize_row(row)
            work_dir = str(normalized.get("work_dir", "") or "")
            if not work_dir:
                raise ResultParseError("experiment row has no work_dir")
            key = str(Path(work_dir).resolve())
            normalized["work_dir"] = key
            by_work_dir[key] = normalized

        ordered = sorted(
            by_work_dir.values(),
            key=lambda row: (
                str(row.get("method") or ""),
                int(row.get("actual_seed") or -1),
                str(row.get("gpu_uuid") or ""),
                str(row.get("work_dir") or ""),
            ),
        )
        csv_buffer = io.StringIO(newline="")
        writer = csv.DictWriter(csv_buffer, fieldnames=EXPERIMENT_FIELDS)
        writer.writeheader()
        writer.writerows(
            {
                field: _csv_value(row.get(field))
                for field in EXPERIMENT_FIELDS
            }
            for row in ordered
        )
        jsonl_text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in ordered
        )
        _atomic_write_text(csv_path, csv_buffer.getvalue())
        _atomic_write_text(jsonl_path, jsonl_text)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return len(ordered)


def _infer_research_root(work_dir: Path) -> Path:
    current = Path(work_dir).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "gpu_job_registry.csv").exists() or candidate.name == "auto_research":
            return candidate
    return current.parent


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--work-dir", type=Path)
    source.add_argument("--research-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--registry", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail a research-root scan on the first incomplete/invalid run",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.work_dir is not None:
        work_dirs = [args.work_dir.resolve()]
        research_root = _infer_research_root(args.work_dir)
    else:
        research_root = args.research_root.resolve()
        work_dirs = discover_run_dirs(research_root)
    output_root = (args.output_root or research_root).resolve()
    registry_path = args.registry or (research_root / "gpu_job_registry.csv")
    registry = load_registry(registry_path)

    parsed_rows = []
    errors = []
    for work_dir in work_dirs:
        try:
            parsed_rows.append(
                parse_work_dir(work_dir, registry.get(str(work_dir.resolve())))
            )
        except ResultParseError as error:
            if args.strict or args.work_dir is not None:
                raise
            errors.append(f"{work_dir}: {error}")
    total_rows = upsert_experiments(
        output_root,
        parsed_rows,
        replace_existing=args.research_root is not None,
    )
    print(
        json.dumps(
            {
                "parsed": len(parsed_rows),
                "stored": total_rows,
                "skipped": len(errors),
                "errors": errors,
                "csv": str(output_root / "experiments.csv"),
                "jsonl": str(output_root / "experiments.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResultParseError as error:
        print(f"parse_results: {error}", file=sys.stderr)
        raise SystemExit(2)
