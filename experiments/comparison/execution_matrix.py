"""Parseable execution matrix for the approved Source-Free OBB comparison.

Main table: A–F plus six external SFOD ports. DRU is required and is not B.
Defensive extras are listed so tests can assert they are out of the queue.
"""
from __future__ import annotations

from tools.dataset.generate_dior_corruptions import remaining_dior_c_corruptions

# Own methods (reuse hash-matched valid cells).
OWN_METHODS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

# External SFOD ports. DRU is required; it is not Mean Teacher / method B.
EXTERNAL_METHODS: tuple[str, ...] = (
    "IRG",
    "LPLD",
    "SFUT",
    "DRU",
    "AASFOD",
    "SFYOLO",
)

MAIN_TABLE_METHODS: tuple[str, ...] = OWN_METHODS + EXTERNAL_METHODS

# DRU-OBB detector type. Must not be UnbiasedTeacher (method B).
DRU_DETECTOR_TYPE = "DRUUnbiasedTeacher"
B_DETECTOR_TYPE = "UnbiasedTeacher"

REQUIRED_TEST_ROLE = "ema"
STUDENT_REQUIRED_TEST = False

# Out of the live queue (already-finished cells may stay in the archive).
DEFENSIVE_EXCLUDED: tuple[str, ...] = (
    "Student-required-TEST",
    "LoRA-CGA-oracle",
    "B_REG",
    "F_text_only",
    "F_veto_only",
    "ROI-expansion",
    "dior17-3seed",
)

NEW_DIOR_SEED = 42
DIOR_REMAINING_C = remaining_dior_c_corruptions()


def dru_is_method_b() -> bool:
    return DRU_DETECTOR_TYPE == B_DETECTOR_TYPE or "DRU" == "B"


def method_is_required(method_id: str) -> bool:
    return method_id in MAIN_TABLE_METHODS


def method_is_defensive(method_id: str) -> bool:
    return method_id in DEFENSIVE_EXCLUDED
