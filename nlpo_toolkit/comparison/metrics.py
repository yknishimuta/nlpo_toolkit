from __future__ import annotations

import math
from typing import Protocol

from .errors import ComparisonEngineError


class ZeroHandlingModeValue(Protocol):
    value: str


class ZeroHandlingValue(Protocol):
    mode: ZeroHandlingModeValue
    value: float


EPSILON = 1e-12


def _validate_count_total(*, count: float, total: float, label: str) -> None:
    if count < 0:
        raise ComparisonEngineError(f"{label} count must be >= 0")
    if total <= 0:
        raise ComparisonEngineError(f"{label} total must be > 0")
    if count > total:
        raise ComparisonEngineError(f"{label} count must not exceed total")


def normalized_rate(count: float, total: float, *, scale: float) -> float:
    _validate_count_total(count=count, total=total, label="normalized rate")
    if not math.isfinite(scale) or scale <= 0:
        raise ComparisonEngineError("scale must be a positive finite number")
    return count / total * scale


def _relative_for_ratio(
    *,
    count: float,
    total: float,
    zero_handling: ZeroHandlingValue,
    vocabulary_size: int,
) -> float:
    if zero_handling.mode.value == "zero_only":
        adjusted = count if count > 0 else zero_handling.value
        return adjusted / total

    denominator = total + zero_handling.value * vocabulary_size
    if denominator <= 0:
        return 0.0
    return (count + zero_handling.value) / denominator


def calculate_ratio(
    *,
    count_a: float,
    count_b: float,
    total_a: float,
    total_b: float,
    zero_handling: ZeroHandlingValue,
    vocabulary_size: int = 1,
) -> float:
    _validate_count_total(count=count_a, total=total_a, label="group_a")
    _validate_count_total(count=count_b, total=total_b, label="group_b")
    if vocabulary_size <= 0:
        raise ComparisonEngineError("vocabulary_size must be > 0")

    relative_a = _relative_for_ratio(
        count=count_a,
        total=total_a,
        zero_handling=zero_handling,
        vocabulary_size=vocabulary_size,
    )
    relative_b = _relative_for_ratio(
        count=count_b,
        total=total_b,
        zero_handling=zero_handling,
        vocabulary_size=vocabulary_size,
    )
    return relative_a / relative_b if relative_b else math.inf


def calculate_log_ratio(
    *,
    count_a: float,
    count_b: float,
    total_a: float,
    total_b: float,
    zero_handling: ZeroHandlingValue,
    vocabulary_size: int = 1,
) -> float:
    ratio = calculate_ratio(
        count_a=count_a,
        count_b=count_b,
        total_a=total_a,
        total_b=total_b,
        zero_handling=zero_handling,
        vocabulary_size=vocabulary_size,
    )
    return math.log2(ratio) if ratio > 0 else -math.inf


def calculate_log_likelihood(
    *,
    count_a: float,
    count_b: float,
    total_a: float,
    total_b: float,
) -> float:
    _validate_count_total(count=count_a, total=total_a, label="group_a")
    _validate_count_total(count=count_b, total=total_b, label="group_b")

    observed = (
        float(count_a),
        float(total_a - count_a),
        float(count_b),
        float(total_b - count_b),
    )
    row_totals = (float(total_a), float(total_b))
    col_totals = (
        float(count_a + count_b),
        float(total_a + total_b - count_a - count_b),
    )
    grand_total = float(total_a + total_b)

    total = 0.0
    for row_index, row_total in enumerate(row_totals):
        for col_index, col_total in enumerate(col_totals):
            obs = observed[row_index * 2 + col_index]
            if obs <= 0:
                continue
            expected = row_total * col_total / grand_total
            if expected > 0:
                total += obs * math.log(obs / expected)

    g2 = 2.0 * total
    return 0.0 if g2 < 0 and abs(g2) < EPSILON else max(0.0, g2)
