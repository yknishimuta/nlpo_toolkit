from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

from .errors import StylometryMetricError


class StylometryMetric(str, Enum):
    BURROWS_DELTA = "burrows_delta"
    MANHATTAN = "manhattan"
    COSINE_DELTA = "cosine_delta"
    COSINE_SIMILARITY = "cosine_similarity"


class ScorePreference(str, Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


def score_preference(metric: StylometryMetric) -> ScorePreference:
    if not isinstance(metric, StylometryMetric):
        raise StylometryMetricError(f"unsupported stylometry metric: {metric!r}")
    if metric is StylometryMetric.COSINE_SIMILARITY:
        return ScorePreference.HIGHER_IS_BETTER
    return ScorePreference.LOWER_IS_BETTER


def _validated_vectors(
    left: Sequence[float], right: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    left_values = tuple(left)
    right_values = tuple(right)
    if not left_values or len(left_values) != len(right_values):
        raise StylometryMetricError("vectors must have the same non-zero width")
    for value in left_values + right_values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise StylometryMetricError("vector values must be finite numbers")
    return (
        tuple(float(value) for value in left_values),
        tuple(float(value) for value in right_values),
    )


def manhattan_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_values, right_values = _validated_vectors(left, right)
    result = float(sum(abs(a - b) for a, b in zip(left_values, right_values)))
    if not math.isfinite(result):
        raise StylometryMetricError("Manhattan distance must be finite")
    return result


def burrows_delta(left: Sequence[float], right: Sequence[float]) -> float:
    left_values, right_values = _validated_vectors(left, right)
    return float(manhattan_distance(left_values, right_values) / len(left_values))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_values, right_values = _validated_vectors(left, right)
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        raise StylometryMetricError("cosine metric is undefined for a zero-norm vector")
    result = sum(a * b for a, b in zip(left_values, right_values)) / (
        left_norm * right_norm
    )
    if not math.isfinite(result):
        raise StylometryMetricError("cosine similarity must be finite")
    return float(min(1.0, max(-1.0, result)))


def cosine_delta(left: Sequence[float], right: Sequence[float]) -> float:
    result = 1.0 - cosine_similarity(left, right)
    return float(min(2.0, max(0.0, result)))


def compute_stylometry_score(
    left: Sequence[float],
    right: Sequence[float],
    *,
    metric: StylometryMetric,
) -> float:
    if metric is StylometryMetric.BURROWS_DELTA:
        return burrows_delta(left, right)
    if metric is StylometryMetric.MANHATTAN:
        return manhattan_distance(left, right)
    if metric is StylometryMetric.COSINE_DELTA:
        return cosine_delta(left, right)
    if metric is StylometryMetric.COSINE_SIMILARITY:
        return cosine_similarity(left, right)
    raise StylometryMetricError(f"unsupported stylometry metric: {metric!r}")
