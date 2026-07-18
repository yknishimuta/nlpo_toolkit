from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributionSummary:
    variance: float
    median: float
    q25: float
    q75: float


def linear_quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    return float(
        ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction
    )


def summarize_distribution(values: Sequence[int]) -> DistributionSummary:
    if not values:
        return DistributionSummary(0.0, 0.0, 0.0, 0.0)
    return DistributionSummary(
        variance=float(statistics.pvariance(values)),
        median=float(statistics.median(values)),
        q25=linear_quantile(values, 0.25),
        q75=linear_quantile(values, 0.75),
    )
