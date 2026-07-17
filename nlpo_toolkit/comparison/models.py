from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from nlpo_toolkit.immutable_collections import freeze_mapping

from .errors import ComparisonEngineError


@dataclass(frozen=True)
class FrequencyTable:
    label: str
    counts: Mapping[str, float]
    total: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ComparisonEngineError("frequency table label must be a non-empty string")
        normalized: dict[str, float] = {}
        for key, value in self.counts.items():
            if not isinstance(key, str):
                raise ComparisonEngineError("frequency table keys must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ComparisonEngineError(f"count for {key!r} must be a finite number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ComparisonEngineError(f"count for {key!r} must be finite and >= 0")
            normalized[key] = numeric
        total = float(self.total)
        expected_total = sum(normalized.values())
        if expected_total <= 0:
            raise ComparisonEngineError(
                f"frequency table '{self.label}' has zero total count"
            )
        if not math.isfinite(total) or not math.isclose(total, expected_total):
            raise ComparisonEngineError(
                "frequency table total must equal a positive count sum"
            )
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "counts", freeze_mapping(normalized))
        object.__setattr__(self, "total", total)

    @classmethod
    def from_counts(
        cls, label: str, counts: Mapping[str, int | float]
    ) -> FrequencyTable:
        try:
            total = sum(float(value) for value in counts.values())
        except (TypeError, ValueError):
            total = math.nan
        return cls(label=label, counts=counts, total=total)
