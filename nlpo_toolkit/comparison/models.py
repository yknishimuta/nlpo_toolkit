from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class ComparisonEngineError(ValueError):
    pass


class ZeroHandlingMode(str, Enum):
    ZERO_ONLY = "zero_only"
    ADDITIVE = "additive"


@dataclass(frozen=True)
class ZeroHandling:
    mode: ZeroHandlingMode
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ZeroHandlingMode):
            object.__setattr__(self, "mode", ZeroHandlingMode(self.mode))
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ComparisonEngineError("zero handling value must be a finite number")
        value = float(self.value)
        if not math.isfinite(value):
            raise ComparisonEngineError("zero handling value must be finite")
        if self.mode is ZeroHandlingMode.ZERO_ONLY and value <= 0:
            raise ComparisonEngineError("zero-only correction must be positive")
        if self.mode is ZeroHandlingMode.ADDITIVE and value < 0:
            raise ComparisonEngineError("additive smoothing must be non-negative")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class FrequencyTable:
    label: str
    counts: Mapping[str, float]
    total: float

    @classmethod
    def from_counts(
        cls,
        label: str,
        counts: Mapping[str, int | float],
    ) -> "FrequencyTable":
        if not isinstance(label, str) or not label.strip():
            raise ComparisonEngineError("frequency table label must be a non-empty string")

        normalized: dict[str, float] = {}
        for key, value in counts.items():
            if not isinstance(key, str):
                raise ComparisonEngineError("frequency table keys must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ComparisonEngineError(f"count for {key!r} must be a finite number")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ComparisonEngineError(f"count for {key!r} must be finite")
            if numeric < 0:
                raise ComparisonEngineError(f"count for {key!r} must be >= 0")
            normalized[key] = numeric

        total = sum(normalized.values())
        if total <= 0:
            raise ComparisonEngineError(f"frequency table '{label}' has zero total count")

        return cls(
            label=label.strip(),
            counts=MappingProxyType(dict(normalized)),
            total=float(total),
        )


@dataclass(frozen=True)
class PairwiseComparisonOptions:
    scale: float = 1.0
    min_total_count: float = 1.0
    zero_handling: ZeroHandling = ZeroHandling(ZeroHandlingMode.ZERO_ONLY, 0.5)

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)):
            raise ComparisonEngineError("scale must be a positive finite number")
        scale = float(self.scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ComparisonEngineError("scale must be a positive finite number")
        if isinstance(self.min_total_count, bool) or not isinstance(self.min_total_count, (int, float)):
            raise ComparisonEngineError("min_total_count must be a finite number")
        min_total_count = float(self.min_total_count)
        if not math.isfinite(min_total_count) or min_total_count < 0:
            raise ComparisonEngineError("min_total_count must be >= 0")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "min_total_count", min_total_count)


@dataclass(frozen=True)
class PairwiseComparisonRow:
    item: str
    count_a: float
    count_b: float
    total_count: float
    rate_a: float
    rate_b: float
    rate_difference: float
    ratio: float
    log_ratio: float
    log_likelihood: float
    direction: str


@dataclass(frozen=True)
class PairwiseComparisonResult:
    table_a: FrequencyTable
    table_b: FrequencyTable
    scale: float
    vocabulary_union_size: int
    rows_before_filter: int
    rows: tuple[PairwiseComparisonRow, ...]


@dataclass(frozen=True)
class MultiComparisonRow:
    item: str
    counts: Mapping[str, float]
    rates: Mapping[str, float]
    total_count: float
    max_label: str
    max_rate: float
    min_label: str
    min_rate: float
    range_relative: float


@dataclass(frozen=True)
class MultiComparisonResult:
    tables: tuple[FrequencyTable, ...]
    scale: float
    vocabulary_union_size: int
    rows_before_filter: int
    rows: tuple[MultiComparisonRow, ...]
