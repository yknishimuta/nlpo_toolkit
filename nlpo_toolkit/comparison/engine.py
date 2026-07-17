from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

from .errors import ComparisonEngineError

from .metrics import (
    EPSILON,
    calculate_log_likelihood,
    calculate_log_ratio,
    calculate_ratio,
    normalized_rate,
)
from .models import FrequencyTable
from .results import (
    MultiComparisonResult,
    MultiComparisonRow,
    PairwiseComparisonResult,
    PairwiseComparisonRow,
)


class ZeroHandlingMode(str, Enum):
    ZERO_ONLY = "zero_only"
    ADDITIVE = "additive"


@dataclass(frozen=True)
class ZeroHandling:
    mode: ZeroHandlingMode
    value: float

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, ZeroHandlingMode) else ZeroHandlingMode(self.mode)
        except ValueError as exc:
            raise ComparisonEngineError(f"unsupported zero handling mode: {self.mode!r}") from exc
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ComparisonEngineError("zero handling value must be a finite number")
        value = float(self.value)
        if not math.isfinite(value):
            raise ComparisonEngineError("zero handling value must be finite")
        if mode is ZeroHandlingMode.ZERO_ONLY and value <= 0:
            raise ComparisonEngineError("zero-only correction must be positive")
        if mode is ZeroHandlingMode.ADDITIVE and value < 0:
            raise ComparisonEngineError("additive smoothing must be non-negative")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class PairwiseComparisonOptions:
    scale: float = 1.0
    min_total_count: float = 1.0
    zero_handling: ZeroHandling = ZeroHandling(ZeroHandlingMode.ZERO_ONLY, 0.5)

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, (int, float)):
            raise ComparisonEngineError("scale must be a positive finite number")
        if isinstance(self.min_total_count, bool) or not isinstance(self.min_total_count, (int, float)):
            raise ComparisonEngineError("min_total_count must be a finite number")
        scale, minimum = float(self.scale), float(self.min_total_count)
        if not math.isfinite(scale) or scale <= 0:
            raise ComparisonEngineError("scale must be a positive finite number")
        if not math.isfinite(minimum) or minimum < 0:
            raise ComparisonEngineError("min_total_count must be >= 0")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "min_total_count", minimum)


def compare_pair(
    table_a: FrequencyTable,
    table_b: FrequencyTable,
    *,
    options: PairwiseComparisonOptions,
) -> PairwiseComparisonResult:
    items = sorted(set(table_a.counts) | set(table_b.counts))
    vocabulary_size = max(len(items), 1)
    rows: list[PairwiseComparisonRow] = []

    for item in items:
        count_a = float(table_a.counts.get(item, 0.0))
        count_b = float(table_b.counts.get(item, 0.0))
        total_count = count_a + count_b
        if total_count < options.min_total_count:
            continue

        rate_a = normalized_rate(count_a, table_a.total, scale=options.scale)
        rate_b = normalized_rate(count_b, table_b.total, scale=options.scale)
        ratio = calculate_ratio(
            count_a=count_a,
            count_b=count_b,
            total_a=table_a.total,
            total_b=table_b.total,
            zero_handling=options.zero_handling,
            vocabulary_size=vocabulary_size,
        )
        log_ratio = calculate_log_ratio(
            count_a=count_a,
            count_b=count_b,
            total_a=table_a.total,
            total_b=table_b.total,
            zero_handling=options.zero_handling,
            vocabulary_size=vocabulary_size,
        )
        log_likelihood = calculate_log_likelihood(
            count_a=count_a,
            count_b=count_b,
            total_a=table_a.total,
            total_b=table_b.total,
        )
        if log_ratio > EPSILON:
            direction = table_a.label
        elif log_ratio < -EPSILON:
            direction = table_b.label
        else:
            direction = "equal"

        rows.append(
            PairwiseComparisonRow(
                item=item,
                count_a=count_a,
                count_b=count_b,
                total_count=total_count,
                rate_a=rate_a,
                rate_b=rate_b,
                rate_difference=rate_a - rate_b,
                ratio=ratio,
                log_ratio=log_ratio,
                log_likelihood=log_likelihood,
                direction=direction,
            )
        )

    return PairwiseComparisonResult(
        table_a=table_a,
        table_b=table_b,
        scale=options.scale,
        vocabulary_union_size=len(items),
        rows_before_filter=len(items),
        rows=tuple(rows),
    )


def compare_many(
    tables: Sequence[FrequencyTable],
    *,
    scale: float = 1.0,
    min_total_count: float = 1.0,
) -> MultiComparisonResult:
    if len(tables) < 2:
        raise ComparisonEngineError("At least two frequency tables are required")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ComparisonEngineError("scale must be a positive finite number")
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ComparisonEngineError("scale must be a positive finite number")
    if isinstance(min_total_count, bool) or not isinstance(min_total_count, (int, float)):
        raise ComparisonEngineError("min_total_count must be a finite number")
    min_total_count = float(min_total_count)
    if not math.isfinite(min_total_count) or min_total_count < 0:
        raise ComparisonEngineError("min_total_count must be >= 0")

    items = sorted(set().union(*(set(table.counts) for table in tables)))
    rows: list[MultiComparisonRow] = []

    for item in items:
        counts = {table.label: float(table.counts.get(item, 0.0)) for table in tables}
        total_count = sum(counts.values())
        if total_count < min_total_count:
            continue

        rates = {
            table.label: normalized_rate(
                float(table.counts.get(item, 0.0)),
                table.total,
                scale=scale,
            )
            for table in tables
        }
        labels = [table.label for table in tables]
        max_label = max(labels, key=lambda label: rates[label])
        min_label = min(labels, key=lambda label: rates[label])
        max_rate = rates[max_label]
        min_rate = rates[min_label]

        rows.append(
            MultiComparisonRow(
                item=item,
                counts=counts,
                rates=rates,
                total_count=total_count,
                max_label=max_label,
                max_rate=max_rate,
                min_label=min_label,
                min_rate=min_rate,
                range_relative=max_rate - min_rate,
            )
        )

    return MultiComparisonResult(
        tables=tuple(tables),
        scale=float(scale),
        vocabulary_union_size=len(items),
        rows_before_filter=len(items),
        rows=tuple(rows),
    )
