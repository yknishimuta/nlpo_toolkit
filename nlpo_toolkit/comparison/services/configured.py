from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from ..config import ComparisonSpec
from ..engine import (
    FrequencyTable, PairwiseComparisonOptions, ZeroHandling, ZeroHandlingMode,
    compare_pair,
)
from ..errors import ComparisonEngineError, ComparisonServiceError
from ..results import ConfiguredComparisonResult, PairwiseComparisonRow


def _frequency_table(label: str, counter: Mapping[str, int]) -> FrequencyTable:
    for item, count in counter.items():
        if not isinstance(item, str):
            raise ComparisonServiceError("configured counter keys must be strings")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ComparisonServiceError("configured counter values must be integers")
    return FrequencyTable.from_counts(label, counter)


def _primary(row: PairwiseComparisonRow, by: str) -> float | str:
    if by == "log_likelihood":
        return row.log_likelihood
    if by == "abs_log_ratio":
        return abs(row.log_ratio)
    if by == "total_count":
        return row.total_count
    return row.item


def _sorted_rows(
    rows: tuple[PairwiseComparisonRow, ...], spec: ComparisonSpec,
) -> tuple[PairwiseComparisonRow, ...]:
    ordered = sorted(rows, key=lambda row: row.item)
    ordered.sort(
        key=lambda row: _primary(row, spec.sort.by),
        reverse=spec.sort.descending,
    )
    return tuple(ordered)


def compare_configured_counters(
    *, counter_a: Mapping[str, int], counter_b: Mapping[str, int],
    spec: ComparisonSpec, analysis_unit: str,
) -> ConfiguredComparisonResult[ComparisonSpec, FrequencyTable]:
    for label, counter in ((spec.group_a, counter_a), (spec.group_b, counter_b)):
        if sum(counter.values()) == 0:
            cause = ComparisonEngineError(
                f"frequency table '{label}' has zero total count"
            )
            raise ComparisonServiceError(
                f"comparison '{spec.name}': group '{label}' has zero target tokens"
            ) from cause
    try:
        table_a = _frequency_table(spec.group_a, counter_a)
        table_b = _frequency_table(spec.group_b, counter_b)
        comparison = compare_pair(
            table_a, table_b,
            options=PairwiseComparisonOptions(
                scale=float(spec.scale),
                min_total_count=float(spec.min_total_count),
                zero_handling=ZeroHandling(
                    ZeroHandlingMode.ZERO_ONLY, spec.zero_correction,
                ),
            ),
        )
    except ComparisonEngineError as exc:
        raise ComparisonServiceError(
            f"comparison '{spec.name}': {exc}"
        ) from exc
    return ConfiguredComparisonResult(
        spec, analysis_unit,
        replace(comparison, rows=_sorted_rows(comparison.rows, spec)),
    )
