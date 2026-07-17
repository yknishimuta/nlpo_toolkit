from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from ..engine import (
    PairwiseComparisonOptions, ZeroHandling, ZeroHandlingMode,
    compare_many, compare_pair,
)
from ..errors import ComparisonEngineError, ComparisonServiceError
from ..frequency_io import derive_frequency_labels, read_frequency_table
from ..results import (
    CsvMultiComparisonResult, CsvPairComparisonResult,
    MultiComparisonRow, PairwiseComparisonRow,
)


CsvComparisonSortKey = Literal[
    "abs-log-ratio", "log-ratio", "difference",
    "range-relative", "total", "term",
]


@dataclass(frozen=True)
class CsvComparisonRequest:
    inputs: tuple[Path, ...]
    labels: tuple[str, ...] | None = None
    smoothing: float = 0.5
    min_total_count: float = 1.0
    top: int | None = None
    sort: CsvComparisonSortKey | None = None
    ascending: bool = False
    key_column: str | None = None
    count_column: str | None = None


CsvComparisonResult = CsvPairComparisonResult | CsvMultiComparisonResult


def _validate(request: CsvComparisonRequest) -> None:
    if len(request.inputs) < 2:
        raise ComparisonServiceError("--inputs requires at least two frequency CSV files")
    if request.labels is not None:
        if len(request.labels) != len(request.inputs):
            raise ComparisonServiceError("--labels must have the same length as --inputs")
        labels = tuple(label.strip() for label in request.labels)
        if any(not label for label in labels):
            raise ComparisonServiceError("labels must be non-empty")
        if len(set(labels)) != len(labels):
            raise ComparisonServiceError("labels must be unique")
    for value, name in ((request.smoothing, "smoothing"), (request.min_total_count, "min_total_count")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ComparisonServiceError(f"{name} must be a finite non-negative number")
    if request.top is not None and (isinstance(request.top, bool) or request.top < 1):
        raise ComparisonServiceError("top must be a positive integer")
    if request.key_column is not None and not request.key_column.strip():
        raise ComparisonServiceError("key_column must be non-empty")
    if request.count_column is not None and not request.count_column.strip():
        raise ComparisonServiceError("count_column must be non-empty")


def _pair_value(row: PairwiseComparisonRow, key: CsvComparisonSortKey) -> float | str:
    return {
        "abs-log-ratio": abs(row.log_ratio), "log-ratio": row.log_ratio,
        "difference": row.rate_difference, "total": row.total_count,
        "term": row.item,
    }[key]


def _many_value(row: MultiComparisonRow, key: CsvComparisonSortKey) -> float | str:
    return {"range-relative": row.range_relative, "total": row.total_count, "term": row.item}[key]


def execute_csv_comparison(request: CsvComparisonRequest) -> CsvComparisonResult:
    _validate(request)
    labels = request.labels or derive_frequency_labels(request.inputs)
    tables = tuple(
        read_frequency_table(path, label=label, key_column=request.key_column,
                             count_column=request.count_column)
        for path, label in zip(request.inputs, labels)
    )
    try:
        if len(tables) == 2:
            sort_key = request.sort or "abs-log-ratio"
            if sort_key == "range-relative":
                raise ComparisonServiceError("range-relative sort requires three or more inputs")
            comparison = compare_pair(
                tables[0], tables[1],
                options=PairwiseComparisonOptions(
                    scale=1.0, min_total_count=request.min_total_count,
                    zero_handling=ZeroHandling(ZeroHandlingMode.ADDITIVE, request.smoothing),
                ),
            )
            rows = sorted(comparison.rows, key=lambda row: row.item)
            rows.sort(key=lambda row: _pair_value(row, sort_key), reverse=not request.ascending)
            if request.top is not None:
                rows = rows[:request.top]
            return CsvPairComparisonResult(
                comparison=replace(comparison, rows=tuple(rows))
            )
        sort_key = request.sort or "range-relative"
        if sort_key in {"abs-log-ratio", "log-ratio", "difference"}:
            raise ComparisonServiceError(f"{sort_key} sort requires exactly two inputs")
        comparison_many = compare_many(tables, scale=1.0, min_total_count=request.min_total_count)
        many_rows = sorted(comparison_many.rows, key=lambda row: row.item)
        many_rows.sort(key=lambda row: _many_value(row, sort_key), reverse=not request.ascending)
        if request.top is not None:
            many_rows = many_rows[:request.top]
        return CsvMultiComparisonResult(
            comparison=replace(comparison_many, rows=tuple(many_rows))
        )
    except ComparisonEngineError as exc:
        raise ComparisonServiceError(str(exc)) from exc
