from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlpo_toolkit.comparison import (
    ComparisonEngineError,
    PairwiseComparisonOptions,
    ZeroHandling,
    ZeroHandlingMode,
    compare_many,
    compare_pair,
)

from .frequency_io import CompareError, labels_from_paths, load_frequency_table
SORT_KEYS = {"abs-log-ratio", "log-ratio", "difference", "range-relative", "total", "term"}




def _render_pair_rows(result: Any) -> list[dict[str, Any]]:
    label_a = result.table_a.label
    label_b = result.table_b.label
    rows: list[dict[str, Any]] = []
    for row in result.rows:
        rows.append(
            {
                "term": row.item,
                f"{label_a}_count": row.count_a,
                f"{label_b}_count": row.count_b,
                f"{label_a}_relative": row.rate_a,
                f"{label_b}_relative": row.rate_b,
                "difference": row.rate_difference,
                "ratio": row.ratio,
                "log_ratio": row.log_ratio,
                "total_count": row.total_count,
            }
        )
    return rows


def _render_many_rows(result: Any) -> list[dict[str, Any]]:
    labels = [table.label for table in result.tables]
    rows: list[dict[str, Any]] = []
    for row in result.rows:
        rendered: dict[str, Any] = {"term": row.item}
        for label in labels:
            rendered[f"{label}_count"] = row.counts[label]
        for label in labels:
            rendered[f"{label}_relative"] = row.rates[label]
        rendered["max_label"] = row.max_label
        rendered["max_relative"] = row.max_rate
        rendered["min_label"] = row.min_label
        rendered["min_relative"] = row.min_rate
        rendered["range_relative"] = row.range_relative
        rendered["total_count"] = row.total_count
        rows.append(rendered)
    return rows


def sort_compare_rows(
    rows: list[dict[str, Any]],
    *,
    sort_key: str,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    if sort_key not in SORT_KEYS:
        raise CompareError(f"Unsupported sort key: {sort_key}")

    def key(row: dict[str, Any]) -> Any:
        if sort_key == "term":
            return str(row["term"])
        if sort_key == "total":
            return float(row.get("total_count", 0.0))
        if sort_key == "abs-log-ratio":
            return abs(float(row.get("log_ratio", 0.0)))
        if sort_key == "log-ratio":
            return float(row.get("log_ratio", 0.0))
        if sort_key == "difference":
            return float(row.get("difference", 0.0))
        if sort_key == "range-relative":
            return float(row.get("range_relative", 0.0))
        return 0.0

    return sorted(rows, key=lambda row: (key(row), str(row["term"])), reverse=not ascending)


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["term"]
    return list(rows[0].keys())


@dataclass(frozen=True)
class CompareRequest:
    inputs: tuple[Path, ...]
    labels: tuple[str, ...] | None = None
    smoothing: float = 0.5
    min_total_count: float = 1
    top: int | None = None
    sort: str | None = None
    ascending: bool = False
    key_column: str | None = None
    count_column: str | None = None


@dataclass(frozen=True)
class CompareCommandResult:
    rows: tuple[dict[str, Any], ...]
    columns: tuple[str, ...]


def execute_compare_command(request: CompareRequest) -> CompareCommandResult:
    inputs = list(request.inputs)
    labels = list(request.labels) if request.labels is not None else None
    if len(inputs) < 2:
        raise CompareError("--inputs requires at least two frequency CSV files")
    if labels is not None and len(labels) != len(inputs):
        raise CompareError("--labels must have the same length as --inputs")
    effective_labels = labels or labels_from_paths(inputs)
    tables = [
        load_frequency_table(
            path,
            label=label,
            key_column=request.key_column,
            count_column=request.count_column,
        )
        for path, label in zip(inputs, effective_labels)
    ]
    try:
        if len(tables) == 2:
            result = compare_pair(
                tables[0],
                tables[1],
                options=PairwiseComparisonOptions(
                    scale=1.0,
                    min_total_count=request.min_total_count,
                    zero_handling=ZeroHandling(ZeroHandlingMode.ADDITIVE, request.smoothing),
                ),
            )
            rows = _render_pair_rows(result)
        else:
            result = compare_many(tables, scale=1.0, min_total_count=request.min_total_count)
            rows = _render_many_rows(result)
    except ComparisonEngineError as exc:
        raise CompareError(str(exc)) from exc
    sort_key = request.sort or ("abs-log-ratio" if len(inputs) == 2 else "range-relative")
    rows = sort_compare_rows(rows, sort_key=sort_key, ascending=request.ascending)
    if request.top is not None:
        rows = rows[: request.top]
    return CompareCommandResult(rows=tuple(rows), columns=tuple(_fieldnames(rows)))
