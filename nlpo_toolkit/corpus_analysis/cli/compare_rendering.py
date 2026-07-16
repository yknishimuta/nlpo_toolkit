from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from nlpo_toolkit.comparison.results import (
    CsvMultiComparisonResult, CsvPairComparisonResult,
)


CsvScalar = str | int | float


@dataclass(frozen=True)
class RenderedComparisonTable:
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, CsvScalar], ...]


def render_csv_comparison_rows(
    result: CsvPairComparisonResult | CsvMultiComparisonResult,
) -> RenderedComparisonTable:
    if isinstance(result, CsvPairComparisonResult):
        comparison = result.comparison
        label_a, label_b = comparison.table_a.label, comparison.table_b.label
        columns = (
            "term", f"{label_a}_count", f"{label_b}_count",
            f"{label_a}_relative", f"{label_b}_relative", "difference",
            "ratio", "log_ratio", "total_count",
        )
        rows = tuple({
            "term": row.item, f"{label_a}_count": row.count_a,
            f"{label_b}_count": row.count_b, f"{label_a}_relative": row.rate_a,
            f"{label_b}_relative": row.rate_b, "difference": row.rate_difference,
            "ratio": row.ratio, "log_ratio": row.log_ratio,
            "total_count": row.total_count,
        } for row in result.rows)
        return RenderedComparisonTable(columns, rows)

    comparison = result.comparison
    labels = tuple(table.label for table in comparison.tables)
    columns = (
        "term", *(f"{label}_count" for label in labels),
        *(f"{label}_relative" for label in labels),
        "max_label", "max_relative", "min_label", "min_relative",
        "range_relative", "total_count",
    )
    rendered = []
    for row in result.rows:
        values: dict[str, CsvScalar] = {"term": row.item}
        values.update({f"{label}_count": row.counts[label] for label in labels})
        values.update({f"{label}_relative": row.rates[label] for label in labels})
        values.update({
            "max_label": row.max_label, "max_relative": row.max_rate,
            "min_label": row.min_label, "min_relative": row.min_rate,
            "range_relative": row.range_relative, "total_count": row.total_count,
        })
        rendered.append(values)
    return RenderedComparisonTable(columns, tuple(rendered))
