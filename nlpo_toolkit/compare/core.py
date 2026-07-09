from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO

from nlpo_toolkit.comparison import (
    ComparisonEngineError,
    FrequencyTable,
    PairwiseComparisonOptions,
    ZeroHandling,
    ZeroHandlingMode,
    compare_many,
    compare_pair,
)

KEY_COLUMN_CANDIDATES = ("lemma", "term", "key", "ngram", "token")
COUNT_COLUMN_CANDIDATES = ("count", "freq", "frequency")
METRICS = {"relative", "difference", "ratio", "log-ratio"}
SORT_KEYS = {"abs-log-ratio", "log-ratio", "difference", "range-relative", "total", "term"}


class CompareError(RuntimeError):
    pass


def detect_columns(
    fieldnames: Iterable[str],
    key_column: str | None = None,
    count_column: str | None = None,
) -> tuple[str, str]:
    fields = [str(f) for f in fieldnames]
    field_lookup = {f.lower(): f for f in fields}

    if key_column is not None:
        if key_column not in fields:
            raise CompareError(f"Key column not found: {key_column}")
        key = key_column
    else:
        key = ""
        for candidate in KEY_COLUMN_CANDIDATES:
            if candidate in field_lookup:
                key = field_lookup[candidate]
                break
        if not key:
            raise CompareError(
                "Could not detect key column. Tried: "
                + ", ".join(KEY_COLUMN_CANDIDATES)
            )

    if count_column is not None:
        if count_column not in fields:
            raise CompareError(f"Count column not found: {count_column}")
        count = count_column
    else:
        count = ""
        for candidate in COUNT_COLUMN_CANDIDATES:
            if candidate in field_lookup:
                count = field_lookup[candidate]
                break
        if not count:
            raise CompareError(
                "Could not detect count column. Tried: "
                + ", ".join(COUNT_COLUMN_CANDIDATES)
            )

    return key, count


def _read_count(raw: str, *, path: Path, row_number: int, column: str) -> float:
    try:
        value = float(str(raw).strip())
    except ValueError as exc:
        raise CompareError(
            f"Invalid numeric count in {path} row {row_number}, column {column}: {raw!r}"
        ) from exc
    if not math.isfinite(value):
        raise CompareError(
            f"Invalid numeric count in {path} row {row_number}, column {column}: {raw!r}"
        )
    return value


def load_frequency_csv(
    path: Path,
    key_column: str | None = None,
    count_column: str | None = None,
) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise CompareError(f"Input file not found: {path}")
    if not path.is_file():
        raise CompareError(f"Input path is not a file: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise CompareError(f"Input CSV has no header: {path}")
        key_col, count_col = detect_columns(
            reader.fieldnames,
            key_column=key_column,
            count_column=count_column,
        )

        table: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            key = str(row.get(key_col) or "").strip()
            if not key:
                continue
            count = _read_count(
                str(row.get(count_col) or ""),
                path=path,
                row_number=row_number,
                column=count_col,
            )
            table[key] = table.get(key, 0.0) + count
    return table


def load_frequency_table(
    path: Path,
    *,
    label: str,
    key_column: str | None = None,
    count_column: str | None = None,
) -> FrequencyTable:
    try:
        return FrequencyTable.from_counts(
            label,
            load_frequency_csv(path, key_column=key_column, count_column=count_column),
        )
    except ComparisonEngineError as exc:
        raise CompareError(str(exc)) from exc


def _frequency_table_from_mapping(label: str, table: dict[str, float]) -> FrequencyTable:
    try:
        return FrequencyTable.from_counts(label, table)
    except ComparisonEngineError as exc:
        raise CompareError(str(exc)) from exc


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


def compare_frequency_tables(
    tables: list[dict[str, float]],
    labels: list[str],
    smoothing: float = 0.5,
    min_total_count: float = 1,
) -> list[dict[str, Any]]:
    if len(tables) < 2:
        raise CompareError("At least two input tables are required")
    if len(labels) != len(tables):
        raise CompareError("--labels must have the same length as --inputs")
    if smoothing < 0:
        raise CompareError("--smoothing must be non-negative")

    frequency_tables = [
        _frequency_table_from_mapping(label, table)
        for label, table in zip(labels, tables)
    ]
    try:
        if len(frequency_tables) == 2:
            result = compare_pair(
                frequency_tables[0],
                frequency_tables[1],
                options=PairwiseComparisonOptions(
                    scale=1.0,
                    min_total_count=min_total_count,
                    zero_handling=ZeroHandling(ZeroHandlingMode.ADDITIVE, smoothing),
                ),
            )
            return _render_pair_rows(result)

        result = compare_many(
            frequency_tables,
            scale=1.0,
            min_total_count=min_total_count,
        )
        return _render_many_rows(result)
    except ComparisonEngineError as exc:
        raise CompareError(str(exc)) from exc


def labels_from_paths(paths: list[Path]) -> list[str]:
    labels: list[str] = []
    used: set[str] = set()
    for path in paths:
        label = Path(path).stem
        for prefix in ("noun_frequency_",):
            if label.startswith(prefix):
                label = label[len(prefix):]
        label = label or "input"
        base = label
        i = 2
        while label in used:
            label = f"{base}_{i}"
            i += 1
        used.add(label)
        labels.append(label)
    return labels


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


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.12g}"
    return value


def _fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["term"]
    return list(rows[0].keys())


def write_compare_output(
    rows: list[dict[str, Any]],
    out: Path | TextIO | None = None,
    format: str = "csv",
) -> None:
    if format not in {"csv", "tsv"}:
        raise CompareError("--format must be csv or tsv")
    delimiter = "," if format == "csv" else "\t"
    close = False
    if out is None:
        f = sys.stdout
    elif hasattr(out, "write"):
        f = out  # type: ignore[assignment]
    else:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = path.open("w", encoding="utf-8", newline="")
        close = True

    try:
        writer = csv.DictWriter(f, fieldnames=_fieldnames(rows), delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_value(v) for k, v in row.items()})
    finally:
        if close:
            f.close()


def run_compare(
    *,
    inputs: list[Path],
    labels: list[str] | None = None,
    out: Path | None = None,
    output_format: str = "csv",
    metric: str = "log-ratio",
    smoothing: float = 0.5,
    min_total_count: float = 1,
    top: int | None = None,
    sort: str | None = None,
    ascending: bool = False,
    key_column: str | None = None,
    count_column: str | None = None,
) -> int:
    if len(inputs) < 2:
        raise CompareError("--inputs requires at least two frequency CSV files")
    if metric not in METRICS:
        raise CompareError(f"Unsupported metric: {metric}")
    if labels is not None and len(labels) != len(inputs):
        raise CompareError("--labels must have the same length as --inputs")
    effective_labels = labels or labels_from_paths(inputs)
    tables = [
        load_frequency_table(
            path,
            label=label,
            key_column=key_column,
            count_column=count_column,
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
                    min_total_count=min_total_count,
                    zero_handling=ZeroHandling(ZeroHandlingMode.ADDITIVE, smoothing),
                ),
            )
            rows = _render_pair_rows(result)
        else:
            result = compare_many(tables, scale=1.0, min_total_count=min_total_count)
            rows = _render_many_rows(result)
    except ComparisonEngineError as exc:
        raise CompareError(str(exc)) from exc
    sort_key = sort or ("abs-log-ratio" if len(inputs) == 2 else "range-relative")
    rows = sort_compare_rows(rows, sort_key=sort_key, ascending=ascending)
    if top is not None:
        rows = rows[:top]
    write_compare_output(rows, out=out, format=output_format)
    return 0
