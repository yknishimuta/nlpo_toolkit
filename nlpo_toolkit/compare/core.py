from __future__ import annotations

import csv
import io
import math
import sys
from pathlib import Path
from typing import Any, Iterable, TextIO


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


def _relative(count: float, total: float, smoothing: float, vocab_size: int) -> float:
    denominator = total + smoothing * vocab_size
    if denominator <= 0:
        return 0.0
    return (count + smoothing) / denominator


def _raw_relative(count: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return count / total


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

    terms = sorted(set().union(*(set(t) for t in tables)))
    totals = [sum(table.values()) for table in tables]
    vocab_size = max(len(terms), 1)
    rows: list[dict[str, Any]] = []

    for term in terms:
        counts = [table.get(term, 0.0) for table in tables]
        total_count = sum(counts)
        if total_count < min_total_count:
            continue

        relatives = [_raw_relative(count, total) for count, total in zip(counts, totals)]

        row: dict[str, Any] = {"term": term}
        for label, count in zip(labels, counts):
            row[f"{label}_count"] = count
        for label, rel in zip(labels, relatives):
            row[f"{label}_relative"] = rel

        if len(tables) == 2:
            rel_a, rel_b = relatives
            smoothed_a, smoothed_b = [
                _relative(count, total, smoothing, vocab_size)
                for count, total in zip(counts, totals)
            ]
            row["difference"] = rel_a - rel_b
            row["ratio"] = smoothed_a / smoothed_b if smoothed_b else math.inf
            row["log_ratio"] = math.log2(row["ratio"]) if row["ratio"] > 0 else -math.inf
        else:
            max_idx = max(range(len(relatives)), key=lambda i: relatives[i])
            min_idx = min(range(len(relatives)), key=lambda i: relatives[i])
            row["max_label"] = labels[max_idx]
            row["max_relative"] = relatives[max_idx]
            row["min_label"] = labels[min_idx]
            row["min_relative"] = relatives[min_idx]
            row["range_relative"] = relatives[max_idx] - relatives[min_idx]

        row["total_count"] = total_count
        rows.append(row)

    return rows


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
        load_frequency_csv(path, key_column=key_column, count_column=count_column)
        for path in inputs
    ]
    rows = compare_frequency_tables(
        tables,
        effective_labels,
        smoothing=smoothing,
        min_total_count=min_total_count,
    )
    sort_key = sort or ("abs-log-ratio" if len(inputs) == 2 else "range-relative")
    rows = sort_compare_rows(rows, sort_key=sort_key, ascending=ascending)
    if top is not None:
        rows = rows[:top]
    write_compare_output(rows, out=out, format=output_format)
    return 0
