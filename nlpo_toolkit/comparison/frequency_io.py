"""Frequency CSV loading for comparison services."""

import csv
import math
from pathlib import Path
from typing import Iterable

from .models import ComparisonEngineError, FrequencyTable

KEY_COLUMN_CANDIDATES = ("lemma", "term", "key", "ngram", "token")
COUNT_COLUMN_CANDIDATES = ("count", "freq", "frequency")


class CompareError(RuntimeError):
    pass


def detect_columns(fieldnames: Iterable[str], key_column: str | None = None, count_column: str | None = None) -> tuple[str, str]:
    fields = [str(field) for field in fieldnames]
    lookup = {field.lower(): field for field in fields}
    if key_column is not None:
        if key_column not in fields:
            raise CompareError(f"Key column not found: {key_column}")
        key = key_column
    else:
        key = next((lookup[item] for item in KEY_COLUMN_CANDIDATES if item in lookup), "")
        if not key:
            raise CompareError("Could not detect key column. Tried: " + ", ".join(KEY_COLUMN_CANDIDATES))
    if count_column is not None:
        if count_column not in fields:
            raise CompareError(f"Count column not found: {count_column}")
        count = count_column
    else:
        count = next((lookup[item] for item in COUNT_COLUMN_CANDIDATES if item in lookup), "")
        if not count:
            raise CompareError("Could not detect count column. Tried: " + ", ".join(COUNT_COLUMN_CANDIDATES))
    return key, count


def load_frequency_csv(path: Path, key_column: str | None = None, count_column: str | None = None) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        raise CompareError(f"Input file not found: {path}")
    if not path.is_file():
        raise CompareError(f"Input path is not a file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CompareError(f"Input CSV has no header: {path}")
        key_col, count_col = detect_columns(reader.fieldnames, key_column, count_column)
        table: dict[str, float] = {}
        for row_number, row in enumerate(reader, start=2):
            key = str(row.get(key_col) or "").strip()
            if not key:
                continue
            try:
                count = float(str(row.get(count_col) or "").strip())
            except ValueError as exc:
                raise CompareError(f"Invalid numeric count in {path} row {row_number}, column {count_col}: {row.get(count_col)!r}") from exc
            if not math.isfinite(count):
                raise CompareError(f"Invalid numeric count in {path} row {row_number}, column {count_col}: {row.get(count_col)!r}")
            table[key] = table.get(key, 0.0) + count
    return table


def load_frequency_table(path: Path, *, label: str, key_column: str | None = None, count_column: str | None = None) -> FrequencyTable:
    try:
        return FrequencyTable.from_counts(label, load_frequency_csv(path, key_column, count_column))
    except ComparisonEngineError as exc:
        raise CompareError(str(exc)) from exc


def labels_from_paths(paths: list[Path]) -> list[str]:
    labels: list[str] = []
    used: set[str] = set()
    for path in paths:
        label = Path(path).stem
        prefix = "frequency_"
        if label.startswith(prefix):
            label = label[len(prefix):]
        label = label or "input"
        base = label
        index = 2
        while label in used:
            label = f"{base}_{index}"
            index += 1
        used.add(label)
        labels.append(label)
    return labels
