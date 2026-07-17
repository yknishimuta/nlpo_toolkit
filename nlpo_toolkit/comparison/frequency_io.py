"""Frequency CSV input for comparison services."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .models import FrequencyTable
from .errors import ComparisonEngineError, FrequencyTableReadError


KEY_COLUMN_CANDIDATES = ("lemma", "term", "key", "ngram", "token")
COUNT_COLUMN_CANDIDATES = ("count", "freq", "frequency")


def detect_frequency_columns(
    fieldnames: Iterable[str], *, key_column: str | None = None,
    count_column: str | None = None,
) -> tuple[str, str]:
    fields = [str(field) for field in fieldnames]
    lookup = {field.casefold(): field for field in fields}
    if key_column is not None:
        if key_column not in fields:
            raise FrequencyTableReadError(f"Key column not found: {key_column}")
        key = key_column
    else:
        key = next((lookup[name] for name in KEY_COLUMN_CANDIDATES if name in lookup), "")
        if not key:
            raise FrequencyTableReadError(
                "Could not detect key column. Tried: " + ", ".join(KEY_COLUMN_CANDIDATES)
            )
    if count_column is not None:
        if count_column not in fields:
            raise FrequencyTableReadError(f"Count column not found: {count_column}")
        count = count_column
    else:
        count = next((lookup[name] for name in COUNT_COLUMN_CANDIDATES if name in lookup), "")
        if not count:
            raise FrequencyTableReadError(
                "Could not detect count column. Tried: " + ", ".join(COUNT_COLUMN_CANDIDATES)
            )
    return key, count


def read_frequency_counts(
    path: Path, *, key_column: str | None = None,
    count_column: str | None = None,
) -> Mapping[str, float]:
    path = Path(path)
    if not path.exists():
        raise FrequencyTableReadError(f"Input file not found: {path}")
    if not path.is_file():
        raise FrequencyTableReadError(f"Input path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise FrequencyTableReadError(f"Input CSV has no header: {path}")
            key_name, count_name = detect_frequency_columns(
                reader.fieldnames, key_column=key_column, count_column=count_column,
            )
            counts: dict[str, float] = {}
            for row_number, row in enumerate(reader, start=2):
                key = str(row.get(key_name) or "").strip()
                if not key:
                    continue
                raw = row.get(count_name)
                try:
                    count = float(str(raw or "").strip())
                except ValueError as exc:
                    raise FrequencyTableReadError(
                        f"Invalid numeric count in {path} row {row_number}, "
                        f"column {count_name}: {raw!r}"
                    ) from exc
                if not math.isfinite(count):
                    raise FrequencyTableReadError(
                        f"Invalid numeric count in {path} row {row_number}, "
                        f"column {count_name}: {raw!r}"
                    )
                counts[key] = counts.get(key, 0.0) + count
            return counts
    except UnicodeError as exc:
        raise FrequencyTableReadError(f"Input CSV is not valid UTF-8: {path}") from exc


def read_frequency_table(
    path: Path, *, label: str, key_column: str | None = None,
    count_column: str | None = None,
) -> FrequencyTable:
    try:
        return FrequencyTable.from_counts(
            label,
            read_frequency_counts(path, key_column=key_column, count_column=count_column),
        )
    except ComparisonEngineError as exc:
        raise FrequencyTableReadError(f"Invalid frequency table {path}: {exc}") from exc


def derive_frequency_labels(paths: Sequence[Path]) -> tuple[str, ...]:
    labels: list[str] = []
    used: set[str] = set()
    for path in paths:
        label = Path(path).stem
        if label.startswith("frequency_"):
            label = label[len("frequency_"):]
        label = label or "input"
        base, index = label, 2
        while label in used:
            label = f"{base}_{index}"
            index += 1
        used.add(label)
        labels.append(label)
    return tuple(labels)
