from __future__ import annotations

import csv
import math
from pathlib import Path

from .errors import StylometryError
from .models import (
    FeatureDataset,
    FeatureObservation,
    FeatureSelection,
    InputFormat,
)


def _read_rows(path: Path, *, delimiter: str) -> list[list[str]]:
    if not path.exists():
        raise StylometryError(f"feature file not found: {path}")
    if not path.is_file():
        raise StylometryError(f"feature path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            return list(csv.reader(stream, delimiter=delimiter))
    except UnicodeDecodeError as exc:
        raise StylometryError(f"feature file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise StylometryError(f"could not read feature file {path}: {exc}") from exc


def read_feature_dataset(
    path: Path,
    *,
    input_format: InputFormat,
    selection: FeatureSelection,
) -> FeatureDataset:
    delimiter = "," if input_format == "csv" else "\t"
    rows = _read_rows(path, delimiter=delimiter)
    if not rows or not rows[0]:
        raise StylometryError("feature table has no header")
    header = tuple(cell.strip() for cell in rows[0])
    if any(not name for name in header):
        raise StylometryError("feature table contains an empty header name")
    seen: set[str] = set()
    for name in header:
        if name in seen:
            raise StylometryError(f"duplicate feature-table column: {name}")
        seen.add(name)
    if selection.id_column not in header:
        raise StylometryError(f"identifier column not found: {selection.id_column}")
    for column in selection.columns:
        if column not in header:
            raise StylometryError(f"feature column not found: {column}")
    if selection.id_column in selection.columns:
        raise StylometryError("identifier column cannot also be a feature column")
    selected = tuple(
        name
        for name in header
        if name != selection.id_column
        and (
            name in selection.columns
            or any(name.startswith(prefix) for prefix in selection.prefixes)
        )
    )
    if not selected:
        raise StylometryError("no feature columns matched the requested selectors")
    id_index = header.index(selection.id_column)
    feature_indices = tuple(header.index(name) for name in selected)
    observations: list[FeatureObservation] = []
    identifier_rows: dict[str, int] = {}
    for row_number, row in enumerate(rows[1:], start=2):
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise StylometryError(f"feature table row {row_number} has the wrong width")
        identifier = row[id_index].strip()
        if not identifier:
            raise StylometryError(
                f"identifier value is empty at row {row_number}, column {selection.id_column}"
            )
        previous = identifier_rows.get(identifier)
        if previous is not None:
            raise StylometryError(
                f"identifier column {selection.id_column!r} contains duplicate value "
                f"{identifier!r} at rows {previous} and {row_number}; "
                "use a unique column such as sample_id"
            )
        identifier_rows[identifier] = row_number
        values: list[float] = []
        for name, index in zip(selected, feature_indices, strict=True):
            cell = row[index].strip()
            if not cell:
                raise StylometryError(
                    f"feature value is empty at row {row_number}, column {name}"
                )
            try:
                value = float(cell)
            except ValueError as exc:
                raise StylometryError(
                    f"feature value is not numeric at row {row_number}, column {name}: {cell!r}"
                ) from exc
            if not math.isfinite(value):
                raise StylometryError(
                    f"feature value must be finite at row {row_number}, column {name}"
                )
            values.append(value)
        observations.append(FeatureObservation(identifier, tuple(values)))
    if len(observations) < 2:
        raise StylometryError("Burrows's Delta requires at least two observations")
    return FeatureDataset(selected, tuple(observations))
