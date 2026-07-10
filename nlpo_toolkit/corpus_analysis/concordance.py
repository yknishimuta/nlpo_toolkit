from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import TextIO

from .analysis_records import TokenRecord
from .token_artifact import TokenArtifactError, read_token_rows, token_artifact_metadata_path


class ConcordanceError(ValueError):
    pass


def _open_output(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline=""), True


def _legacy_fieldnames(path: Path) -> list[str]:
    if token_artifact_metadata_path(path).exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        return next(reader, [])


def _field_value(record: TokenRecord, field: str) -> str:
    if field == "token":
        return record.token
    return record.lemma or ""


def _sequence_key(record: TokenRecord) -> tuple[str, str, int, int]:
    return (
        record.group,
        record.source_file or "",
        record.chunk_index,
        record.sentence_index,
    )


def _kwic_from_sequence(
    sequence: list[TokenRecord],
    record: TokenRecord,
    *,
    window: int,
) -> tuple[str, str, str]:
    ordered = sorted(sequence, key=lambda item: item.token_index)
    try:
        idx = ordered.index(record)
    except ValueError:
        idx = next(
            (i for i, item in enumerate(ordered) if item.global_token_index == record.global_token_index),
            -1,
        )
    if idx < 0:
        return "", record.token, ""
    if len(ordered) == 1 and record.sentence:
        tokens = record.sentence.split()
        if 0 <= record.token_index < len(tokens):
            left = " ".join(tokens[max(0, record.token_index - window):record.token_index])
            right = " ".join(tokens[record.token_index + 1:record.token_index + 1 + window])
            return left, tokens[record.token_index], right
    left = " ".join(item.token for item in ordered[max(0, idx - window):idx])
    right = " ".join(item.token for item in ordered[idx + 1:idx + 1 + window])
    return left, ordered[idx].token, right


def build_concordance_rows(
    *,
    trace_path: Path,
    keys: list[str],
    field: str,
    window: int,
) -> tuple[list[str], list[dict[str, str]]]:
    if field not in {"token", "lemma"}:
        raise ConcordanceError("field must be 'token' or 'lemma'.")
    if window < 0:
        raise ConcordanceError("window must be zero or greater.")
    if not keys:
        raise ConcordanceError("--keys must contain at least one search key.")
    if not trace_path.exists():
        raise ConcordanceError(f"Trace not found: {trace_path}")

    key_set = {key.strip().lower() for key in keys if key.strip()}
    if not key_set:
        raise ConcordanceError("--keys must contain at least one search key.")

    fieldnames = _legacy_fieldnames(trace_path)
    if fieldnames and field not in fieldnames:
        raise ConcordanceError(f"Trace must contain '{field}' column.")

    try:
        records = [record for record in read_token_rows(trace_path) if record.included]
    except TokenArtifactError as exc:
        raise ConcordanceError(str(exc)) from exc

    metadata_columns = [
        column
        for column, predicate in (
            ("file", any(record.source_file for record in records)),
            ("group", any(record.group for record in records)),
            ("sentence", any(record.sentence for record in records)),
        )
        if predicate
    ]
    output_columns = metadata_columns + ["key", "field", "token", "lemma", "left", "node", "right"]
    rows: list[dict[str, str]] = []
    sequences: dict[tuple[str, str, int, int], list[TokenRecord]] = defaultdict(list)
    for record in records:
        sequences[_sequence_key(record)].append(record)

    for record in records:
        value = _field_value(record, field).strip()
        if value.lower() not in key_set:
            continue

        left, node, right = _kwic_from_sequence(
            sequences[_sequence_key(record)],
            record,
            window=window,
        )
        out_row: dict[str, str] = {}
        if "file" in metadata_columns:
            out_row["file"] = record.source_file or ""
        if "group" in metadata_columns:
            out_row["group"] = record.group
        if "sentence" in metadata_columns:
            out_row["sentence"] = record.sentence
        out_row.update(
            {
                "key": value,
                "field": field,
                "token": record.token,
                "lemma": record.lemma or "",
                "left": left,
                "node": node,
                "right": right,
            }
        )
        rows.append(out_row)

    return output_columns, rows


def write_concordance(
    *,
    trace_path: Path,
    keys: list[str],
    field: str,
    window: int,
    output_format: str,
    out_path: Path | None = None,
) -> int:
    if output_format not in {"tsv", "csv"}:
        raise ConcordanceError("output format must be 'tsv' or 'csv'.")

    columns, rows = build_concordance_rows(
        trace_path=trace_path,
        keys=keys,
        field=field,
        window=window,
    )
    delimiter = "\t" if output_format == "tsv" else ","
    out, should_close = _open_output(out_path)
    try:
        writer = csv.DictWriter(out, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if should_close:
            out.close()
    return 0
