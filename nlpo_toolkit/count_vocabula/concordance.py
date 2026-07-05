from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import TextIO


class ConcordanceError(ValueError):
    pass


def _open_output(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline=""), True


def _split_sentence(sentence: str) -> list[str]:
    return sentence.split()


def _kwic(row: dict[str, str], field: str, window: int) -> tuple[str, str, str]:
    node = row.get("token") or row.get(field, "")
    sentence = row.get("sentence", "")
    if not sentence:
        return "", node, ""

    tokens = _split_sentence(sentence)
    try:
        idx = int(row.get("token_idx", ""))
    except ValueError:
        idx = -1

    if idx < 0 or idx >= len(tokens):
        needle = row.get("token", row.get(field, ""))
        lowered = needle.lower()
        idx = next((i for i, token in enumerate(tokens) if token.lower() == lowered), -1)

    if idx < 0:
        return "", node, ""

    left = " ".join(tokens[max(0, idx - window):idx])
    right = " ".join(tokens[idx + 1:idx + 1 + window])
    return left, tokens[idx], right


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

    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if field not in fieldnames:
            raise ConcordanceError(f"Trace must contain '{field}' column.")

        metadata_columns = [
            column for column in ("file", "group", "sentence") if column in fieldnames
        ]
        output_columns = metadata_columns + ["key", "field", "token", "lemma", "left", "node", "right"]
        rows: list[dict[str, str]] = []

        for row in reader:
            value = (row.get(field) or "").strip()
            if value.lower() not in key_set:
                continue

            left, node, right = _kwic(row, field, window)
            out_row = {column: row.get(column, "") for column in metadata_columns}
            out_row.update(
                {
                    "key": value,
                    "field": field,
                    "token": row.get("token", ""),
                    "lemma": row.get("lemma", ""),
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
