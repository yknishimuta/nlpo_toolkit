"""KWIC concordance generation from complete token artifacts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .analysis_records import TokenRecord
from .token_artifact import TokenArtifactError, read_token_records


class ConcordanceError(ValueError):
    pass


@dataclass(frozen=True)
class ConcordanceRequest:
    tokens_path: Path
    keys: tuple[str, ...]
    field: str
    window: int


@dataclass(frozen=True)
class ConcordanceCommandResult:
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def _field_value(record: TokenRecord, field: str) -> str:
    if field == "token":
        return record.token
    return record.lemma or ""


def _sequence_key(record: TokenRecord) -> tuple[str, str, str, int, int]:
    return (
        record.group,
        record.source_file or "",
        record.section or "",
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
    tokens_path: Path,
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
    key_set = {key.strip().lower() for key in keys if key.strip()}
    if not key_set:
        raise ConcordanceError("--keys must contain at least one search key.")

    try:
        all_records = list(read_token_records(tokens_path, verify_hash=True))
    except TokenArtifactError as exc:
        raise ConcordanceError(str(exc)) from exc
    matched_records = [record for record in all_records if record.included]

    metadata_columns = [
        column
        for column, predicate in (
            ("file", any(record.source_file for record in all_records)),
            ("group", any(record.group for record in all_records)),
            ("sentence", any(record.sentence for record in all_records)),
        )
        if predicate
    ]
    output_columns = metadata_columns + ["key", "field", "token", "lemma", "left", "node", "right"]
    rows: list[dict[str, str]] = []
    sequences: dict[tuple[str, str, str, int, int], list[TokenRecord]] = defaultdict(list)
    for record in all_records:
        sequences[_sequence_key(record)].append(record)

    for record in matched_records:
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


def build_concordance(request: ConcordanceRequest) -> ConcordanceCommandResult:
    columns, rows = build_concordance_rows(
        tokens_path=request.tokens_path,
        keys=list(request.keys),
        field=request.field,
        window=request.window,
    )
    return ConcordanceCommandResult(columns=tuple(columns), rows=tuple(rows))
