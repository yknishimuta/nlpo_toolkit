"""KWIC concordance generation from complete token artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from nlpo_toolkit.immutable_collections import freeze_mapping
from pathlib import Path

from .token_artifact.errors import TokenArtifactError
from .token_artifact.reader import read_token_records
from .token_sequences.context import build_token_context
from .token_sequences.fields import TokenField, token_field_value
from .token_sequences.grouping import TokenSequenceError, build_token_sequence_collection


class ConcordanceError(ValueError):
    pass


@dataclass(frozen=True)
class ConcordanceRequest:
    tokens_path: Path
    keys: tuple[str, ...]
    field: TokenField
    window: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "keys", tuple(self.keys))


@dataclass(frozen=True)
class ConcordanceCommandResult:
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "rows", tuple(freeze_mapping(row) for row in self.rows))


def build_concordance_rows(
    *, tokens_path: Path, keys: list[str], field: TokenField, window: int,
) -> tuple[list[str], list[dict[str, str]]]:
    if field not in {"token", "lemma"}:
        raise ConcordanceError("field must be 'token' or 'lemma'.")
    if window < 0:
        raise ConcordanceError("window must be zero or greater.")
    key_set = {key.strip().casefold() for key in keys if key.strip()}
    if not key_set:
        raise ConcordanceError("--keys must contain at least one search key.")

    try:
        collection = build_token_sequence_collection(
            read_token_records(tokens_path, verify_hash=True)
        )
    except (TokenArtifactError, TokenSequenceError) as exc:
        raise ConcordanceError(str(exc)) from exc

    all_items = tuple(
        item for sequence in collection.sequences for item in sequence.items
    )
    metadata_columns = [
        column for column, present in (
            ("file", any(item.source_file for item in all_items)),
            ("group", any(item.group for item in all_items)),
            ("sentence", any(item.sentence for item in all_items)),
        ) if present
    ]
    output_columns = metadata_columns + [
        "key", "field", "token", "lemma", "left", "node", "right",
    ]
    rows: list[dict[str, str]] = []
    candidates = sorted(
        (item for item in all_items if item.included),
        key=lambda item: item.global_token_index,
    )
    for item in candidates:
        value = token_field_value(item, field).strip()
        if value.casefold() not in key_set:
            continue
        location = collection.require_location(item.global_token_index)
        context = build_token_context(location, window=window)
        row: dict[str, str] = {}
        if "file" in metadata_columns:
            row["file"] = item.source_file or ""
        if "group" in metadata_columns:
            row["group"] = item.group
        if "sentence" in metadata_columns:
            row["sentence"] = item.sentence
        row.update({
            "key": value, "field": field, "token": item.token,
            "lemma": item.lemma or "", "left": context.left_text(),
            "node": context.node, "right": context.right_text(),
        })
        rows.append(row)
    return output_columns, rows


def build_concordance(request: ConcordanceRequest) -> ConcordanceCommandResult:
    columns, rows = build_concordance_rows(
        tokens_path=request.tokens_path, keys=list(request.keys),
        field=request.field, window=request.window,
    )
    return ConcordanceCommandResult(tuple(columns), tuple(rows))
