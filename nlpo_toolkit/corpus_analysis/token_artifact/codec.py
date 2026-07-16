from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..analysis_records import TokenRecord
from .errors import TokenArtifactRowError
from .schema import TOKEN_ARTIFACT_COLUMNS


def encode_token_record(record: TokenRecord) -> dict[str, str]:
    def optional(value: object) -> str:
        return "" if value is None else str(value)

    return {
        "group": record.group,
        "source_file": optional(record.source_file),
        "section": optional(record.section),
        "chunk_index": str(record.chunk_index),
        "sentence_index": str(record.sentence_index),
        "token_index": str(record.token_index),
        "global_token_index": str(record.global_token_index),
        "char_start_in_chunk": optional(record.char_start_in_chunk),
        "char_end_in_chunk": optional(record.char_end_in_chunk),
        "char_start_in_text": optional(record.char_start_in_text),
        "char_end_in_text": optional(record.char_end_in_text),
        "sentence": record.sentence,
        "token": record.token,
        "lemma": optional(record.lemma),
        "upos": optional(record.upos),
        "analysis_key": optional(record.analysis_key),
        "included": "true" if record.included else "false",
        "exclusion_reason": optional(record.exclusion_reason),
        "ref_tag": optional(record.ref_tag),
    }


def _value(row: Mapping[str, str], column: str, path: Path, line: int) -> str:
    if column not in row or row[column] is None:
        raise TokenArtifactRowError(
            f"Missing column {column} at {path.resolve()}:{line}"
        )
    value = row[column]
    if not isinstance(value, str):
        raise TokenArtifactRowError(
            f"Invalid value type in column {column} at {path.resolve()}:{line}"
        )
    return value


def _integer(
    row: Mapping[str, str], column: str, path: Path, line: int, *, optional: bool
) -> int | None:
    value = _value(row, column, path, line)
    if optional and value == "":
        return None
    if value == "":
        raise TokenArtifactRowError(
            f"Missing integer in column {column} at {path.resolve()}:{line}"
        )
    try:
        parsed = int(value)
    except ValueError as exc:
        raise TokenArtifactRowError(
            f"Invalid integer {value!r} in column {column} at {path.resolve()}:{line}"
        ) from exc
    if parsed < 0:
        raise TokenArtifactRowError(
            f"Negative integer {parsed} in column {column} at {path.resolve()}:{line}"
        )
    return parsed


def decode_token_record(
    row: Mapping[str, str], *, source_path: Path, line_number: int
) -> TokenRecord:
    for column in TOKEN_ARTIFACT_COLUMNS:
        _value(row, column, source_path, line_number)
    included_raw = _value(row, "included", source_path, line_number)
    if included_raw not in {"true", "false"}:
        raise TokenArtifactRowError(
            f"Invalid boolean {included_raw!r} in column included at "
            f"{source_path.resolve()}:{line_number}; expected 'true' or 'false'"
        )
    def optional(column: str) -> str | None:
        return _value(row, column, source_path, line_number) or None

    return TokenRecord(
        group=_value(row, "group", source_path, line_number),
        source_file=optional("source_file"),
        section=optional("section"),
        chunk_index=_integer(row, "chunk_index", source_path, line_number, optional=False),
        sentence_index=_integer(row, "sentence_index", source_path, line_number, optional=False),
        token_index=_integer(row, "token_index", source_path, line_number, optional=False),
        global_token_index=_integer(row, "global_token_index", source_path, line_number, optional=False),
        char_start_in_chunk=_integer(row, "char_start_in_chunk", source_path, line_number, optional=True),
        char_end_in_chunk=_integer(row, "char_end_in_chunk", source_path, line_number, optional=True),
        char_start_in_text=_integer(row, "char_start_in_text", source_path, line_number, optional=True),
        char_end_in_text=_integer(row, "char_end_in_text", source_path, line_number, optional=True),
        sentence=_value(row, "sentence", source_path, line_number),
        token=_value(row, "token", source_path, line_number),
        lemma=optional("lemma"),
        upos=optional("upos"),
        analysis_key=optional("analysis_key"),
        included=included_raw == "true",
        exclusion_reason=optional("exclusion_reason"),
        ref_tag=optional("ref_tag"),
    )
