"""Stable and complete token artifact serialization and validation.

Token artifacts are the supported input for downstream token-based commands
such as concordance and n-gram analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping

from .analysis_records import TokenRecord as _TokenRecord

TOKEN_ARTIFACT_SCHEMA_NAME = "nlpo-token-artifact"
TOKEN_ARTIFACT_SCHEMA_VERSION = 1

TOKEN_ARTIFACT_COLUMNS = (
    "group",
    "source_file",
    "section",
    "chunk_index",
    "sentence_index",
    "token_index",
    "global_token_index",
    "char_start_in_chunk",
    "char_end_in_chunk",
    "char_start_in_text",
    "char_end_in_text",
    "sentence",
    "token",
    "lemma",
    "upos",
    "analysis_key",
    "included",
    "exclusion_reason",
    "ref_tag",
)

__all__ = [
    "TOKEN_ARTIFACT_COLUMNS",
    "TOKEN_ARTIFACT_SCHEMA_NAME",
    "TOKEN_ARTIFACT_SCHEMA_VERSION",
    "TokenArtifactError",
    "TokenArtifactMetadata",
    "TokenArtifactWriter",
    "read_token_artifact_metadata",
    "read_token_records",
    "token_artifact_metadata_path",
    "validate_token_artifact",
]


class TokenArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class TokenArtifactMetadata:
    schema: str = TOKEN_ARTIFACT_SCHEMA_NAME
    schema_version: int = TOKEN_ARTIFACT_SCHEMA_VERSION
    format: str = "tsv"
    encoding: str = "utf-8"
    delimiter: str = "\t"
    complete: bool = True
    row_count: int = 0
    included_row_count: int = 0
    excluded_row_count: int = 0
    group: str = ""
    source_files: tuple[str, ...] = ()
    analysis_unit: str = ""
    upos_targets: tuple[str, ...] = ()
    nlp: Mapping[str, object] | None = None
    filters: Mapping[str, object] | None = None
    artifact_path: str = ""
    sha256: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["source_files"] = list(self.source_files)
        data["upos_targets"] = list(self.upos_targets)
        data["nlp"] = dict(self.nlp or {})
        data["filters"] = dict(self.filters or {})
        return data


def token_artifact_metadata_path(tsv_path: Path) -> Path:
    return tsv_path.with_name(f"{tsv_path.stem}.meta.json")


def _tmp_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.tmp")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: str, *, path: Path, line_number: int, column: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise TokenArtifactError(
            f"Invalid integer in column {column} at {path}:{line_number}"
        ) from exc


def _required_int(value: str, *, path: Path, line_number: int, column: str) -> int:
    parsed = _optional_int(value, path=path, line_number=line_number, column=column)
    if parsed is None:
        raise TokenArtifactError(
            f"Missing integer in column {column} at {path}:{line_number}"
        )
    return parsed


def _parse_bool(value: str, *, path: Path, line_number: int, column: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise TokenArtifactError(
        f"Invalid boolean in column {column} at {path}:{line_number}"
    )


def _record_to_row(record: _TokenRecord) -> dict[str, str]:
    def opt(value: object) -> str:
        return "" if value is None else str(value)

    return {
        "group": record.group,
        "source_file": opt(record.source_file),
        "section": opt(record.section),
        "chunk_index": str(record.chunk_index),
        "sentence_index": str(record.sentence_index),
        "token_index": str(record.token_index),
        "global_token_index": str(record.global_token_index),
        "char_start_in_chunk": opt(record.char_start_in_chunk),
        "char_end_in_chunk": opt(record.char_end_in_chunk),
        "char_start_in_text": opt(record.char_start_in_text),
        "char_end_in_text": opt(record.char_end_in_text),
        "sentence": record.sentence,
        "token": record.token,
        "lemma": opt(record.lemma),
        "upos": opt(record.upos),
        "analysis_key": opt(record.analysis_key),
        "included": "true" if record.included else "false",
        "exclusion_reason": opt(record.exclusion_reason),
        "ref_tag": opt(record.ref_tag),
    }


def _row_to_record(row: Mapping[str, str], *, path: Path, line_number: int) -> _TokenRecord:
    return _TokenRecord(
        group=row.get("group", ""),
        source_file=row.get("source_file") or None,
        section=row.get("section") or None,
        chunk_index=_required_int(row.get("chunk_index", ""), path=path, line_number=line_number, column="chunk_index"),
        sentence_index=_required_int(row.get("sentence_index", ""), path=path, line_number=line_number, column="sentence_index"),
        token_index=_required_int(row.get("token_index", ""), path=path, line_number=line_number, column="token_index"),
        global_token_index=_required_int(row.get("global_token_index", ""), path=path, line_number=line_number, column="global_token_index"),
        char_start_in_chunk=_optional_int(row.get("char_start_in_chunk", ""), path=path, line_number=line_number, column="char_start_in_chunk"),
        char_end_in_chunk=_optional_int(row.get("char_end_in_chunk", ""), path=path, line_number=line_number, column="char_end_in_chunk"),
        char_start_in_text=_optional_int(row.get("char_start_in_text", ""), path=path, line_number=line_number, column="char_start_in_text"),
        char_end_in_text=_optional_int(row.get("char_end_in_text", ""), path=path, line_number=line_number, column="char_end_in_text"),
        sentence=row.get("sentence", ""),
        token=row.get("token", ""),
        lemma=row.get("lemma") or None,
        upos=row.get("upos") or None,
        analysis_key=row.get("analysis_key") or None,
        included=_parse_bool(row.get("included", ""), path=path, line_number=line_number, column="included"),
        exclusion_reason=row.get("exclusion_reason") or None,
        ref_tag=row.get("ref_tag") or None,
    )


class TokenArtifactWriter:
    def __init__(self, path: Path, metadata_path: Path, *,
                 metadata: TokenArtifactMetadata) -> None:
        self.path = Path(path)
        self.metadata_path = Path(metadata_path)
        self._metadata = metadata
        self._tmp = _tmp_path(self.path)
        self._meta_tmp = _tmp_path(self.metadata_path)
        self._file: Any = None
        self._writer: csv.DictWriter[str] | None = None
        self.row_count = 0
        self.included_row_count = 0
        self.excluded_row_count = 0
        self.final_metadata: TokenArtifactMetadata | None = None

    def __enter__(self) -> "TokenArtifactWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._tmp.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=TOKEN_ARTIFACT_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        self._writer.writeheader()
        return self

    def write(self, record: _TokenRecord) -> None:
        if self._writer is None:
            raise TokenArtifactError("TokenArtifactWriter is not open")
        self._writer.writerow(_record_to_row(record))
        self.row_count += 1
        if record.included:
            self.included_row_count += 1
        else:
            self.excluded_row_count += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self._file is not None:
                self._file.close()
            if exc_type is not None:
                self._cleanup_temp()
                return

            self._tmp.replace(self.path)
            metadata = replace(
                self._metadata,
                complete=True,
                row_count=self.row_count,
                included_row_count=self.included_row_count,
                excluded_row_count=self.excluded_row_count,
                artifact_path=str(self.path),
                sha256=_sha256_file(self.path),
                size_bytes=self.path.stat().st_size,
            )
            self._meta_tmp.write_text(
                json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._meta_tmp.replace(self.metadata_path)
            self.final_metadata = metadata
        except Exception:
            self._cleanup_temp()
            raise

    def _cleanup_temp(self) -> None:
        for path in (self._tmp, self._meta_tmp):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def read_token_artifact_metadata(path: Path) -> TokenArtifactMetadata:
    metadata_path = token_artifact_metadata_path(path)
    if not metadata_path.exists():
        raise TokenArtifactError(f"Token artifact metadata was not found: {metadata_path}")
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise TokenArtifactError(f"Failed to read token artifact metadata: {metadata_path}") from exc

    if data.get("schema") != TOKEN_ARTIFACT_SCHEMA_NAME:
        raise TokenArtifactError(f"Unsupported token artifact schema: {metadata_path}")
    if data.get("schema_version") != TOKEN_ARTIFACT_SCHEMA_VERSION:
        raise TokenArtifactError(
            f"Unsupported token artifact schema version {data.get('schema_version')}: {path}"
        )

    return TokenArtifactMetadata(
        complete=bool(data.get("complete", False)),
        row_count=int(data.get("row_count", 0)),
        included_row_count=int(data.get("included_row_count", 0)),
        excluded_row_count=int(data.get("excluded_row_count", 0)),
        group=str(data.get("group", "")),
        source_files=tuple(str(item) for item in data.get("source_files", [])),
        analysis_unit=str(data.get("analysis_unit", "")),
        upos_targets=tuple(str(item) for item in data.get("upos_targets", [])),
        nlp=data.get("nlp") if isinstance(data.get("nlp"), dict) else {},
        filters=data.get("filters") if isinstance(data.get("filters"), dict) else {},
        artifact_path=str(data.get("artifact_path", "")),
        sha256=str(data.get("sha256", "")),
        size_bytes=int(data.get("size_bytes", 0)),
    )


def read_token_records(
    path: Path,
    *,
    require_complete: bool = True,
    verify_hash: bool = False,
) -> Iterator[_TokenRecord]:
    path = Path(path)
    if not path.exists():
        raise TokenArtifactError(f"Token artifact not found: {path}")
    metadata = read_token_artifact_metadata(path)
    if require_complete and not metadata.complete:
        raise TokenArtifactError(f"Token artifact is incomplete: {path}")
    if verify_hash and metadata.sha256 and _sha256_file(path) != metadata.sha256:
        raise TokenArtifactError(f"Token artifact hash mismatch: {path}")

    count = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if tuple(reader.fieldnames or ()) != TOKEN_ARTIFACT_COLUMNS:
            raise TokenArtifactError(f"Token artifact header does not match schema: {path}")
        for line_number, row in enumerate(reader, start=2):
            count += 1
            yield _row_to_record(row, path=path, line_number=line_number)
    if count != metadata.row_count:
        raise TokenArtifactError(
            f"Token artifact row_count mismatch: {path} expected {metadata.row_count}, found {count}"
        )


def validate_token_artifact(path: Path) -> TokenArtifactMetadata:
    metadata = read_token_artifact_metadata(path)
    for _record in read_token_records(path, verify_hash=True):
        pass
    return metadata
