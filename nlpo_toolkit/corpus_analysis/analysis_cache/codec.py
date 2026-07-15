from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Mapping

from ..analysis_records import NLPAnalysisRecord
from .constants import (
    ANALYSIS_BEHAVIOR_VERSION,
    ANALYSIS_CACHE_FORMAT,
    ANALYSIS_CACHE_SCHEMA_VERSION,
)
from .errors import AnalysisCacheError
from .models import AnalysisCacheMetadata, CacheObjectPaths


def encode_record(record: NLPAnalysisRecord) -> dict[str, object]:
    return asdict(record)


def _required_int(data: Mapping[str, object], key: str, path: Path, line: int) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}") from exc


def _optional_int(
    data: Mapping[str, object], key: str, path: Path, line: int
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}")
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}") from exc


def decode_record(
    data: Mapping[str, object], *, path: Path, line_number: int
) -> NLPAnalysisRecord:
    token = data.get("token")
    sentence = data.get("sentence")
    if not isinstance(token, str) or not isinstance(sentence, str):
        raise AnalysisCacheError(f"Invalid token analysis record at {path}:{line_number}")
    lemma = data.get("lemma")
    upos = data.get("upos")
    if lemma is not None and not isinstance(lemma, str):
        raise AnalysisCacheError(f"Invalid lemma in token analysis record at {path}:{line_number}")
    if upos is not None and not isinstance(upos, str):
        raise AnalysisCacheError(f"Invalid upos in token analysis record at {path}:{line_number}")
    return NLPAnalysisRecord(
        chunk_index=_required_int(data, "chunk_index", path, line_number),
        sentence_index=_required_int(data, "sentence_index", path, line_number),
        token_index=_required_int(data, "token_index", path, line_number),
        global_token_index=_required_int(data, "global_token_index", path, line_number),
        char_start_in_chunk=_optional_int(data, "char_start_in_chunk", path, line_number),
        char_end_in_chunk=_optional_int(data, "char_end_in_chunk", path, line_number),
        char_start_in_text=_optional_int(data, "char_start_in_text", path, line_number),
        char_end_in_text=_optional_int(data, "char_end_in_text", path, line_number),
        sentence=sentence,
        token=token,
        lemma=lemma,
        upos=upos,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cache_metadata(path: Path) -> AnalysisCacheMetadata:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AnalysisCacheError(f"Invalid analysis cache metadata: {path}") from exc
    if not isinstance(raw, dict):
        raise AnalysisCacheError(f"Invalid analysis cache metadata: {path}")
    data: Mapping[str, object] = raw
    if data.get("format") != ANALYSIS_CACHE_FORMAT:
        raise AnalysisCacheError(f"Invalid analysis cache format: {path}")
    if data.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
        raise AnalysisCacheError(f"Unsupported analysis cache schema version: {path}")
    if data.get("behavior_version") != ANALYSIS_BEHAVIOR_VERSION:
        raise AnalysisCacheError(f"Unsupported analysis cache behavior version: {path}")
    if data.get("complete") is not True:
        raise AnalysisCacheError(f"Incomplete analysis cache object: {path}")
    fingerprint = data.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise AnalysisCacheError(f"Invalid analysis cache fingerprint: {path}")
    try:
        return AnalysisCacheMetadata(
            format=str(data["format"]),
            schema_version=int(data["schema_version"]),  # type: ignore[arg-type]
            behavior_version=int(data["behavior_version"]),  # type: ignore[arg-type]
            complete=True,
            cache_key=str(data["cache_key"]),
            created_at=str(data["created_at"]),
            prepared_text_sha256=str(data["prepared_text_sha256"]),
            prepared_text_length=int(data["prepared_text_length"]),  # type: ignore[arg-type]
            record_count=int(data["record_count"]),  # type: ignore[arg-type]
            payload_path=str(data["payload_path"]),
            payload_sha256=str(data["payload_sha256"]),
            payload_size_bytes=int(data["payload_size_bytes"]),  # type: ignore[arg-type]
            fingerprint=fingerprint,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisCacheError(f"Invalid analysis cache metadata: {path}") from exc


def read_analysis_records(paths: CacheObjectPaths) -> Iterator[NLPAnalysisRecord]:
    metadata = read_cache_metadata(paths.metadata)
    if not paths.payload.exists():
        raise AnalysisCacheError(f"Analysis cache payload not found: {paths.payload}")
    if metadata.payload_sha256 and sha256_file(paths.payload) != metadata.payload_sha256:
        raise AnalysisCacheError(f"Analysis cache payload hash mismatch: {paths.payload}")
    count = 0
    with paths.payload.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                raw: object = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnalysisCacheError(
                    f"Invalid analysis cache JSON at {paths.payload}:{line_number}"
                ) from exc
            if not isinstance(raw, dict):
                raise AnalysisCacheError(
                    f"Invalid analysis cache record at {paths.payload}:{line_number}"
                )
            count += 1
            yield decode_record(raw, path=paths.payload, line_number=line_number)
    if count != metadata.record_count:
        raise AnalysisCacheError(
            f"Analysis cache record count mismatch: {paths.payload} "
            f"expected {metadata.record_count}, found {count}"
        )


def validate_cache_object(paths: CacheObjectPaths) -> AnalysisCacheMetadata:
    metadata = read_cache_metadata(paths.metadata)
    for _record in read_analysis_records(paths):
        pass
    return metadata
