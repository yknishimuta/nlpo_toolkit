from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator, TypedDict

from nlpo_toolkit.serialization.types import JsonObject
from nlpo_toolkit.nlp.contracts import UDMorphFeature

from ..analysis_records import NLPAnalysisRecord
from .constants import (
    ANALYSIS_BEHAVIOR_VERSION,
    ANALYSIS_CACHE_FORMAT,
    ANALYSIS_CACHE_SCHEMA_VERSION,
)
from .errors import AnalysisCacheError
from .models import AnalysisCacheMetadata, AnalysisFingerprint, CacheObjectPaths


def analysis_fingerprint_to_json_value(
    fingerprint: AnalysisFingerprint,
) -> JsonObject:
    return {
        "backend": fingerprint.backend,
        "language": fingerprint.language,
        "processors": list(fingerprint.processors),
        "chunk_size": fingerprint.chunk_size,
        "chunk_strategy": fingerprint.chunk_strategy,
        "model": fingerprint.model,
        "package": dict(fingerprint.package)
        if isinstance(fingerprint.package, Mapping)
        else fingerprint.package,
        "model_revision": fingerprint.model_revision,
        "backend_version": fingerprint.backend_version,
        "adapter_version": fingerprint.adapter_version,
        "device": fingerprint.device,
    }


def analysis_cache_metadata_to_json_value(
    metadata: AnalysisCacheMetadata,
) -> JsonObject:
    return {
        "format": metadata.format,
        "schema_version": metadata.schema_version,
        "behavior_version": metadata.behavior_version,
        "complete": metadata.complete,
        "cache_key": metadata.cache_key,
        "created_at": metadata.created_at,
        "prepared_text_sha256": metadata.prepared_text_sha256,
        "prepared_text_length": metadata.prepared_text_length,
        "record_count": metadata.record_count,
        "payload_path": metadata.payload_path,
        "payload_sha256": metadata.payload_sha256,
        "payload_size_bytes": metadata.payload_size_bytes,
        "fingerprint": analysis_fingerprint_to_json_value(metadata.fingerprint),
    }


class AnalysisRecordPayload(TypedDict):
    chunk_index: int
    sentence_index: int
    token_index: int
    global_token_index: int
    char_start_in_chunk: int | None
    char_end_in_chunk: int | None
    char_start_in_text: int | None
    char_end_in_text: int | None
    sentence: str
    token: str
    lemma: str | None
    upos: str | None
    morphology: list[dict[str, str]]


def encode_record(record: NLPAnalysisRecord) -> AnalysisRecordPayload:
    return {
        "chunk_index": record.chunk_index,
        "sentence_index": record.sentence_index,
        "token_index": record.token_index,
        "global_token_index": record.global_token_index,
        "char_start_in_chunk": record.char_start_in_chunk,
        "char_end_in_chunk": record.char_end_in_chunk,
        "char_start_in_text": record.char_start_in_text,
        "char_end_in_text": record.char_end_in_text,
        "sentence": record.sentence,
        "token": record.token,
        "lemma": record.lemma,
        "upos": record.upos,
        "morphology": [
            {"attribute": item.attribute, "value": item.value}
            for item in record.morphology
        ],
    }


def _required_int(data: dict[object, object], key: str, path: Path, line: int) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}")
    if not isinstance(value, int):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}")
    return value


def _optional_int(
    data: dict[object, object], key: str, path: Path, line: int
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}")
    if not isinstance(value, int):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line}")
    return value


def parse_analysis_record_payload(
    raw: object,
    *,
    path: Path,
    line_number: int,
) -> AnalysisRecordPayload:
    if not isinstance(raw, dict):
        raise AnalysisCacheError(
            f"Invalid analysis cache record at {path}:{line_number}"
        )
    expected = set(AnalysisRecordPayload.__required_keys__)
    if set(raw) != expected or not all(isinstance(key, str) for key in raw):
        raise AnalysisCacheError(
            f"Invalid analysis cache record fields at {path}:{line_number}"
        )
    integers = {
        key: _required_int(raw, key, path, line_number)
        for key in (
            "chunk_index",
            "sentence_index",
            "token_index",
            "global_token_index",
        )
    }
    optional = {
        key: _optional_int(raw, key, path, line_number)
        for key in (
            "char_start_in_chunk",
            "char_end_in_chunk",
            "char_start_in_text",
            "char_end_in_text",
        )
    }
    sentence, token = raw["sentence"], raw["token"]
    lemma, upos = raw["lemma"], raw["upos"]
    if not isinstance(sentence, str) or not isinstance(token, str):
        raise AnalysisCacheError(
            f"Invalid token analysis strings at {path}:{line_number}"
        )
    if lemma is not None and not isinstance(lemma, str):
        raise AnalysisCacheError(
            f"Invalid lemma in token analysis record at {path}:{line_number}"
        )
    if upos is not None and not isinstance(upos, str):
        raise AnalysisCacheError(
            f"Invalid upos in token analysis record at {path}:{line_number}"
        )
    morphology_raw = raw["morphology"]
    if not isinstance(morphology_raw, list):
        raise AnalysisCacheError(
            f"Invalid morphology in token analysis record at {path}:{line_number}"
        )
    morphology: list[dict[str, str]] = []
    try:
        for item in morphology_raw:
            if not isinstance(item, dict) or set(item) != {"attribute", "value"}:
                raise AnalysisCacheError(
                    f"Invalid morphology item at {path}:{line_number}"
                )
            attribute, value = item["attribute"], item["value"]
            if not isinstance(attribute, str) or not isinstance(value, str):
                raise AnalysisCacheError(
                    f"Invalid morphology strings at {path}:{line_number}"
                )
            feature = UDMorphFeature(attribute, value)
            morphology.append({"attribute": feature.attribute, "value": feature.value})
        canonical = tuple(
            sorted(
                UDMorphFeature(item["attribute"], item["value"]) for item in morphology
            )
        )
        if len({item.attribute for item in canonical}) != len(canonical):
            raise AnalysisCacheError(
                f"Duplicate morphology attribute at {path}:{line_number}"
            )
        morphology = [
            {"attribute": item.attribute, "value": item.value} for item in canonical
        ]
    except (TypeError, ValueError) as exc:
        raise AnalysisCacheError(f"Invalid morphology at {path}:{line_number}") from exc
    return {
        "chunk_index": integers["chunk_index"],
        "sentence_index": integers["sentence_index"],
        "token_index": integers["token_index"],
        "global_token_index": integers["global_token_index"],
        "char_start_in_chunk": optional["char_start_in_chunk"],
        "char_end_in_chunk": optional["char_end_in_chunk"],
        "char_start_in_text": optional["char_start_in_text"],
        "char_end_in_text": optional["char_end_in_text"],
        "sentence": sentence,
        "token": token,
        "lemma": lemma,
        "upos": upos,
        "morphology": morphology,
    }


def decode_record(
    data: AnalysisRecordPayload, *, path: Path, line_number: int
) -> NLPAnalysisRecord:
    token = data.get("token")
    sentence = data.get("sentence")
    if not isinstance(token, str) or not isinstance(sentence, str):
        raise AnalysisCacheError(
            f"Invalid token analysis record at {path}:{line_number}"
        )
    lemma = data.get("lemma")
    upos = data.get("upos")
    if lemma is not None and not isinstance(lemma, str):
        raise AnalysisCacheError(
            f"Invalid lemma in token analysis record at {path}:{line_number}"
        )
    if upos is not None and not isinstance(upos, str):
        raise AnalysisCacheError(
            f"Invalid upos in token analysis record at {path}:{line_number}"
        )
    return NLPAnalysisRecord(
        chunk_index=data["chunk_index"],
        sentence_index=data["sentence_index"],
        token_index=data["token_index"],
        global_token_index=data["global_token_index"],
        char_start_in_chunk=data["char_start_in_chunk"],
        char_end_in_chunk=data["char_end_in_chunk"],
        char_start_in_text=data["char_start_in_text"],
        char_end_in_text=data["char_end_in_text"],
        sentence=sentence,
        token=token,
        lemma=lemma,
        upos=upos,
        morphology=tuple(
            UDMorphFeature(item["attribute"], item["value"])
            for item in data["morphology"]
        ),
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
    data = raw
    if data.get("format") != ANALYSIS_CACHE_FORMAT:
        raise AnalysisCacheError(f"Invalid analysis cache format: {path}")
    if data.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
        raise AnalysisCacheError(f"Unsupported analysis cache schema version: {path}")
    if data.get("behavior_version") != ANALYSIS_BEHAVIOR_VERSION:
        raise AnalysisCacheError(f"Unsupported analysis cache behavior version: {path}")
    if data.get("complete") is not True:
        raise AnalysisCacheError(f"Incomplete analysis cache object: {path}")
    fingerprint_raw = data.get("fingerprint")
    if not isinstance(fingerprint_raw, dict):
        raise AnalysisCacheError(f"Invalid analysis cache fingerprint: {path}")

    def required_string(key: str) -> str:
        value = data.get(key)
        if not isinstance(value, str):
            raise AnalysisCacheError(
                f"Invalid {key} in analysis cache metadata: {path}"
            )
        return value

    def required_integer(key: str) -> int:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise AnalysisCacheError(
                f"Invalid {key} in analysis cache metadata: {path}"
            )
        return value

    def fingerprint_string(key: str, *, optional: bool = False) -> str | None:
        value = fingerprint_raw.get(key)
        if optional and value is None:
            return None
        if not isinstance(value, str):
            raise AnalysisCacheError(f"Invalid fingerprint {key}: {path}")
        return value

    processors_raw = fingerprint_raw.get("processors")
    package_raw = fingerprint_raw.get("package")
    if not isinstance(processors_raw, list) or not all(
        isinstance(value, str) for value in processors_raw
    ):
        raise AnalysisCacheError(f"Invalid fingerprint processors: {path}")
    package: str | dict[str, str] | None
    if package_raw is None or isinstance(package_raw, str):
        package = package_raw
    elif isinstance(package_raw, dict) and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in package_raw.items()
    ):
        package = dict(package_raw)
    else:
        raise AnalysisCacheError(f"Invalid fingerprint package: {path}")
    chunk_size = fingerprint_raw.get("chunk_size")
    adapter_version = fingerprint_raw.get("adapter_version")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise AnalysisCacheError(f"Invalid fingerprint chunk_size: {path}")
    if isinstance(adapter_version, bool) or not isinstance(adapter_version, int):
        raise AnalysisCacheError(f"Invalid fingerprint adapter_version: {path}")
    fingerprint = AnalysisFingerprint(
        backend=fingerprint_string("backend") or "",
        language=fingerprint_string("language") or "",
        processors=tuple(processors_raw),
        chunk_size=chunk_size,
        chunk_strategy=fingerprint_string("chunk_strategy") or "",
        model=fingerprint_string("model", optional=True),
        package=package,
        model_revision=fingerprint_string("model_revision", optional=True),
        backend_version=fingerprint_string("backend_version", optional=True),
        adapter_version=adapter_version,
        device=fingerprint_string("device", optional=True),
    )
    try:
        return AnalysisCacheMetadata(
            format=required_string("format"),
            schema_version=required_integer("schema_version"),
            behavior_version=required_integer("behavior_version"),
            complete=True,
            cache_key=required_string("cache_key"),
            created_at=required_string("created_at"),
            prepared_text_sha256=required_string("prepared_text_sha256"),
            prepared_text_length=required_integer("prepared_text_length"),
            record_count=required_integer("record_count"),
            payload_path=required_string("payload_path"),
            payload_sha256=required_string("payload_sha256"),
            payload_size_bytes=required_integer("payload_size_bytes"),
            fingerprint=fingerprint,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisCacheError(f"Invalid analysis cache metadata: {path}") from exc


def read_analysis_records(paths: CacheObjectPaths) -> Iterator[NLPAnalysisRecord]:
    metadata = read_cache_metadata(paths.metadata)
    if not paths.payload.exists():
        raise AnalysisCacheError(f"Analysis cache payload not found: {paths.payload}")
    if (
        metadata.payload_sha256
        and sha256_file(paths.payload) != metadata.payload_sha256
    ):
        raise AnalysisCacheError(
            f"Analysis cache payload hash mismatch: {paths.payload}"
        )
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
            payload = parse_analysis_record_payload(
                raw,
                path=paths.payload,
                line_number=line_number,
            )
            yield decode_record(payload, path=paths.payload, line_number=line_number)
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
