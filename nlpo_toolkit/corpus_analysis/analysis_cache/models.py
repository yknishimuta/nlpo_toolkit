from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping

from ..analysis_records import NLPAnalysisRecord
from .constants import ANALYSIS_BEHAVIOR_VERSION

CacheStatus = Literal["hit", "miss"]


@dataclass(frozen=True)
class AnalysisFingerprint:
    backend: str
    language: str
    processors: tuple[str, ...]
    chunk_size: int
    chunk_strategy: str
    model: str | None = None
    package: str | Mapping[str, str] | None = None
    model_revision: str | None = None
    backend_version: str | None = None
    adapter_version: int = ANALYSIS_BEHAVIOR_VERSION
    device: str | None = None


@dataclass(frozen=True)
class AnalysisCacheMetadata:
    format: str
    schema_version: int
    behavior_version: int
    complete: bool
    cache_key: str
    created_at: str
    prepared_text_sha256: str
    prepared_text_length: int
    record_count: int
    payload_path: str
    payload_sha256: str
    payload_size_bytes: int
    fingerprint: Mapping[str, object]


@dataclass(frozen=True)
class CacheObjectPaths:
    payload: Path
    metadata: Path
    lock: Path


@dataclass(frozen=True)
class AnalysisCacheRecordSource:
    records: Iterator[NLPAnalysisRecord]
    status: CacheStatus
    cache_key: str
    paths: CacheObjectPaths
