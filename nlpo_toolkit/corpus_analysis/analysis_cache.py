from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from .cache_storage import (
    PruneReport,
    acquire_cache_lock,
    release_cache_lock,
)
from .analysis_records import NLPAnalysisRecord


ANALYSIS_CACHE_FORMAT = "nlpo-analysis-cache"
ANALYSIS_CACHE_SCHEMA_VERSION = 1
ANALYSIS_BEHAVIOR_VERSION = 1


class AnalysisCacheError(ValueError):
    pass


@dataclass(frozen=True)
class AnalysisFingerprint:
    backend: str
    language: str
    processors: tuple[str, ...]
    chunk_size: int
    chunk_strategy: str
    model: str | None = None
    package: object | None = None
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
class AnalysisCacheGroupResult:
    group: str
    status: str
    cache_key: str
    record_count: int


@dataclass
class AnalysisCacheRunStats:
    enabled: bool
    directory: str
    hits: int = 0
    misses: int = 0
    objects_written: int = 0
    records_read: int = 0
    records_written: int = 0
    groups: list[AnalysisCacheGroupResult] | None = None

    def __post_init__(self) -> None:
        if self.groups is None:
            self.groups = []

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "directory": self.directory,
            "hits": self.hits,
            "misses": self.misses,
            "objects_written": self.objects_written,
            "records_read": self.records_read,
            "records_written": self.records_written,
            "groups": [asdict(group) for group in (self.groups or [])],
        }


def prepared_text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_analysis_cache_key(
    *,
    prepared_text_sha256: str,
    fingerprint: AnalysisFingerprint,
) -> str:
    payload = {
        "prepared_text_sha256": prepared_text_sha256,
        "schema_version": ANALYSIS_CACHE_SCHEMA_VERSION,
        "behavior_version": ANALYSIS_BEHAVIOR_VERSION,
        "fingerprint": asdict(fingerprint),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_object_path(cache_dir: Path, cache_key: str) -> Path:
    return Path(cache_dir) / "objects" / cache_key[:2] / f"{cache_key}.jsonl"


def cache_metadata_path(payload_path: Path) -> Path:
    return payload_path.with_name(f"{payload_path.stem}.meta.json")


def cache_lock_path(cache_dir: Path, cache_key: str) -> Path:
    return Path(cache_dir) / "locks" / cache_key[:2] / f"{cache_key}.lock"


def _record_to_json_obj(record: NLPAnalysisRecord) -> dict[str, object]:
    return asdict(record)


def _required_int(data: Mapping[str, object], key: str, path: Path, line_number: int) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line_number}")
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line_number}") from exc


def _optional_int(data: Mapping[str, object], key: str, path: Path, line_number: int) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line_number}")
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise AnalysisCacheError(f"Invalid integer in {key} at {path}:{line_number}") from exc


def _record_from_json_obj(data: Mapping[str, object], *, path: Path, line_number: int) -> NLPAnalysisRecord:
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp = Path(tf.name)
    os.replace(str(tmp), str(path))


class AnalysisCacheWriter:
    def __init__(
        self,
        *,
        payload_path: Path,
        metadata_path: Path,
        cache_key: str,
        prepared_text_sha256: str,
        prepared_text_length: int,
        fingerprint: AnalysisFingerprint,
    ) -> None:
        self.payload_path = payload_path
        self.metadata_path = metadata_path
        self.cache_key = cache_key
        self.prepared_text_sha256 = prepared_text_sha256
        self.prepared_text_length = prepared_text_length
        self.fingerprint = fingerprint
        self._tmp_payload = payload_path.with_name(f"{payload_path.name}.tmp")
        self._tmp_metadata = metadata_path.with_name(f"{metadata_path.name}.tmp")
        self._file: Any = None
        self.record_count = 0

    def __enter__(self) -> "AnalysisCacheWriter":
        self.payload_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._tmp_payload.open("w", encoding="utf-8", newline="")
        return self

    def write(self, record: NLPAnalysisRecord) -> None:
        if self._file is None:
            raise AnalysisCacheError("AnalysisCacheWriter is not open")
        self._file.write(
            json.dumps(
                _record_to_json_obj(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        self.record_count += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
            if exc_type is not None:
                self._cleanup()
                return
            self._tmp_payload.replace(self.payload_path)
            metadata = AnalysisCacheMetadata(
                format=ANALYSIS_CACHE_FORMAT,
                schema_version=ANALYSIS_CACHE_SCHEMA_VERSION,
                behavior_version=ANALYSIS_BEHAVIOR_VERSION,
                complete=True,
                cache_key=self.cache_key,
                created_at=datetime.now(timezone.utc).isoformat(),
                prepared_text_sha256=self.prepared_text_sha256,
                prepared_text_length=self.prepared_text_length,
                record_count=self.record_count,
                payload_path=self.payload_path.name,
                payload_sha256=_sha256_file(self.payload_path),
                payload_size_bytes=self.payload_path.stat().st_size,
                fingerprint=asdict(self.fingerprint),
            )
            _atomic_write_text(
                self._tmp_metadata,
                json.dumps(asdict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            self._tmp_metadata.replace(self.metadata_path)
        except Exception:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        for path in (self._tmp_payload, self._tmp_metadata):
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def read_cache_metadata(path: Path) -> AnalysisCacheMetadata:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AnalysisCacheError(f"Invalid analysis cache metadata: {path}") from exc
    if data.get("format") != ANALYSIS_CACHE_FORMAT:
        raise AnalysisCacheError(f"Invalid analysis cache format: {path}")
    if data.get("schema_version") != ANALYSIS_CACHE_SCHEMA_VERSION:
        raise AnalysisCacheError(f"Unsupported analysis cache schema version: {path}")
    if data.get("behavior_version") != ANALYSIS_BEHAVIOR_VERSION:
        raise AnalysisCacheError(f"Unsupported analysis cache behavior version: {path}")
    if not data.get("complete"):
        raise AnalysisCacheError(f"Incomplete analysis cache object: {path}")
    fingerprint = data.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise AnalysisCacheError(f"Invalid analysis cache fingerprint: {path}")
    return AnalysisCacheMetadata(
        format=str(data["format"]),
        schema_version=int(data["schema_version"]),
        behavior_version=int(data["behavior_version"]),
        complete=bool(data["complete"]),
        cache_key=str(data["cache_key"]),
        created_at=str(data["created_at"]),
        prepared_text_sha256=str(data["prepared_text_sha256"]),
        prepared_text_length=int(data["prepared_text_length"]),
        record_count=int(data["record_count"]),
        payload_path=str(data["payload_path"]),
        payload_sha256=str(data["payload_sha256"]),
        payload_size_bytes=int(data["payload_size_bytes"]),
        fingerprint=fingerprint,
    )


def read_analysis_records(payload_path: Path, metadata_path: Path) -> Iterator[NLPAnalysisRecord]:
    metadata = read_cache_metadata(metadata_path)
    if not payload_path.exists():
        raise AnalysisCacheError(f"Analysis cache payload not found: {payload_path}")
    if metadata.payload_sha256 and _sha256_file(payload_path) != metadata.payload_sha256:
        raise AnalysisCacheError(f"Analysis cache payload hash mismatch: {payload_path}")
    count = 0
    with payload_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except Exception as exc:
                raise AnalysisCacheError(f"Invalid analysis cache JSON at {payload_path}:{line_number}") from exc
            if not isinstance(data, dict):
                raise AnalysisCacheError(f"Invalid analysis cache record at {payload_path}:{line_number}")
            count += 1
            yield _record_from_json_obj(data, path=payload_path, line_number=line_number)
    if count != metadata.record_count:
        raise AnalysisCacheError(
            f"Analysis cache record count mismatch: {payload_path} expected {metadata.record_count}, found {count}"
        )


def validate_cache_object(payload_path: Path, metadata_path: Path) -> AnalysisCacheMetadata:
    metadata = read_cache_metadata(metadata_path)
    for _record in read_analysis_records(payload_path, metadata_path):
        pass
    return metadata


def get_or_compute_analysis_records(
    *,
    cache_dir: Path,
    cache_key: str,
    prepared_text_sha256: str,
    prepared_text_length: int,
    fingerprint: AnalysisFingerprint,
    compute_records: Callable[[], Iterable[NLPAnalysisRecord]],
    lock_timeout_sec: float = 300.0,
) -> tuple[Iterator[NLPAnalysisRecord], str, Path, Path]:
    cache_dir = Path(cache_dir).resolve()
    payload_path = cache_object_path(cache_dir, cache_key)
    metadata_path = cache_metadata_path(payload_path)

    def _hit_iterator() -> Iterator[NLPAnalysisRecord]:
        yield from read_analysis_records(payload_path, metadata_path)

    if payload_path.exists() and metadata_path.exists():
        try:
            validate_cache_object(payload_path, metadata_path)
            return _hit_iterator(), "hit", payload_path, metadata_path
        except AnalysisCacheError:
            pass

    lock_path = cache_lock_path(cache_dir, cache_key)
    acquire_cache_lock(lock_path, timeout_sec=lock_timeout_sec)
    if payload_path.exists() and metadata_path.exists():
        try:
            validate_cache_object(payload_path, metadata_path)

            def _locked_hit_iterator() -> Iterator[NLPAnalysisRecord]:
                try:
                    yield from read_analysis_records(payload_path, metadata_path)
                finally:
                    release_cache_lock(lock_path)

            return _locked_hit_iterator(), "hit", payload_path, metadata_path
        except AnalysisCacheError:
            pass

    def _miss_iterator() -> Iterator[NLPAnalysisRecord]:
        try:
            with AnalysisCacheWriter(
                payload_path=payload_path,
                metadata_path=metadata_path,
                cache_key=cache_key,
                prepared_text_sha256=prepared_text_sha256,
                prepared_text_length=prepared_text_length,
                fingerprint=fingerprint,
            ) as writer:
                for record in compute_records():
                    writer.write(record)
                    yield record
        finally:
            release_cache_lock(lock_path)

    return _miss_iterator(), "miss", payload_path, metadata_path


def prune_analysis_cache(
    cache_dir: Path,
    *,
    keep_days: int = 30,
    keep_objects: int = 50_000,
    lock_ttl_sec: int = 3600,
    verbose: bool = False,
) -> PruneReport:
    cache_dir = Path(cache_dir).resolve()
    objects_dir = cache_dir / "objects"
    locks_dir = cache_dir / "locks"
    import time

    now = time.time()
    cutoff_ts = now - (keep_days * 86400)
    removed_objects = 0
    removed_locks = 0
    removed_empty_dirs = 0
    bytes_freed = 0

    if locks_dir.exists():
        for path in locks_dir.rglob("*.lock"):
            try:
                st = path.stat()
                if (now - st.st_mtime) > lock_ttl_sec:
                    bytes_freed += st.st_size
                    path.unlink(missing_ok=True)
                    removed_locks += 1
            except OSError:
                pass

    payloads = sorted(
        (p for p in objects_dir.rglob("*.jsonl") if p.is_file()) if objects_dir.exists() else [],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    survivors = set(payloads[: max(0, int(keep_objects))])
    for payload in payloads:
        try:
            st = payload.stat()
        except OSError:
            continue
        if payload in survivors:
            continue
        if st.st_mtime >= cutoff_ts:
            continue
        meta = cache_metadata_path(payload)
        for path in (payload, meta):
            try:
                bytes_freed += path.stat().st_size
                path.unlink(missing_ok=True)
            except OSError:
                pass
        removed_objects += 1

    for base in (objects_dir, locks_dir):
        if not base.exists():
            continue
        dirs = sorted([d for d in base.rglob("*") if d.is_dir()], key=lambda d: len(str(d)), reverse=True)
        for directory in dirs:
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
                    removed_empty_dirs += 1
            except OSError:
                pass

    return PruneReport(
        removed_objects=removed_objects,
        removed_locks=removed_locks,
        removed_empty_dirs=removed_empty_dirs,
        bytes_freed=bytes_freed,
    )
