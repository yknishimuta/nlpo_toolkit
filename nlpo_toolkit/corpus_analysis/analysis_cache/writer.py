from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TextIO

from ..analysis_records import NLPAnalysisRecord
from .codec import encode_record, sha256_file
from .constants import ANALYSIS_BEHAVIOR_VERSION, ANALYSIS_CACHE_FORMAT, ANALYSIS_CACHE_SCHEMA_VERSION
from .errors import AnalysisCacheError
from .models import AnalysisCacheMetadata, AnalysisFingerprint, CacheObjectPaths


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=str(path.parent), encoding="utf-8"
    ) as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


class AnalysisCacheWriter:
    def __init__(
        self,
        *,
        paths: CacheObjectPaths,
        cache_key: str,
        prepared_text_sha256: str,
        prepared_text_length: int,
        fingerprint: AnalysisFingerprint,
    ) -> None:
        self.paths = paths
        self.cache_key = cache_key
        self.prepared_text_sha256 = prepared_text_sha256
        self.prepared_text_length = prepared_text_length
        self.fingerprint = fingerprint
        self._tmp_payload = paths.payload.with_name(f"{paths.payload.name}.tmp")
        self._file: TextIO | None = None
        self.record_count = 0

    def __enter__(self) -> AnalysisCacheWriter:
        self.paths.payload.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._tmp_payload.open("w", encoding="utf-8", newline="")
        return self

    def write(self, record: NLPAnalysisRecord) -> None:
        if self._file is None:
            raise AnalysisCacheError("AnalysisCacheWriter is not open")
        self._file.write(
            json.dumps(
                encode_record(record), ensure_ascii=False, sort_keys=True,
                separators=(",", ":"), allow_nan=False,
            ) + "\n"
        )
        self.record_count += 1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._file is not None:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()
                self._file = None
            if exc_type is not None:
                self._cleanup()
                return
            self._tmp_payload.replace(self.paths.payload)
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
                payload_path=self.paths.payload.name,
                payload_sha256=sha256_file(self.paths.payload),
                payload_size_bytes=self.paths.payload.stat().st_size,
                fingerprint=asdict(self.fingerprint),
            )
            _atomic_write_text(
                self.paths.metadata,
                json.dumps(asdict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        except BaseException:
            self._cleanup()
            raise

    def _cleanup(self) -> None:
        try:
            self._tmp_payload.unlink()
        except FileNotFoundError:
            pass
