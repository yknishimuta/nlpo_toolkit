from __future__ import annotations

import csv
import os
import tempfile
from enum import Enum, auto
from pathlib import Path
from typing import TextIO

from ..analysis_records import TokenRecord
from .codec import encode_token_record
from .errors import TokenArtifactWriterStateError
from .integrity import file_size, sha256_file
from .paths import token_artifact_metadata_path
from .schema import (
    TOKEN_ARTIFACT_COLUMNS,
    TOKEN_ARTIFACT_DELIMITER,
    TOKEN_ARTIFACT_ENCODING,
    TokenArtifactDescriptor,
    TokenArtifactMetadata,
    metadata_to_json,
)


class _WriterState(Enum):
    NEW = auto()
    OPEN = auto()
    CLOSED = auto()


def _temporary_path(path: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(name)


class TokenArtifactWriter:
    def __init__(
        self,
        artifact_path: Path,
        *,
        metadata_path: Path,
        descriptor: TokenArtifactDescriptor,
    ) -> None:
        self.path = Path(artifact_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        expected = token_artifact_metadata_path(self.path).resolve()
        if self.metadata_path != expected:
            raise TokenArtifactWriterStateError(
                f"Token artifact metadata path does not follow protocol: "
                f"expected={expected}; actual={self.metadata_path}"
            )
        self.descriptor = descriptor
        self._state = _WriterState.NEW
        self._artifact_tmp: Path | None = None
        self._metadata_tmp: Path | None = None
        self._stream: TextIO | None = None
        self._writer: csv.DictWriter[str] | None = None
        self.row_count = 0
        self.included_row_count = 0
        self.excluded_row_count = 0
        self.final_metadata: TokenArtifactMetadata | None = None

    def __enter__(self) -> "TokenArtifactWriter":
        if self._state is not _WriterState.NEW:
            raise TokenArtifactWriterStateError("TokenArtifactWriter cannot be reused")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_tmp = _temporary_path(self.path)
        self._metadata_tmp = _temporary_path(self.metadata_path)
        self._stream = self._artifact_tmp.open(
            "w", encoding=TOKEN_ARTIFACT_ENCODING, newline=""
        )
        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=TOKEN_ARTIFACT_COLUMNS,
            delimiter=TOKEN_ARTIFACT_DELIMITER,
            lineterminator="\n",
        )
        self._writer.writeheader()
        self._state = _WriterState.OPEN
        return self

    def write(self, record: TokenRecord) -> None:
        if self._state is not _WriterState.OPEN or self._writer is None:
            raise TokenArtifactWriterStateError("TokenArtifactWriter is not open")
        self._writer.writerow(encode_token_record(record))
        self.row_count += 1
        if record.included:
            self.included_row_count += 1
        else:
            self.excluded_row_count += 1

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            if self._stream is not None:
                self._stream.flush()
                self._stream.close()
            if exc_type is not None:
                return
            assert self._artifact_tmp is not None
            assert self._metadata_tmp is not None
            metadata = TokenArtifactMetadata(
                complete=True,
                row_count=self.row_count,
                included_row_count=self.included_row_count,
                excluded_row_count=self.excluded_row_count,
                group=self.descriptor.group,
                source_files=self.descriptor.source_files,
                analysis_unit=self.descriptor.analysis_unit,
                upos_targets=self.descriptor.upos_targets,
                nlp=self.descriptor.nlp,
                filters=self.descriptor.filters,
                artifact_path=str(self.path),
                sha256=sha256_file(self._artifact_tmp),
                size_bytes=file_size(self._artifact_tmp),
            )
            self._metadata_tmp.write_text(
                metadata_to_json(metadata), encoding=TOKEN_ARTIFACT_ENCODING
            )
            self._artifact_tmp.replace(self.path)
            self._metadata_tmp.replace(self.metadata_path)
            self.final_metadata = metadata
        finally:
            self._state = _WriterState.CLOSED
            self._writer = None
            self._stream = None
            self._cleanup_temporary_files()

    def _cleanup_temporary_files(self) -> None:
        for path in (self._artifact_tmp, self._metadata_tmp):
            if path is not None:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
