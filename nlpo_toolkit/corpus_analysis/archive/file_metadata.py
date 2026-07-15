from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .models import ExternalReferenceMetadata


@dataclass(frozen=True)
class SourceFileMetadata:
    path: Path
    sha256: str
    size_bytes: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_file_metadata(path: Path) -> SourceFileMetadata:
    resolved = Path(path).resolve()
    return SourceFileMetadata(resolved, file_sha256(resolved), resolved.stat().st_size)


def read_source_files_metadata(paths: tuple[Path, ...]) -> tuple[SourceFileMetadata, ...]:
    return tuple(read_source_file_metadata(path) for path in paths)


def read_external_reference_metadata(kind: str, path: Path) -> ExternalReferenceMetadata:
    metadata = read_source_file_metadata(path)
    return ExternalReferenceMetadata(kind, metadata.path, metadata.sha256, metadata.size_bytes)
