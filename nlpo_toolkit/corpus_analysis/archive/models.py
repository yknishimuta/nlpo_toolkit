from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Mapping

from nlpo_toolkit.immutable_collections import freeze_tuple_mapping

from ..config_references import ConfigFileReference


def _validate_relative(path: Path, field: str) -> None:
    if path.is_absolute() or ".." in path.parts or path in {Path(""), Path(".")}:
        raise ValueError(f"{field} must be a safe relative path")


@dataclass(frozen=True)
class ArchiveCopySource:
    source_path: Path
    destination_relative_path: Path

    def __post_init__(self) -> None:
        if not self.source_path.is_absolute():
            raise ValueError("source_path must be absolute")
        _validate_relative(self.destination_relative_path, "destination_relative_path")


@dataclass(frozen=True)
class ArchivedFile:
    source_path: Path
    archive_relative_path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.source_path.is_absolute():
            raise ValueError("ArchivedFile source_path must be absolute")
        _validate_relative(self.archive_relative_path, "archive_relative_path")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


@dataclass(frozen=True)
class SourceFileMetadata:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ExternalReferenceMetadata:
    kind: str
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ArchiveInventory:
    project_root: Path
    archive_directory: Path
    run_name: str
    creation_time: datetime
    command_line: tuple[str, ...]
    config_path: Path
    output_dir: Path
    groups_files: Mapping[str, tuple[Path, ...]]
    output_sources: tuple[ArchiveCopySource, ...]
    trace_sources: tuple[ArchiveCopySource, ...]
    config_snapshot_sources: tuple[ArchiveCopySource, ...]
    input_sources: tuple[ArchiveCopySource, ...]
    cleaned_sources: tuple[ArchiveCopySource, ...]
    input_files: tuple[Path, ...]
    cleaned_files: tuple[Path, ...]
    metadata_only_references: tuple[ConfigFileReference, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_line", tuple(self.command_line))
        object.__setattr__(self, "groups_files", freeze_tuple_mapping(self.groups_files))
        for field_name in (
            "output_sources", "trace_sources", "config_snapshot_sources",
            "input_sources", "cleaned_sources", "input_files", "cleaned_files",
            "metadata_only_references",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))


@dataclass(frozen=True)
class ArchiveCopyResult:
    outputs: tuple[ArchivedFile, ...] = ()
    traces: tuple[ArchivedFile, ...] = ()
    config_snapshots: tuple[ArchivedFile, ...] = ()
    inputs: tuple[ArchivedFile, ...] = ()
    cleaned: tuple[ArchivedFile, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("outputs", "traces", "config_snapshots", "inputs", "cleaned"):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))


@dataclass(frozen=True)
class ExternalReferenceManifestEntry:
    kind: str
    path: Path
    exists: bool
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ArchiveGitReport:
    branch: str | None
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True)
class ArchiveManifest:
    run_name: str
    created_at: datetime
    command_line: tuple[str, ...]
    project_root: Path
    config_path: Path
    output_dir: Path
    git: ArchiveGitReport
    groups_files: Mapping[str, tuple[Path, ...]]
    input_files: tuple[SourceFileMetadata, ...]
    cleaned_files: tuple[SourceFileMetadata, ...]
    generated_outputs: tuple[SourceFileMetadata, ...]
    copied_outputs: tuple[ArchivedFile, ...]
    trace_files: tuple[ArchivedFile, ...]
    config_snapshot_files: tuple[ArchivedFile, ...]
    external_references: tuple[ExternalReferenceManifestEntry, ...]
    copied_input_files: tuple[ArchivedFile, ...]
    copied_cleaned_files: tuple[ArchivedFile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "command_line", tuple(self.command_line))
        object.__setattr__(self, "groups_files", freeze_tuple_mapping(self.groups_files))
        for field_name in (
            "input_files", "cleaned_files", "generated_outputs", "copied_outputs",
            "trace_files", "config_snapshot_files", "external_references",
            "copied_input_files", "copied_cleaned_files",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))
