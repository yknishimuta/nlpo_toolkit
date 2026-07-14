"""Data contracts shared by archive ports and the archive implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunArchiveRequest:
    archive_root: Path = Path("runs")
    run_name: str | None = None
    include_input_files: bool = False
    include_cleaned_files: bool = False
    command_line: tuple[str, ...] = ()
    created_at: datetime | None = None


    def __post_init__(self) -> None:
        if not isinstance(self.archive_root, Path):
            raise TypeError("archive_root must be a Path")


@dataclass(frozen=True)
class ArchivedFileCounts:
    outputs: int = 0
    traces: int = 0
    inputs: int = 0
    cleaned: int = 0
    config_snapshots: int = 0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.outputs, self.traces, self.inputs, self.cleaned, self.config_snapshots)):
            raise ValueError("Archived file counts must be non-negative")


@dataclass(frozen=True)
class RunArchiveResult:
    archive_directory: Path
    copied_files: ArchivedFileCounts
