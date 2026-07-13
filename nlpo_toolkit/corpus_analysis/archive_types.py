"""Data contracts shared by archive ports and the archive implementation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ArchiveOptions:
    run_name: str | None = None
    runs_dir: Path | str = Path("runs")
    include_cleaned: bool = False
    include_input: bool = False
    command_line: tuple[str, ...] = ()
    created_at: datetime | None = None


@dataclass(frozen=True)
class RunArchiveResult:
    run_dir: Path
    copied_output_count: int
    copied_trace_count: int
    copied_input_count: int
    copied_cleaned_count: int
    copied_config_count: int
