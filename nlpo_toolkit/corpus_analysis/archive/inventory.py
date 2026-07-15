from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

from ..archive_types import RunArchiveRequest
from ..config_references import ConfigArchivePolicy
from ..runner_types import RunResult
from .errors import RunArchiveError
from .models import ArchiveCopySource, ArchiveInventory


def _sanitize_run_name(name: str) -> str:
    raw = str(name).strip()
    if (
        not raw or Path(raw).is_absolute() or "/" in raw or "\\" in raw or ".." in raw
    ):
        raise ValueError("run name must be a safe directory name")
    sanitized = re.sub(r"\s+", "_", raw)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sanitized):
        raise ValueError("run name must contain only safe characters")
    return sanitized


def _validate_source(path: Path, category: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise RunArchiveError(
            f"Archive source file is missing or not a regular file: {category}: {resolved}"
        )
    return resolved


def _copy_sources(paths: tuple[Path, ...]) -> tuple[ArchiveCopySource, ...]:
    return tuple(ArchiveCopySource(path, Path(path.name)) for path in paths)


def collect_archive_inventory(
    *, run_result: RunResult, request: RunArchiveRequest
) -> ArchiveInventory:
    project_root = run_result.plan.project_root.resolve()
    creation_time = request.created_at or datetime.now().astimezone()
    run_name = _sanitize_run_name(
        request.run_name or creation_time.strftime("%Y%m%d-%H%M%S")
    )
    archive_root = request.archive_root
    if not archive_root.is_absolute():
        archive_root = (project_root / archive_root).resolve()
    archive_directory = archive_root / run_name
    if archive_directory.exists():
        raise RunArchiveError(f"Run archive already exists: {archive_directory}")

    outputs = tuple(_validate_source(path, "run output") for path in run_result.output_files)
    traces = tuple(_validate_source(path, "run trace") for path in run_result.trace_files)
    inputs = tuple(_validate_source(path, "run input") for path in run_result.input_files)
    cleaned = tuple(_validate_source(path, "cleaned run file") for path in run_result.cleaned_files)
    generated = tuple(
        _validate_source(path, "generated run output")
        for path in run_result.generated_outputs
    )

    snapshots = []
    metadata_only = []
    for reference in run_result.config_references:
        source = _validate_source(reference.source_path, f"config reference {reference.kind}")
        if reference.archive_policy is ConfigArchivePolicy.SNAPSHOT:
            if reference.snapshot_relative_path is None:
                raise RunArchiveError(
                    f"Snapshot config reference has no archive path: {reference.kind}: {source}"
                )
            snapshots.append(
                ArchiveCopySource(source, Path(reference.snapshot_relative_path))
            )
        elif reference.archive_policy is ConfigArchivePolicy.METADATA_ONLY:
            metadata_only.append(reference)

    cleaned_root = (
        run_result.plan.cleaned_dir.resolve()
        if run_result.plan.cleaned_dir is not None else project_root
    )
    input_sources = tuple(
        ArchiveCopySource(
            path,
            path.relative_to(project_root) if path.is_relative_to(project_root) else Path(path.name),
        )
        for path in inputs
    ) if request.include_input_files else ()
    cleaned_sources = tuple(
        ArchiveCopySource(
            path,
            path.relative_to(cleaned_root) if path.is_relative_to(cleaned_root) else Path(path.name),
        )
        for path in cleaned
    ) if request.include_cleaned_files else ()

    return ArchiveInventory(
        project_root=project_root,
        archive_directory=archive_directory,
        run_name=run_name,
        creation_time=creation_time,
        command_line=request.command_line or tuple(sys.argv),
        config_path=run_result.plan.config_path.resolve(),
        output_dir=run_result.plan.out_dir.resolve(),
        groups_files=run_result.groups_files,
        output_sources=_copy_sources(outputs),
        trace_sources=_copy_sources(traces),
        config_snapshot_sources=tuple(snapshots),
        input_sources=input_sources,
        cleaned_sources=cleaned_sources,
        input_files=inputs,
        cleaned_files=cleaned,
        generated_outputs=generated,
        metadata_only_references=tuple(metadata_only),
    )
