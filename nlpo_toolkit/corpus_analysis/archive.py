"""Run archive creation from an exact, in-memory run inventory."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .archive_types import ArchivedFileCounts, RunArchiveRequest, RunArchiveResult
from .config_references import ConfigArchivePolicy, ConfigFileReference
from .runner_types import RunResult

__all__ = [
    "ArchivedFileCounts",
    "RunArchiveError",
    "RunArchiveRequest",
    "RunArchiveResult",
    "create_run_archive",
]


class RunArchiveError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveCopySource:
    source_path: Path
    destination_relative_path: Path


@dataclass(frozen=True)
class ArchivedFile:
    source_path: Path
    archive_relative_path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.source_path.is_absolute():
            raise ValueError("ArchivedFile source_path must be absolute")
        if (
            self.archive_relative_path.is_absolute()
            or ".." in self.archive_relative_path.parts
        ):
            raise ValueError("archive_relative_path must be a safe relative path")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


def sanitize_run_name(name: str) -> str:
    raw = str(name).strip()
    if (
        not raw
        or Path(raw).is_absolute()
        or "/" in raw
        or "\\" in raw
        or ".." in raw
    ):
        raise ValueError("run name must be a safe directory name")
    sanitized = re.sub(r"\s+", "_", raw)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sanitized):
        raise ValueError("run name must contain only safe characters")
    return sanitized


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_archive_source_file(path: Path, kind: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists() or not resolved.is_file():
        raise RunArchiveError(
            f"Archive source file is missing or not a regular file: {kind}: {resolved}"
        )
    return resolved


def _allocate_unique_archive_path(
    *, destination_root: Path, requested_relative_path: Path, used: set[Path]
) -> Path:
    safe_relative_path = Path(
        *[
            part
            for part in requested_relative_path.parts
            if part not in {"", ".", ".."}
        ]
    )
    candidate = safe_relative_path
    index = 2
    while candidate in used or (destination_root / candidate).exists():
        candidate = safe_relative_path.with_name(
            f"{safe_relative_path.stem}_{index}{safe_relative_path.suffix}"
        )
        index += 1
    used.add(candidate)
    return destination_root / candidate


def _copy_files_into_archive(
    files: Iterable[ArchiveCopySource],
    *,
    destination_root: Path,
    archive_directory: Path,
) -> tuple[ArchivedFile, ...]:
    archived_files: list[ArchivedFile] = []
    used: set[Path] = set()
    for item in files:
        destination = _allocate_unique_archive_path(
            destination_root=destination_root,
            requested_relative_path=item.destination_relative_path,
            used=used,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, destination)
        archived_files.append(
            ArchivedFile(
                source_path=item.source_path,
                archive_relative_path=destination.relative_to(archive_directory),
                sha256=file_sha256(destination),
                size_bytes=destination.stat().st_size,
            )
        )
    return tuple(archived_files)


def _build_source_file_metadata(files: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for path in files
    ]


def _build_archived_file_metadata(
    files: Iterable[ArchivedFile],
) -> list[dict[str, object]]:
    return [
        {
            "source_path": str(item.source_path),
            "archive_path": str(item.archive_relative_path),
            "sha256": item.sha256,
            "size": item.size_bytes,
        }
        for item in files
    ]


def _read_git_value(project_root: Path, *args: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", *args],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return process.stdout.strip() or None if process.returncode == 0 else None


def _build_metadata_only_config_entries(
    references: Sequence[ConfigFileReference],
) -> list[dict[str, object]]:
    return [
        {
            "kind": reference.kind,
            "path": str(reference.source_path),
            "exists": True,
            "sha256": file_sha256(reference.source_path),
            "size": reference.source_path.stat().st_size,
        }
        for reference in references
    ]


def _build_archive_manifest(
    *,
    run_result: RunResult,
    request: RunArchiveRequest,
    run_name: str,
    creation_time: datetime,
    git_metadata: Mapping[str, object],
    input_file_metadata: Sequence[Mapping[str, object]],
    cleaned_file_metadata: Sequence[Mapping[str, object]],
    generated_output_metadata: Sequence[Mapping[str, object]],
    archived_output_files: Sequence[ArchivedFile],
    archived_trace_files: Sequence[ArchivedFile],
    archived_config_files: Sequence[ArchivedFile],
    archived_input_files: Sequence[ArchivedFile],
    archived_cleaned_files: Sequence[ArchivedFile],
    metadata_only_config_entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "created_at": creation_time.isoformat(),
        "command_line": list(request.command_line or tuple(sys.argv)),
        "project_root": str(run_result.plan.project_root.resolve()),
        "config_path": str(run_result.plan.config_path.resolve()),
        "output_dir": str(run_result.plan.out_dir.resolve()),
        "git": dict(git_metadata),
        "groups_files": {
            name: [str(path) for path in files]
            for name, files in run_result.groups_files.items()
        },
        "input_files": list(input_file_metadata),
        "cleaned_files": list(cleaned_file_metadata),
        "generated_outputs": list(generated_output_metadata),
        "output_files": _build_archived_file_metadata(archived_output_files),
        "copied_outputs": _build_archived_file_metadata(archived_output_files),
        "trace_files": _build_archived_file_metadata(archived_trace_files),
        "config_snapshot_files": _build_archived_file_metadata(
            archived_config_files
        ),
        "external_references": list(metadata_only_config_entries),
        "included_input_files": _build_archived_file_metadata(archived_input_files),
        "included_cleaned_files": _build_archived_file_metadata(
            archived_cleaned_files
        ),
        "copied_input_files": _build_archived_file_metadata(archived_input_files),
        "copied_cleaned_files": _build_archived_file_metadata(
            archived_cleaned_files
        ),
    }


def create_run_archive(
    *, run_result: RunResult, request: RunArchiveRequest
) -> RunArchiveResult:
    project_root = run_result.plan.project_root.resolve()
    creation_time = request.created_at or datetime.now().astimezone()
    run_name = sanitize_run_name(
        request.run_name or creation_time.strftime("%Y%m%d-%H%M%S")
    )
    archive_root = request.archive_root
    if not archive_root.is_absolute():
        archive_root = (project_root / archive_root).resolve()
    archive_directory = archive_root / run_name
    if archive_directory.exists():
        raise RunArchiveError(f"Run archive already exists: {archive_directory}")

    output_files = tuple(
        _validate_archive_source_file(path, "run output")
        for path in run_result.output_files
    )
    trace_files = tuple(
        _validate_archive_source_file(path, "run trace")
        for path in run_result.trace_files
    )
    input_files = tuple(
        _validate_archive_source_file(path, "run input")
        for path in run_result.input_files
    )
    cleaned_files = tuple(
        _validate_archive_source_file(path, "cleaned run file")
        for path in run_result.cleaned_files
    )
    snapshot_references = tuple(
        reference
        for reference in run_result.config_references
        if reference.archive_policy is ConfigArchivePolicy.SNAPSHOT
    )
    metadata_only_references = tuple(
        reference
        for reference in run_result.config_references
        if reference.archive_policy is ConfigArchivePolicy.METADATA_ONLY
    )
    for reference in (*snapshot_references, *metadata_only_references):
        _validate_archive_source_file(
            reference.source_path, f"config reference {reference.kind}"
        )

    try:
        archive_directory.mkdir(parents=True)
        archived_output_files = _copy_files_into_archive(
            (
                ArchiveCopySource(
                    source_path=path,
                    destination_relative_path=Path(path.name),
                )
                for path in output_files
            ),
            destination_root=archive_directory / "outputs",
            archive_directory=archive_directory,
        )
        archived_trace_files = _copy_files_into_archive(
            (
                ArchiveCopySource(
                    source_path=path,
                    destination_relative_path=Path(path.name),
                )
                for path in trace_files
            ),
            destination_root=archive_directory / "trace",
            archive_directory=archive_directory,
        )
        archived_config_files = _copy_files_into_archive(
            (
                ArchiveCopySource(
                    source_path=reference.source_path,
                    destination_relative_path=reference.snapshot_relative_path,
                )
                for reference in snapshot_references
                if reference.snapshot_relative_path is not None
            ),
            destination_root=archive_directory / "config_snapshot",
            archive_directory=archive_directory,
        )
        archived_input_files = (
            _copy_files_into_archive(
                (
                    ArchiveCopySource(
                        source_path=path,
                        destination_relative_path=path.relative_to(project_root)
                        if path.is_relative_to(project_root)
                        else Path(path.name),
                    )
                    for path in input_files
                ),
                destination_root=archive_directory / "input",
                archive_directory=archive_directory,
            )
            if request.include_input_files
            else ()
        )
        cleaned_root = (
            run_result.plan.cleaned_dir.resolve()
            if run_result.plan.cleaned_dir
            else project_root
        )
        archived_cleaned_files = (
            _copy_files_into_archive(
                (
                    ArchiveCopySource(
                        source_path=path,
                        destination_relative_path=path.relative_to(cleaned_root)
                        if path.is_relative_to(cleaned_root)
                        else Path(path.name),
                    )
                    for path in cleaned_files
                ),
                destination_root=archive_directory / "cleaned",
                archive_directory=archive_directory,
            )
            if request.include_cleaned_files
            else ()
        )
        metadata_only_config_entries = _build_metadata_only_config_entries(
            metadata_only_references
        )
        input_file_metadata = _build_source_file_metadata(input_files)
        cleaned_file_metadata = _build_source_file_metadata(cleaned_files)
        generated_output_metadata = _build_source_file_metadata(
            run_result.generated_outputs
        )
        git_metadata = {
            "branch": _read_git_value(project_root, "branch", "--show-current"),
            "commit": _read_git_value(project_root, "rev-parse", "HEAD"),
            "dirty": bool(
                _read_git_value(project_root, "status", "--porcelain")
            ),
        }
        manifest = _build_archive_manifest(
            run_result=run_result,
            request=request,
            run_name=run_name,
            creation_time=creation_time,
            git_metadata=git_metadata,
            input_file_metadata=input_file_metadata,
            cleaned_file_metadata=cleaned_file_metadata,
            generated_output_metadata=generated_output_metadata,
            archived_output_files=archived_output_files,
            archived_trace_files=archived_trace_files,
            archived_config_files=archived_config_files,
            archived_input_files=archived_input_files,
            archived_cleaned_files=archived_cleaned_files,
            metadata_only_config_entries=metadata_only_config_entries,
        )
        (archive_directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (archive_directory / "README.md").write_text(
            "# Run Archive\n\n"
            f"- input files: {len(input_files)}\n"
            f"- Included input files: {len(archived_input_files)}\n"
            f"- Included cleaned files: {len(archived_cleaned_files)}\n",
            encoding="utf-8",
        )
    except Exception as exc:
        if archive_directory.exists():
            shutil.rmtree(archive_directory)
        if isinstance(exc, RunArchiveError):
            raise
        raise RunArchiveError(f"Failed to create run archive: {exc}") from exc
    return RunArchiveResult(
        archive_directory=archive_directory,
        copied_files=ArchivedFileCounts(
            outputs=len(archived_output_files),
            traces=len(archived_trace_files),
            inputs=len(archived_input_files),
            cleaned=len(archived_cleaned_files),
            config_snapshots=len(archived_config_files),
        ),
    )
