from __future__ import annotations

import shutil
from pathlib import Path

from .file_metadata import read_source_file_metadata
from .models import ArchiveCopyResult, ArchiveCopySource, ArchivedFile, ArchiveInventory


def _allocate_unique_path(
    *, destination_root: Path, requested: Path, used: set[Path]
) -> Path:
    candidate = requested
    index = 2
    while candidate in used or (destination_root / candidate).exists():
        candidate = requested.with_name(f"{requested.stem}_{index}{requested.suffix}")
        index += 1
    used.add(candidate)
    return destination_root / candidate


def copy_archive_files(
    files: tuple[ArchiveCopySource, ...],
    *, destination_root: Path, archive_directory: Path,
) -> tuple[ArchivedFile, ...]:
    copied: list[ArchivedFile] = []
    used: set[Path] = set()
    for item in files:
        destination = _allocate_unique_path(
            destination_root=destination_root,
            requested=item.destination_relative_path,
            used=used,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, destination)
        metadata = read_source_file_metadata(destination)
        copied.append(ArchivedFile(
            source_path=item.source_path,
            archive_relative_path=destination.relative_to(archive_directory),
            sha256=metadata.sha256,
            size_bytes=metadata.size_bytes,
        ))
    return tuple(copied)


def copy_archive_inventory(inventory: ArchiveInventory) -> ArchiveCopyResult:
    archive = inventory.archive_directory
    return ArchiveCopyResult(
        outputs=copy_archive_files(
            inventory.output_sources, destination_root=archive / "outputs",
            archive_directory=archive,
        ),
        traces=copy_archive_files(
            inventory.trace_sources, destination_root=archive / "trace",
            archive_directory=archive,
        ),
        config_snapshots=copy_archive_files(
            inventory.config_snapshot_sources,
            destination_root=archive / "config_snapshot", archive_directory=archive,
        ),
        inputs=copy_archive_files(
            inventory.input_sources, destination_root=archive / "input",
            archive_directory=archive,
        ),
        cleaned=copy_archive_files(
            inventory.cleaned_sources, destination_root=archive / "cleaned",
            archive_directory=archive,
        ),
    )
