from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .file_metadata import SourceFileMetadata
from .git_metadata import GitMetadata
from .models import (
    ArchiveCopyResult, ArchiveInventory, ArchivedFile, ExternalReferenceMetadata,
)


def _source_metadata(files: tuple[SourceFileMetadata, ...]) -> list[dict[str, object]]:
    return [
        {"path": str(item.path), "sha256": item.sha256, "size": item.size_bytes}
        for item in files
    ]


def _copied_metadata(files: tuple[ArchivedFile, ...]) -> list[dict[str, object]]:
    return [
        {
            "source_path": str(item.source_path),
            "archive_path": str(item.archive_relative_path),
            "sha256": item.sha256,
            "size": item.size_bytes,
        }
        for item in files
    ]


def build_archive_manifest(
    *,
    inventory: ArchiveInventory,
    copied: ArchiveCopyResult,
    git: GitMetadata,
    input_metadata: tuple[SourceFileMetadata, ...],
    cleaned_metadata: tuple[SourceFileMetadata, ...],
    generated_output_metadata: tuple[SourceFileMetadata, ...],
    external_reference_metadata: tuple[ExternalReferenceMetadata, ...],
) -> dict[str, object]:
    external = [
        {
            "kind": item.kind, "path": str(item.path), "exists": True,
            "sha256": item.sha256, "size": item.size_bytes,
        }
        for item in external_reference_metadata
    ]
    return {
        "run_name": inventory.run_name,
        "created_at": inventory.creation_time.isoformat(),
        "command_line": list(inventory.command_line),
        "project_root": str(inventory.project_root),
        "config_path": str(inventory.config_path),
        "output_dir": str(inventory.output_dir),
        "git": {"branch": git.branch, "commit": git.commit, "dirty": git.dirty},
        "groups_files": {
            name: [str(path) for path in files]
            for name, files in inventory.groups_files.items()
        },
        "input_files": _source_metadata(input_metadata),
        "cleaned_files": _source_metadata(cleaned_metadata),
        "generated_outputs": _source_metadata(generated_output_metadata),
        "output_files": _copied_metadata(copied.outputs),
        "copied_outputs": _copied_metadata(copied.outputs),
        "trace_files": _copied_metadata(copied.traces),
        "config_snapshot_files": _copied_metadata(copied.config_snapshots),
        "external_references": external,
        "included_input_files": _copied_metadata(copied.inputs),
        "included_cleaned_files": _copied_metadata(copied.cleaned),
        "copied_input_files": _copied_metadata(copied.inputs),
        "copied_cleaned_files": _copied_metadata(copied.cleaned),
    }


def write_archive_manifest(
    archive_directory: Path, manifest: Mapping[str, object]
) -> Path:
    path = archive_directory / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def render_archive_readme(
    *, inventory: ArchiveInventory, copied: ArchiveCopyResult
) -> str:
    return (
        "# Run Archive\n\n"
        f"- input files: {len(inventory.input_files)}\n"
        f"- Included input files: {len(copied.inputs)}\n"
        f"- Included cleaned files: {len(copied.cleaned)}\n"
    )


def write_archive_readme(archive_directory: Path, text: str) -> Path:
    path = archive_directory / "README.md"
    path.write_text(text, encoding="utf-8")
    return path
