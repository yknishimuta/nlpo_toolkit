from __future__ import annotations

import json
from pathlib import Path

from nlpo_toolkit.serialization.types import JsonObject, JsonValue

from .git_metadata import GitMetadata
from .models import (
    ArchiveCopyResult, ArchiveGitReport, ArchiveInventory, ArchiveManifest,
    ArchivedFile, ExternalReferenceManifestEntry, ExternalReferenceMetadata,
    SourceFileMetadata,
)


def _source_value(item: SourceFileMetadata) -> JsonObject:
    return {"path": str(item.path), "sha256": item.sha256, "size": item.size_bytes}


def _copied_value(item: ArchivedFile) -> JsonObject:
    return {
        "source_path": str(item.source_path),
        "archive_path": str(item.archive_relative_path),
        "sha256": item.sha256, "size": item.size_bytes,
    }


def build_archive_manifest(
    *, inventory: ArchiveInventory, copied: ArchiveCopyResult, git: GitMetadata,
    input_metadata: tuple[SourceFileMetadata, ...],
    cleaned_metadata: tuple[SourceFileMetadata, ...],
    generated_output_metadata: tuple[SourceFileMetadata, ...],
    external_reference_metadata: tuple[ExternalReferenceMetadata, ...],
) -> ArchiveManifest:
    return ArchiveManifest(
        run_name=inventory.run_name, created_at=inventory.creation_time,
        command_line=inventory.command_line, project_root=inventory.project_root,
        config_path=inventory.config_path, output_dir=inventory.output_dir,
        git=ArchiveGitReport(git.branch, git.commit, git.dirty),
        groups_files=inventory.groups_files, input_files=input_metadata,
        cleaned_files=cleaned_metadata, generated_outputs=generated_output_metadata,
        copied_outputs=copied.outputs, trace_files=copied.traces,
        config_snapshot_files=copied.config_snapshots,
        external_references=tuple(
            ExternalReferenceManifestEntry(
                item.kind, item.path, True, item.sha256, item.size_bytes,
            ) for item in external_reference_metadata
        ),
        copied_input_files=copied.inputs, copied_cleaned_files=copied.cleaned,
    )


def archive_manifest_to_json_value(manifest: ArchiveManifest) -> JsonObject:
    copied_outputs = [_copied_value(item) for item in manifest.copied_outputs]
    copied_inputs = [_copied_value(item) for item in manifest.copied_input_files]
    copied_cleaned = [_copied_value(item) for item in manifest.copied_cleaned_files]
    external: list[JsonValue] = [
        {"kind": item.kind, "path": str(item.path), "exists": item.exists,
         "sha256": item.sha256, "size": item.size_bytes}
        for item in manifest.external_references
    ]
    return {
        "run_name": manifest.run_name, "created_at": manifest.created_at.isoformat(),
        "command_line": list(manifest.command_line),
        "project_root": str(manifest.project_root), "config_path": str(manifest.config_path),
        "output_dir": str(manifest.output_dir),
        "git": {"branch": manifest.git.branch, "commit": manifest.git.commit,
                "dirty": manifest.git.dirty},
        "groups_files": {name: [str(path) for path in paths]
                         for name, paths in manifest.groups_files.items()},
        "input_files": [_source_value(item) for item in manifest.input_files],
        "cleaned_files": [_source_value(item) for item in manifest.cleaned_files],
        "generated_outputs": [_source_value(item) for item in manifest.generated_outputs],
        "output_files": copied_outputs, "copied_outputs": copied_outputs,
        "trace_files": [_copied_value(item) for item in manifest.trace_files],
        "config_snapshot_files": [_copied_value(item) for item in manifest.config_snapshot_files],
        "external_references": external,
        "included_input_files": copied_inputs, "included_cleaned_files": copied_cleaned,
        "copied_input_files": copied_inputs, "copied_cleaned_files": copied_cleaned,
    }


def write_archive_manifest(archive_directory: Path, manifest: ArchiveManifest) -> Path:
    path = archive_directory / "manifest.json"
    path.write_text(
        json.dumps(archive_manifest_to_json_value(manifest), ensure_ascii=False,
                   indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def render_archive_readme(*, inventory: ArchiveInventory, copied: ArchiveCopyResult) -> str:
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
