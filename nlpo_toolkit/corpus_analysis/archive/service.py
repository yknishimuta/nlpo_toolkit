from __future__ import annotations

import shutil

from ..archive_types import ArchivedFileCounts, RunArchiveRequest, RunArchiveResult
from ..runner_types import RunResult
from .copying import copy_archive_inventory
from .errors import RunArchiveError
from .file_metadata import read_external_reference_metadata, read_source_files_metadata
from .git_metadata import read_git_metadata
from .inventory import collect_archive_inventory
from .manifest import (
    build_archive_manifest, render_archive_readme,
    write_archive_manifest, write_archive_readme,
)


def create_run_archive(
    *, run_result: RunResult, request: RunArchiveRequest
) -> RunArchiveResult:
    inventory = collect_archive_inventory(run_result=run_result, request=request)
    archive_created = False
    try:
        inventory.archive_directory.mkdir(parents=True)
        archive_created = True
        copied = copy_archive_inventory(inventory)
        input_metadata = read_source_files_metadata(inventory.input_files)
        cleaned_metadata = read_source_files_metadata(inventory.cleaned_files)
        generated_metadata = read_source_files_metadata(inventory.generated_outputs)
        external_metadata = tuple(
            read_external_reference_metadata(reference.kind, reference.source_path)
            for reference in inventory.metadata_only_references
        )
        git = read_git_metadata(inventory.project_root)
        manifest = build_archive_manifest(
            inventory=inventory, copied=copied, git=git,
            input_metadata=input_metadata, cleaned_metadata=cleaned_metadata,
            generated_output_metadata=generated_metadata,
            external_reference_metadata=external_metadata,
        )
        write_archive_manifest(inventory.archive_directory, manifest)
        write_archive_readme(
            inventory.archive_directory,
            render_archive_readme(inventory=inventory, copied=copied),
        )
    except Exception as exc:
        if archive_created and inventory.archive_directory.exists():
            shutil.rmtree(inventory.archive_directory)
        if isinstance(exc, RunArchiveError):
            raise
        raise RunArchiveError(f"Failed to create run archive: {exc}") from exc

    return RunArchiveResult(
        archive_directory=inventory.archive_directory,
        copied_files=ArchivedFileCounts(
            outputs=len(copied.outputs), traces=len(copied.traces),
            inputs=len(copied.inputs), cleaned=len(copied.cleaned),
            config_snapshots=len(copied.config_snapshots),
        ),
    )
