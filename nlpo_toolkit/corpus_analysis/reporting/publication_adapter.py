"""Production filesystem adapter for the typed run-report publication port."""

from __future__ import annotations

from ..artifacts.models import ArtifactKind
from ..artifacts.writers.report import (
    write_run_metadata_artifact,
    write_summary_artifact,
)
from ..publication_models import RunReportPublication
from .metadata import run_metadata_to_json_value


def publish_run_report(request: RunReportPublication) -> None:
    write_summary_artifact(
        request.artifact_plan.require(ArtifactKind.SUMMARY), content=request.summary
    )
    write_run_metadata_artifact(
        request.artifact_plan.require(ArtifactKind.RUN_METADATA),
        data=run_metadata_to_json_value(request.metadata),
    )

