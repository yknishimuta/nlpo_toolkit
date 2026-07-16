from __future__ import annotations

from ..models import ArtifactKind, PlannedArtifact
from ..publication import ArtifactPublicationError, publish_json, publish_text


def _require(artifact: PlannedArtifact, kind: ArtifactKind) -> None:
    if artifact.kind is not kind:
        raise ArtifactPublicationError(f"Wrong artifact kind: expected={kind.value} actual={artifact.kind.value} path={artifact.path}")


def write_summary_artifact(artifact: PlannedArtifact, *, content: str) -> None:
    _require(artifact, ArtifactKind.SUMMARY)
    publish_text(artifact, content=content)


def write_run_metadata_artifact(artifact: PlannedArtifact, *, data: object) -> None:
    _require(artifact, ArtifactKind.RUN_METADATA)
    publish_json(artifact, data=data)
