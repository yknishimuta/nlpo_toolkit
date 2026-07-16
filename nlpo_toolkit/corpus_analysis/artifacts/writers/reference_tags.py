from __future__ import annotations

from typing import Mapping

from ..models import ArtifactKind, PlannedArtifact
from ..publication import ArtifactPublicationError, open_csv_artifact


def write_reference_tags_artifact(artifact: PlannedArtifact, *, counter: Mapping[str, int]) -> None:
    if artifact.kind is not ArtifactKind.REFERENCE_TAGS:
        raise ArtifactPublicationError(
            f"Wrong artifact kind: expected=reference_tags actual={artifact.kind.value} "
            f"group={artifact.group} path={artifact.path}"
        )
    with open_csv_artifact(artifact, fieldnames=("tag", "count")) as writer:
        writer.writeheader()
        for tag, count in sorted(counter.items(), key=lambda value: (-value[1], value[0])):
            writer.writerow({"tag": tag, "count": count})
