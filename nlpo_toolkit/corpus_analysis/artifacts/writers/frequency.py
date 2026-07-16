from __future__ import annotations

from typing import Mapping

from ..models import ArtifactKind, PlannedArtifact
from ..publication import ArtifactPublicationError, open_csv_artifact


def _require_kind(artifact: PlannedArtifact, expected: ArtifactKind) -> None:
    if artifact.kind is not expected:
        owner = f"group={artifact.group}" if artifact.group is not None else "run"
        raise ArtifactPublicationError(
            f"Wrong artifact kind: expected={expected.value} actual={artifact.kind.value} "
            f"{owner} path={artifact.path}"
        )


def _write(artifact: PlannedArtifact, counter: Mapping[str, int], header: tuple[str, str]) -> None:
    with open_csv_artifact(artifact, fieldnames=header) as writer:
        writer.writeheader()
        for item, count in sorted(counter.items(), key=lambda value: (-value[1], value[0])):
            writer.writerow({header[0]: item, header[1]: count})


def write_frequency_artifact(artifact: PlannedArtifact, *, counter: Mapping[str, int], header: tuple[str, str]) -> None:
    _require_kind(artifact, ArtifactKind.FREQUENCY)
    _write(artifact, counter, header)


def write_known_frequency_artifact(artifact: PlannedArtifact, *, counter: Mapping[str, int], header: tuple[str, str]) -> None:
    _require_kind(artifact, ArtifactKind.DICTCHECK_KNOWN)
    _write(artifact, counter, header)


def write_unknown_frequency_artifact(artifact: PlannedArtifact, *, counter: Mapping[str, int], header: tuple[str, str]) -> None:
    _require_kind(artifact, ArtifactKind.DICTCHECK_UNKNOWN)
    _write(artifact, counter, header)
