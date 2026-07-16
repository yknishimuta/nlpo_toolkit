from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ArtifactKind(str, Enum):
    FREQUENCY = "frequency"
    DICTCHECK_KNOWN = "dictcheck_known"
    DICTCHECK_UNKNOWN = "dictcheck_unknown"
    REFERENCE_TAGS = "reference_tags"
    DIAGNOSTIC_TRACE = "diagnostic_trace"
    TOKEN_ARTIFACT = "token_artifact"
    TOKEN_ARTIFACT_METADATA = "token_artifact_metadata"
    PARTITION_VALIDATION_CSV = "partition_validation_csv"
    PARTITION_VALIDATION_JSON = "partition_validation_json"
    GROUP_COMPARISON_CSV = "group_comparison_csv"
    GROUP_COMPARISONS_JSON = "group_comparisons_json"
    SUMMARY = "summary"
    RUN_METADATA = "run_metadata"


_GROUP_KINDS = frozenset({
    ArtifactKind.FREQUENCY, ArtifactKind.DICTCHECK_KNOWN,
    ArtifactKind.DICTCHECK_UNKNOWN, ArtifactKind.REFERENCE_TAGS,
    ArtifactKind.DIAGNOSTIC_TRACE, ArtifactKind.TOKEN_ARTIFACT,
    ArtifactKind.TOKEN_ARTIFACT_METADATA,
})
_NAMED_KINDS = frozenset({
    ArtifactKind.PARTITION_VALIDATION_CSV, ArtifactKind.GROUP_COMPARISON_CSV,
})
_RUN_KINDS = frozenset(set(ArtifactKind) - _GROUP_KINDS - _NAMED_KINDS)


@dataclass(frozen=True)
class PlannedArtifact:
    kind: ArtifactKind
    path: Path
    group: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        if self.group is not None and self.name is not None:
            raise ValueError("PlannedArtifact cannot have both group and name")
        if self.kind in _GROUP_KINDS and not self.group:
            raise ValueError(f"{self.kind.value} requires group")
        if self.kind in _NAMED_KINDS and not self.name:
            raise ValueError(f"{self.kind.value} requires name")
        if self.kind in _RUN_KINDS and (self.group is not None or self.name is not None):
            raise ValueError(f"{self.kind.value} is a run-level artifact")

    @property
    def owner(self) -> str:
        if self.group is not None:
            return f"group={self.group}"
        if self.name is not None:
            return f"name={self.name}"
        return "run"


@dataclass(frozen=True)
class ArtifactPlan:
    artifacts: tuple[PlannedArtifact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        logical: dict[tuple[ArtifactKind, str | None, str | None], PlannedArtifact] = {}
        paths: dict[str, PlannedArtifact] = {}
        for artifact in self.artifacts:
            key = (artifact.kind, artifact.group, artifact.name)
            if key in logical:
                raise ValueError(
                    f"Duplicate artifact logical identifier: {artifact.kind.value} {artifact.owner}"
                )
            logical[key] = artifact
            portable_path = str(artifact.path).casefold()
            previous = paths.get(portable_path)
            if previous is not None:
                raise ValueError(
                    "Artifact path collision: "
                    f"{previous.kind.value} {previous.owner} and "
                    f"{artifact.kind.value} {artifact.owner} use {artifact.path}"
                )
            paths[portable_path] = artifact
            self._validate_filesystem_path(artifact)

    @staticmethod
    def _validate_filesystem_path(artifact: PlannedArtifact) -> None:
        path = artifact.path
        if path.exists() and path.is_dir():
            raise ValueError(
                f"Artifact path is an existing directory: {artifact.kind.value} "
                f"{artifact.owner} uses {path}"
            )
        parent = path.parent
        while parent != parent.parent:
            if parent.exists():
                if not parent.is_dir():
                    raise ValueError(
                        f"Artifact path ancestor is not a directory: {artifact.kind.value} "
                        f"{artifact.owner} uses {path}; ancestor={parent}"
                    )
                break
            parent = parent.parent

    def select(self, *, kinds: Collection[ArtifactKind] | None = None,
               group: str | None = None, name: str | None = None) -> tuple[PlannedArtifact, ...]:
        selected = set(kinds) if kinds is not None else None
        return tuple(a for a in self.artifacts if
                     (selected is None or a.kind in selected) and
                     (group is None or a.group == group) and
                     (name is None or a.name == name))

    def optional(self, kind: ArtifactKind, *, group: str | None = None,
                 name: str | None = None) -> PlannedArtifact | None:
        found = self.select(kinds={kind}, group=group, name=name)
        if len(found) > 1:
            raise ValueError(f"Multiple artifacts found: {kind.value} group={group} name={name}")
        return found[0] if found else None

    def require(self, kind: ArtifactKind, *, group: str | None = None,
                name: str | None = None) -> PlannedArtifact:
        found = self.select(kinds={kind}, group=group, name=name)
        if len(found) != 1:
            raise ValueError(
                f"Expected exactly one artifact: {kind.value} group={group} "
                f"name={name}; found={len(found)}"
            )
        return found[0]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(artifact.path for artifact in self.artifacts)
