"""Resolve and validate the exact file inventory referenced by typed config."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection

from .config import AppConfig


class ConfigReferenceError(RuntimeError):
    """A configured file reference is missing or is not a regular file."""


class ConfigArchivePolicy(StrEnum):
    SNAPSHOT = "snapshot"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class ConfigFileReference:
    kind: str
    source_path: Path
    archive_policy: ConfigArchivePolicy
    snapshot_relative_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.archive_policy, ConfigArchivePolicy):
            raise TypeError("archive_policy must be a ConfigArchivePolicy")
        if not self.source_path.is_absolute():
            raise ValueError("Config reference source_path must be absolute")
        snapshot_relative_path = self.snapshot_relative_path
        if self.archive_policy is ConfigArchivePolicy.SNAPSHOT:
            if snapshot_relative_path is None or snapshot_relative_path in {Path(), Path(".")}:
                raise ValueError("Snapshot references require a relative snapshot path")
            if snapshot_relative_path.is_absolute() or ".." in snapshot_relative_path.parts:
                raise ValueError("Snapshot path must be a safe relative path")
        elif snapshot_relative_path is not None:
            raise ValueError("Metadata-only references cannot have a snapshot path")


@dataclass(frozen=True)
class ResolvedConfigFiles:
    references: tuple[ConfigFileReference, ...] = ()

    def __post_init__(self) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for reference in self.references:
            if reference.kind in seen:
                duplicates.add(reference.kind)
            seen.add(reference.kind)
        if duplicates:
            kinds = ", ".join(sorted(duplicates))
            raise ConfigReferenceError(f"Duplicate config file reference kind: {kinds}")

    def get(self, kind: str) -> ConfigFileReference | None:
        return next(
            (reference for reference in self.references if reference.kind == kind),
            None,
        )

    def require(self, kind: str) -> ConfigFileReference:
        reference = self.get(kind)
        if reference is None:
            raise ConfigReferenceError(f"Config file reference is not configured: {kind}")
        return reference

    def path(self, kind: str) -> Path | None:
        reference = self.get(kind)
        return reference.source_path if reference is not None else None


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _snapshot_relative_path(path: Path, project_root: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError:
        return Path("external") / path.name


def _snapshot_reference(
    *,
    kind: str,
    path: Path,
    project_root: Path,
) -> ConfigFileReference:
    return ConfigFileReference(
        kind=kind,
        source_path=path.resolve(),
        archive_policy=ConfigArchivePolicy.SNAPSHOT,
        snapshot_relative_path=_snapshot_relative_path(path, project_root),
    )


def _metadata_only_reference(*, kind: str, path: Path) -> ConfigFileReference:
    return ConfigFileReference(
        kind=kind,
        source_path=path.resolve(),
        archive_policy=ConfigArchivePolicy.METADATA_ONLY,
    )


def _validate(reference: ConfigFileReference) -> None:
    if not reference.source_path.exists():
        raise ConfigReferenceError(
            f"Configured file does not exist: {reference.kind}: {reference.source_path}"
        )
    if not reference.source_path.is_file():
        raise ConfigReferenceError(
            f"Configured path is not a file: {reference.kind}: {reference.source_path}"
        )


def resolve_config_files(
    *,
    config: AppConfig,
    config_path: Path,
    project_root: Path,
    cleaner_inspection: CleanerConfigInspection | None,
) -> ResolvedConfigFiles:
    """Resolve and validate every explicitly configured file reference once."""
    root = Path(project_root).resolve()
    items = [
        _snapshot_reference(
            kind="root_config",
            path=_resolve_path(root, config_path),
            project_root=root,
        )
    ]

    preprocess_path: str | Path | None = config.preprocess.config
    if cleaner_inspection is not None:
        preprocess_path = cleaner_inspection.config.source_path
    if preprocess_path is not None:
        items.append(
            _snapshot_reference(
                kind="preprocess.config",
                path=_resolve_path(root, preprocess_path),
                project_root=root,
            )
        )

    if cleaner_inspection is not None:
        for cleaner_reference in cleaner_inspection.referenced_files:
            items.append(
                _snapshot_reference(
                    kind=cleaner_reference.kind,
                    path=_resolve_path(root, cleaner_reference.path),
                    project_root=root,
                )
            )

    for kind, raw_path, archive_policy in (
        ("dictcheck.lemma_normalize", config.dictcheck.lemma_normalize, ConfigArchivePolicy.SNAPSHOT),
        ("dictcheck.wordlist", config.dictcheck.wordlist, ConfigArchivePolicy.METADATA_ONLY),
        ("ref_tags.patterns", config.ref_tags.patterns, ConfigArchivePolicy.SNAPSHOT),
        (
            "filters.roman_exceptions_file",
            config.filters.roman_exceptions_file,
            ConfigArchivePolicy.SNAPSHOT,
        ),
    ):
        if raw_path:
            items.append(
                (
                    _snapshot_reference(
                        kind=kind,
                        path=_resolve_path(root, raw_path),
                        project_root=root,
                    )
                    if archive_policy is ConfigArchivePolicy.SNAPSHOT
                    else _metadata_only_reference(
                        kind=kind,
                        path=_resolve_path(root, raw_path),
                    )
                )
            )

    resolved = ResolvedConfigFiles(tuple(items))
    for reference in resolved.references:
        _validate(reference)
    return resolved
