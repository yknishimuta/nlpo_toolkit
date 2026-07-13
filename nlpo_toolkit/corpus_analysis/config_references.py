"""Resolve and validate the exact file inventory referenced by typed config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection

from .config import AppConfig


class ConfigReferenceError(RuntimeError):
    """A configured file reference is missing or is not a regular file."""


@dataclass(frozen=True)
class ConfigFileReference:
    kind: str
    path: Path
    required: bool
    copy_to_snapshot: bool
    snapshot_path: Path | None = None


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
        return reference.path if reference is not None else None


def _resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _snapshot_path(path: Path, project_root: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError:
        return Path("external") / path.name


def _reference(
    *,
    kind: str,
    path: Path,
    project_root: Path,
    required: bool = True,
    copy_to_snapshot: bool = True,
) -> ConfigFileReference:
    return ConfigFileReference(
        kind=kind,
        path=path,
        required=required,
        copy_to_snapshot=copy_to_snapshot,
        snapshot_path=(
            _snapshot_path(path, project_root) if copy_to_snapshot else None
        ),
    )


def _validate(reference: ConfigFileReference) -> None:
    if not reference.required:
        return
    if not reference.path.exists():
        raise ConfigReferenceError(
            f"Configured file does not exist: {reference.kind}: {reference.path}"
        )
    if not reference.path.is_file():
        raise ConfigReferenceError(
            f"Configured path is not a file: {reference.kind}: {reference.path}"
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
        _reference(
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
            _reference(
                kind="preprocess.config",
                path=_resolve_path(root, preprocess_path),
                project_root=root,
            )
        )

    if cleaner_inspection is not None:
        for cleaner_reference in cleaner_inspection.referenced_files:
            items.append(
                _reference(
                    kind=cleaner_reference.kind,
                    path=_resolve_path(root, cleaner_reference.path),
                    project_root=root,
                    # An inspection entry represents an explicitly configured
                    # path. Absence is represented by no entry at all.
                    required=True,
                )
            )

    for kind, raw_path, copy_to_snapshot in (
        ("dictcheck.lemma_normalize", config.dictcheck.lemma_normalize, True),
        ("dictcheck.wordlist", config.dictcheck.wordlist, False),
        ("ref_tags.patterns", config.ref_tags.patterns, True),
        (
            "filters.roman_exceptions_file",
            config.filters.roman_exceptions_file,
            True,
        ),
    ):
        if raw_path:
            items.append(
                _reference(
                    kind=kind,
                    path=_resolve_path(root, raw_path),
                    project_root=root,
                    copy_to_snapshot=copy_to_snapshot,
                )
            )

    resolved = ResolvedConfigFiles(tuple(items))
    for reference in resolved.references:
        _validate(reference)
    return resolved
