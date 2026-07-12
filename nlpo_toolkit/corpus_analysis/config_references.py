"""Build exact config-file inventories from typed planning results."""

from pathlib import Path

from nlpo_toolkit.cleaner_contracts import CleanerConfigInspection

from .config import AppConfig
from .corpus import resolve_project_path
from .runner_types import ReferencedConfigFile


def _snapshot(
    path: Path,
    project_root: Path,
    kind: str,
    *,
    required: bool = True,
) -> ReferencedConfigFile:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        relative = Path("external") / path.name
    return ReferencedConfigFile(kind, path.resolve(), required, relative)


def build_config_file_inventory(
    *,
    config: AppConfig,
    config_path: Path,
    project_root: Path,
    cleaner_inspection: CleanerConfigInspection | None,
) -> tuple[ReferencedConfigFile, ...]:
    items = [_snapshot(config_path, project_root, "root_config")]
    if cleaner_inspection is not None:
        items.append(
            _snapshot(
                cleaner_inspection.config.source_path,
                project_root,
                "preprocess.config",
            )
        )
        items.extend(
            _snapshot(
                reference.path,
                project_root,
                reference.kind,
                required=reference.required,
            )
            for reference in cleaner_inspection.referenced_files
        )
    for kind, raw_path in (
        ("dictcheck.lemma_normalize", config.dictcheck.lemma_normalize),
        ("ref_tags.patterns", config.ref_tags.patterns),
        ("filters.roman_exceptions_file", config.filters.roman_exceptions_file),
    ):
        if raw_path:
            items.append(
                _snapshot(
                    resolve_project_path(project_root, raw_path),
                    project_root,
                    kind,
                )
            )
    if config.dictcheck.wordlist:
        items.append(
            ReferencedConfigFile(
                "dictcheck.wordlist",
                resolve_project_path(
                    project_root, config.dictcheck.wordlist
                ).resolve(),
                False,
            )
        )
    return tuple(dict.fromkeys(items))
