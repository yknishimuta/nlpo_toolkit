"""Build exact config-file inventories before archiving."""

from pathlib import Path

import yaml

from .config import AppConfig
from .corpus import resolve_project_path
from .runner_types import ReferencedConfigFile


def _snapshot(path: Path, project_root: Path, kind: str) -> ReferencedConfigFile:
    try:
        relative = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        relative = Path("external") / path.name
    return ReferencedConfigFile(kind, path.resolve(), True, relative)


def build_config_file_inventory(
    *, config: AppConfig, config_path: Path, project_root: Path
) -> tuple[ReferencedConfigFile, ...]:
    items = [_snapshot(config_path, project_root, "root_config")]
    if config.preprocess.kind == "cleaner" and config.preprocess.config:
        cleaner = resolve_project_path(project_root, config.preprocess.config)
        items.append(_snapshot(cleaner, project_root, "preprocess.config"))
        if cleaner.exists():
            raw = yaml.safe_load(cleaner.read_text(encoding="utf-8")) or {}
            if isinstance(raw, dict):
                for key in ("rules_path", "lexicon_map_path"):
                    if raw.get(key):
                        path = (cleaner.parent / str(raw[key])).resolve()
                        items.append(_snapshot(path, project_root, f"preprocess.{key}"))
    for kind, raw_path in (
        ("dictcheck.lemma_normalize", config.dictcheck.lemma_normalize),
        ("ref_tags.patterns", config.ref_tags.patterns),
        ("filters.roman_exceptions_file", config.filters.roman_exceptions_file),
    ):
        if raw_path:
            items.append(_snapshot(resolve_project_path(project_root, raw_path), project_root, kind))
    if config.dictcheck.wordlist:
        items.append(
            ReferencedConfigFile(
                "dictcheck.wordlist",
                resolve_project_path(project_root, config.dictcheck.wordlist).resolve(),
                False,
            )
        )
    return tuple(dict.fromkeys(items))


def cleaner_input_files(config: AppConfig, project_root: Path) -> tuple[Path, ...]:
    if config.preprocess.kind != "cleaner" or not config.preprocess.config:
        return ()
    cleaner = resolve_project_path(project_root, config.preprocess.config)
    raw = yaml.safe_load(cleaner.read_text(encoding="utf-8")) or {}
    input_raw = raw.get("input") if isinstance(raw, dict) else None
    if not input_raw:
        return ()
    path = (cleaner.parent / str(input_raw)).resolve()
    if path.is_file():
        return (path,)
    if path.is_dir():
        return tuple(sorted(p.resolve() for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"))
    return ()
