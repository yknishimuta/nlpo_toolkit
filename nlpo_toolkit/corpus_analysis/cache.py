from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class CacheClearError(ValueError):
    pass


@dataclass(frozen=True)
class CacheClearRequest:
    project_root: Path
    config_path: Path | None = None


@dataclass(frozen=True)
class CacheClearResult:
    cache_dir: Path
    removed: bool


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CacheClearError(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise CacheClearError("Top-level YAML must be a mapping.")
    return data


def _cache_dir_from_config(path: Path) -> str | None:
    data = _load_yaml_mapping(path)
    analysis_cache = data.get("analysis_cache")
    if not isinstance(analysis_cache, dict):
        return None
    cache_dir = analysis_cache.get("dir")
    if cache_dir is None:
        return None
    if not isinstance(cache_dir, str) or not cache_dir.strip():
        raise CacheClearError("analysis_cache.dir must be a non-empty string path.")
    return cache_dir


def resolve_cache_dir(project_root: Path, config_path: Path | None = None) -> Path:
    project_root = project_root.resolve()
    default_cache_dir = ".analysis_cache"
    raw_cache_dir = default_cache_dir

    if config_path is None:
        default_config = project_root / "config" / "groups.config.yml"
        if default_config.exists():
            configured = _cache_dir_from_config(default_config)
            if configured is not None:
                raw_cache_dir = configured
    else:
        if not config_path.is_absolute():
            config_path = project_root / config_path
        configured = _cache_dir_from_config(config_path.resolve())
        if configured is not None:
            raw_cache_dir = configured

    cache_dir = Path(raw_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    cache_dir = cache_dir.resolve()

    if cache_dir == project_root:
        raise CacheClearError("Refusing to clear project root as a cache directory.")
    try:
        cache_dir.relative_to(project_root)
    except ValueError as exc:
        raise CacheClearError(
            f"Refusing to clear cache outside project root: {cache_dir}"
        ) from exc

    return cache_dir


def clear_cache(request: CacheClearRequest) -> CacheClearResult:
    project_root = request.project_root.resolve()
    cache_dir = resolve_cache_dir(project_root, request.config_path)

    if not cache_dir.exists() and not cache_dir.is_symlink():
        return CacheClearResult(cache_dir=cache_dir, removed=False)

    if cache_dir.is_symlink() or cache_dir.is_file():
        cache_dir.unlink()
    else:
        shutil.rmtree(cache_dir)

    return CacheClearResult(cache_dir=cache_dir, removed=True)
