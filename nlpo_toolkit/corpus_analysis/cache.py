from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from pydantic import ValidationError

from nlpo_toolkit.configuration.yaml_loader import YamlLoadError, load_yaml_mapping
from .config.models import AnalysisCacheConfig


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


def _cache_dir_from_config(path: Path) -> str | None:
    try:
        data = load_yaml_mapping(path)
    except YamlLoadError as exc:
        raise CacheClearError(str(exc)) from exc
    section = data.get("analysis_cache")
    if section is None:
        return None
    try:
        config = AnalysisCacheConfig.model_validate(
            section, by_alias=True, by_name=False
        )
    except ValidationError as exc:
        raise CacheClearError(f"Invalid analysis_cache config: {exc}") from exc
    return config.directory


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
