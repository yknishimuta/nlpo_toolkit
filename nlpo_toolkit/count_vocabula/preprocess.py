from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .config import AppConfig, ensure_app_config


def _load_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return obj


def resolve_cleaner_output_dir(cleaner_yml: Path) -> Path:
    cfg = _load_yaml(cleaner_yml)
    out = cfg.get("output")
    if not out:
        raise ValueError(f"cleaner config missing 'output': {cleaner_yml}")
    out_p = Path(str(out))
    if out_p.is_absolute():
        return out_p
    return (cleaner_yml.parent / out_p).resolve()


def expand_cleaned_dir_placeholders(patterns: list[str], cleaned_dir: Optional[Path]) -> list[str]:
    if cleaned_dir is None:
        return patterns
    return [p.replace("{cleaned_dir}", str(cleaned_dir)) for p in patterns]


def run_preprocess_if_needed(
    *,
    cfg: AppConfig | Mapping[str, object],
    project_root: Path,
    clean_mod: Any,
) -> Optional[Path]:
    """
    If cfg.preprocess.kind == 'cleaner', run clean_mod.main([...]) and return cleaned_dir.
    Otherwise return None.
    """
    from .corpus import run_preprocess_if_needed as _run_preprocess_if_needed

    return _run_preprocess_if_needed(
        config=ensure_app_config(cfg),
        project_root=project_root,
        clean_mod=clean_mod,
    )
