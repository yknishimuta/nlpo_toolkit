from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .config import AppConfig, ensure_app_config
from .cleaner_runtime import CleanerLoader, CleanerRunner, load_default_cleaner
from .corpus_errors import CleanerInspectionError


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CleanerInspectionError(
            f"Failed to read cleaner config: {path}: {exc}"
        ) from exc
    except UnicodeError as exc:
        raise CleanerInspectionError(
            f"Cleaner config is not valid UTF-8: {path}: {exc}"
        ) from exc
    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise CleanerInspectionError(
            f"Invalid cleaner YAML: {path}: {exc}"
        ) from exc
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise CleanerInspectionError(f"Cleaner config root must be a mapping: {path}")
    return obj


def resolve_cleaner_output_dir(cleaner_yml: Path) -> Path:
    cfg = _load_yaml(cleaner_yml)
    out = cfg.get("output")
    if not out:
        raise CleanerInspectionError(f"cleaner config missing 'output': {cleaner_yml}")
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
    cleaner: CleanerRunner | None = None,
    cleaner_loader: CleanerLoader = load_default_cleaner,
) -> Optional[Path]:
    """
    If cfg.preprocess.kind == 'cleaner', run its main entry point and return cleaned_dir.
    Otherwise return None.
    """
    from . import corpus

    return corpus.run_preprocess_if_needed(
        config=ensure_app_config(cfg),
        project_root=project_root,
        cleaner=cleaner,
        cleaner_loader=cleaner_loader,
    )
