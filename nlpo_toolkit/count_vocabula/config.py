from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

import yaml


class GroupDef(TypedDict):
    files: list[str]


class PreprocessCleaner(TypedDict):
    kind: str            # must be "cleaner"
    config: str          # path to cleaner yaml


class Config(TypedDict, total=False):
    # One of these may be provided in YAML; internally we normalize to "groups".
    group: Dict[str, Any]
    groups: Dict[str, GroupDef]
    preprocess: Dict[str, Any]
    out_dir: str
    vocab_path: Optional[str]
    language: str
    stanza_package: Optional[str]
    cpu_only: bool
    analysis_unit: str
    grouping: Dict[str, Any]
    validations: Dict[str, Any]
    comparisons: list[Dict[str, Any]]


def normalize_groups(cfg: dict) -> dict:
    """
    Normalize single-group sugar 'group' into 'groups'.
    After normalization, cfg['groups'] must exist and be a mapping.
    """
    if "groups" in cfg and cfg["groups"] is not None:
        return cfg

    if "group" in cfg and cfg["group"]:
        g = cfg["group"]
        if not isinstance(g, dict):
            raise ValueError("'group' must be a mapping.")
        name = g.get("name", "text")
        files = g.get("files")

        if not files:
            raise ValueError("'group.files' is required.")
        if not isinstance(files, list) or not all(isinstance(x, str) for x in files):
            raise ValueError("'group.files' must be list[str].")

        cfg["groups"] = {name: {"files": files}}
        return cfg

    grouping = cfg.get("grouping") or {}
    if isinstance(grouping, dict) and grouping.get("mode") == "auto_single_cleaned":
        name = str(grouping.get("auto_group_name") or "text")
        cfg["groups"] = {name: {"files": []}}
        return cfg

    raise ValueError("Config must define 'groups' or 'group'.")


def _validate_groups(groups: Any) -> None:
    if not isinstance(groups, dict):
        raise ValueError("Config 'groups' must be a mapping.")
    for k, v in groups.items():
        if not isinstance(k, str) or not k:
            raise ValueError("Group name must be a non-empty string.")
        if not isinstance(v, dict) or "files" not in v:
            raise ValueError(f"Group '{k}' must have 'files' list.")
        files = v["files"]
        if not isinstance(files, list) or not all(isinstance(x, str) for x in files):
            raise ValueError(f"Group '{k}' must have 'files' as list[str].")


def _validate_preprocess(pp: Any) -> None:
    if pp is None:
        return
    if not isinstance(pp, dict):
        raise ValueError("'preprocess' must be a mapping.")
    kind = pp.get("kind")
    if kind is None:
        raise ValueError("'preprocess.kind' is required when preprocess is provided.")
    if kind != "cleaner":
        raise ValueError(f"Unsupported preprocess.kind: {kind!r} (only 'cleaner' is supported).")
    cfg_path = pp.get("config")
    if not isinstance(cfg_path, str) or not cfg_path.strip():
        raise ValueError("'preprocess.config' must be a non-empty string path.")

def _validate_analysis_unit(cfg: dict) -> None:
    unit = cfg.get("analysis_unit", "lemma")
    if unit not in {"lemma", "surface"}:
        raise ValueError("analysis_unit must be 'lemma' or 'surface'")

def _validate_grouping(cfg: dict) -> None:
    grouping = cfg.get("grouping")
    if grouping is None:
        return
    if not isinstance(grouping, dict):
        raise ValueError("'grouping' must be a mapping.")
    mode = grouping.get("mode", "groups")
    if mode not in {"groups", "per_file", "auto_single_cleaned"}:
        raise ValueError("grouping.mode must be 'groups', 'per_file', or 'auto_single_cleaned'")

def _validate_partition_validations(cfg: dict) -> None:
    from .partition_validation import parse_partition_specs

    specs = parse_partition_specs(cfg)
    if not specs:
        return

    grouping = cfg.get("grouping") or {}
    if isinstance(grouping, dict) and grouping.get("mode", "groups") == "per_file":
        raise ValueError("validations.partitions cannot be used with grouping.mode: per_file")


def _validate_comparisons(cfg: dict) -> None:
    from .comparison import parse_comparison_specs

    parse_comparison_specs(cfg)

def load_config(path: Path) -> Config:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError("Config file must be YAML (.yml / .yaml)")

    text = path.read_text(encoding="utf-8")
    config_data = yaml.safe_load(text) or {}
    if not isinstance(config_data, dict):
        raise ValueError("Top-level YAML must be a mapping.")

    # Hard deprecations: reject old keys to prevent silent misconfig.
    if "cleaner_config" in config_data:
        raise ValueError("Deprecated key 'cleaner_config' is not supported. Use preprocess: {kind: cleaner, config: ...}.")
    if "stanza_pkg" in config_data:
        raise ValueError("Deprecated key 'stanza_pkg' is not supported. Use 'stanza_package'.")

    # Normalize and validate groups/group
    config_data = normalize_groups(config_data)
    _validate_groups(config_data["groups"])

    _validate_preprocess(config_data.get("preprocess"))
    _validate_analysis_unit(config_data)
    _validate_grouping(config_data)
    _validate_partition_validations(config_data)
    _validate_comparisons(config_data)

    return config_data  # type: ignore[return-value]
