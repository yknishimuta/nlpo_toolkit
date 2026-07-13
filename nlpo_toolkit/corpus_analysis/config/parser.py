from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml
from pydantic import ValidationError

from .models import AppConfig


class ConfigError(ValueError):
    """A configuration file could not be read, parsed, or validated."""


def _format_location(location: tuple[object, ...]) -> str:
    return ".".join(str(part) for part in location) or "config"


def format_validation_error(error: ValidationError) -> str:
    return "\n".join(
        f"{_format_location(item['loc'])}: {item['msg']}"
        for item in error.errors(include_url=False, include_context=False)
    )


def parse_config(raw: Mapping[str, object]) -> AppConfig:
    try:
        return AppConfig.model_validate(raw, by_alias=True, by_name=False)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc)) from exc


def load_config(path: Path) -> AppConfig:
    if path.suffix.lower() not in {".yml", ".yaml"}:
        raise ConfigError("Config file must be YAML (.yml / .yaml)")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Config not found or unreadable: {path}: {exc}") from exc
    except UnicodeError as exc:
        raise ConfigError(f"Config file is not valid UTF-8: {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError("Top-level YAML must be a mapping.")
    return parse_config(data)


def ensure_app_config(config: AppConfig | Mapping[str, object]) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    return parse_config(config)
