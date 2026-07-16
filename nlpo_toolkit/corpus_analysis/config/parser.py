from __future__ import annotations

from pathlib import Path
from typing import Mapping

from pydantic import ValidationError
from nlpo_toolkit.configuration.yaml_loader import YamlLoadError, load_yaml_mapping

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
    try:
        raw = load_yaml_mapping(path)
    except YamlLoadError as exc:
        raise ConfigError(str(exc)) from exc
    return parse_config(raw)


def ensure_app_config(config: AppConfig | Mapping[str, object]) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    return parse_config(config)
