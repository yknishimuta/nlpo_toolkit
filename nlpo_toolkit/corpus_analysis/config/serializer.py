from __future__ import annotations

from .models import AppConfig


def config_to_dict(config: AppConfig) -> dict[str, object]:
    return config.to_external_dict()
