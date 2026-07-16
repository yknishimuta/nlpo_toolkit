from __future__ import annotations

from .models import AppConfig
from nlpo_toolkit.serialization.types import JsonObject


def config_to_dict(config: AppConfig) -> JsonObject:
    return config.to_external_dict()
