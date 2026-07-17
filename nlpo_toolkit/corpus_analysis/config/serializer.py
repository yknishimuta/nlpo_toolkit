from __future__ import annotations

from .models import AppConfig
from nlpo_toolkit.serialization.types import JsonObject, validate_json_value


def config_to_dict(config: AppConfig) -> JsonObject:
    value = validate_json_value(config.model_dump(mode="json", by_alias=True))
    if not isinstance(value, dict):
        raise TypeError("AppConfig serialization must produce an object")
    if config.preprocess.kind is None:
        value.pop("preprocess", None)
    if config.csv_header is None:
        value.pop("csv_header", None)
    return value
