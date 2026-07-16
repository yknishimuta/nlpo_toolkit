from __future__ import annotations

from typing import Mapping, TypeAlias
import math


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonMapping: TypeAlias = Mapping[str, JsonValue]

CsvScalar: TypeAlias = str | int | float | bool | None
CsvRow: TypeAlias = Mapping[str, CsvScalar]
MutableCsvRow: TypeAlias = dict[str, CsvScalar]

ConfigScalar: TypeAlias = JsonScalar
ConfigValue: TypeAlias = JsonValue
ConfigObject: TypeAlias = dict[str, ConfigValue]


class SerializationTypeError(ValueError):
    pass


def validate_json_value(raw: object, *, location: str = "$") -> JsonValue:
    if raw is None or isinstance(raw, (str, bool)):
        return raw
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if not math.isfinite(raw):
            raise SerializationTypeError(f"Non-finite JSON number at {location}")
        return raw
    if isinstance(raw, list):
        return [validate_json_value(value, location=f"{location}[{index}]")
                for index, value in enumerate(raw)]
    if isinstance(raw, dict):
        result: JsonObject = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise SerializationTypeError(f"Non-string JSON key at {location}")
            result[key] = validate_json_value(value, location=f"{location}.{key}")
        return result
    raise SerializationTypeError(
        f"Unsupported JSON value at {location}: {type(raw).__name__}"
    )
