from __future__ import annotations

import math
from typing import Mapping


def as_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def optional_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if value is None:
        return {}
    return as_mapping(value, context=context)


def str_value(value: object, *, context: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def optional_str(value: object, *, context: str) -> str | None:
    if value is None:
        return None
    return str_value(value, context=context)


def bool_value(value: object, *, context: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be bool")
    return value


def int_value(value: object, *, context: str, default: int | None = None, minimum: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{context} is required")
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be an integer greater than or equal to {minimum}")
    return value


def optional_int(value: object, *, context: str, default: int | None = None, minimum: int | None = None) -> int | None:
    if value is None:
        return default
    return int_value(value, context=context, minimum=minimum)


def float_value(value: object, *, context: str, default: float, minimum_exclusive: float | None = None) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{context} must be a finite number")
    if minimum_exclusive is not None and out <= minimum_exclusive:
        raise ValueError(f"{context} must be greater than {minimum_exclusive}")
    return out


def string_tuple(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list of strings")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}[{index}] must be a string")
        out.append(item)
    return tuple(out)


def optional_string_set(value: object, *, context: str, default: frozenset[str]) -> frozenset[str]:
    if value is None:
        return default
    if isinstance(value, str) or not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    out: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"{context}[{index}] must be a string")
        if stripped := item.strip():
            out.add(stripped.upper())
    return frozenset(out)
