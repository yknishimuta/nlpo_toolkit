from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TypeVar


K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def freeze_mapping(values: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(values))


def freeze_tuple_mapping(
    values: Mapping[K, Iterable[T]],
) -> Mapping[K, tuple[T, ...]]:
    return MappingProxyType({key: tuple(items) for key, items in values.items()})


def freeze_count_mapping(values: Mapping[str, int]) -> Mapping[str, int]:
    frozen: dict[str, int] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise TypeError("count keys must be strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("count values must be integers")
        frozen[key] = value
    return MappingProxyType(frozen)

