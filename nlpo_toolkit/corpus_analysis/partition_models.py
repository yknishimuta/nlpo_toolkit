from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionSpec:
    name: str
    whole: str
    parts: tuple[str, ...]
    on_mismatch: str
    report: str
