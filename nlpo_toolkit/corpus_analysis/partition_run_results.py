from __future__ import annotations

from dataclasses import dataclass

from .partition_validation import PartitionResult


@dataclass(frozen=True)
class PartitionMismatchSummary:
    name: str
    level: str
    token_delta: int
    mismatched_items: int


@dataclass(frozen=True)
class PartitionValidationRunResult:
    validations: tuple[PartitionResult, ...]
    exit_code: int
    mismatches: tuple[PartitionMismatchSummary, ...] = ()

