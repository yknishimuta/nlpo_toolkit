from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


AnalysisRecordCacheStatus = Literal["hit", "miss", "disabled"]


@dataclass(frozen=True)
class AnalysisCacheGroupResult:
    group: str
    status: AnalysisRecordCacheStatus
    cache_key: str
    record_count: int


@dataclass(frozen=True)
class AnalysisCacheStatsSnapshot:
    enabled: bool
    directory: str
    hits: int
    misses: int
    objects_written: int
    records_read: int
    records_written: int
    groups: tuple[AnalysisCacheGroupResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", tuple(self.groups))
