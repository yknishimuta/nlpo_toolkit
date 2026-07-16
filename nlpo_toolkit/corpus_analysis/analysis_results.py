from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from .analysis_cache.stats import AnalysisCacheRunStats
from .token_artifact.schema import TokenArtifactMetadata


@dataclass(frozen=True)
class GroupAnalysisResult:
    files: tuple[Path, ...]
    counter: Counter[str]
    ref_tag_counts: Counter[str]
    token_artifact: TokenArtifactMetadata | None = None


@dataclass(frozen=True)
class AnalysisResults:
    groups: Mapping[str, GroupAnalysisResult]
    cache_stats: AnalysisCacheRunStats

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    @classmethod
    def from_groups(
        cls,
        groups: Iterable[tuple[str, GroupAnalysisResult]],
        *,
        cache_stats: AnalysisCacheRunStats,
    ) -> AnalysisResults:
        indexed: dict[str, GroupAnalysisResult] = {}
        for label, result in groups:
            if label in indexed:
                raise ValueError(f"Duplicate group analysis result label: {label}")
            indexed[label] = result
        return cls(groups=indexed, cache_stats=cache_stats)
