from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

from .analysis_cache import AnalysisCacheRunStats


def _deduplicate_resolved_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


@dataclass(frozen=True)
class GroupAnalysisResult:
    files: tuple[Path, ...]
    counter: Counter[str]
    ref_tag_counts: Counter[str]
    output_files: tuple[Path, ...]
    trace_path: Path | None = None
    token_artifact: Mapping[str, object] | None = None


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

    @property
    def generated_outputs(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for group in self.groups.values():
            paths.extend(group.output_files)
            if group.trace_path is not None:
                paths.append(group.trace_path)
        return _deduplicate_resolved_paths(paths)

    @property
    def token_artifact_metadata(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            group.token_artifact
            for group in self.groups.values()
            if group.token_artifact is not None
        )
