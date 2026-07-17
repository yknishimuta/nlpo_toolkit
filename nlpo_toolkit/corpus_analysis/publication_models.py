"""Typed values passed from Count application services to publication ports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from nlpo_toolkit.comparison.config import ComparisonSpec
from nlpo_toolkit.comparison.engine import FrequencyTable
from nlpo_toolkit.comparison.results import ConfiguredComparisonResult

from .artifacts.models import ArtifactPlan, PlannedArtifact
from .partition_models import PartitionSpec
from .partition_validation import PartitionResult
from .postprocessing.dictionary import DictionaryClassification
from .reporting.models import RunMetadata
from .token_artifact.schema import TokenArtifactDescriptor


ConfiguredResult = ConfiguredComparisonResult[ComparisonSpec, FrequencyTable]


@dataclass(frozen=True)
class GroupArtifactPublication:
    artifact_plan: ArtifactPlan
    group: str
    counter: Counter[str]
    dictionary: DictionaryClassification | None
    reference_tag_counts: Counter[str]
    csv_header: tuple[str, str]
    reference_tags_enabled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "counter", self.counter.copy())
        object.__setattr__(self, "reference_tag_counts", self.reference_tag_counts.copy())
        if self.dictionary is not None:
            object.__setattr__(
                self,
                "dictionary",
                DictionaryClassification(
                    known=self.dictionary.known.copy(),
                    unknown=self.dictionary.unknown.copy(),
                ),
            )


@dataclass(frozen=True)
class PartitionArtifactPublication:
    artifact_plan: ArtifactPlan
    specs: tuple[PartitionSpec, ...]
    results: tuple[PartitionResult, ...]


@dataclass(frozen=True)
class ComparisonArtifactPublication:
    artifact_plan: ArtifactPlan
    results: tuple[ConfiguredResult, ...]


@dataclass(frozen=True)
class RunReportPublication:
    artifact_plan: ArtifactPlan
    summary: str
    metadata: RunMetadata


@dataclass(frozen=True)
class RecordArtifactPublicationRequest:
    token_artifact: PlannedArtifact | None
    token_artifact_metadata: PlannedArtifact | None
    diagnostic_trace: PlannedArtifact | None
    descriptor: TokenArtifactDescriptor
    trace_max_rows: int
    trace_only_keys: tuple[str, ...]
    trace_write_truncation_marker: bool
