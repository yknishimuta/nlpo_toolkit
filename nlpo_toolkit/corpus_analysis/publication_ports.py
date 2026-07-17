"""Publication contracts consumed by Count application services."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from .analysis_records import TokenRecord
from .publication_models import (
    ComparisonArtifactPublication,
    GroupArtifactPublication,
    PartitionArtifactPublication,
    RecordArtifactPublicationRequest,
    RunReportPublication,
)
from .token_artifact.schema import TokenArtifactMetadata


class GroupArtifactPublisher(Protocol):
    def __call__(self, request: GroupArtifactPublication) -> None: ...


class PartitionArtifactPublisher(Protocol):
    def __call__(self, request: PartitionArtifactPublication) -> None: ...


class ComparisonArtifactPublisher(Protocol):
    def __call__(self, request: ComparisonArtifactPublication) -> None: ...


class RunReportPublisher(Protocol):
    def __call__(self, request: RunReportPublication) -> None: ...


class RecordArtifactSession(Protocol):
    def write(self, record: TokenRecord) -> None: ...

    @property
    def token_artifact_metadata(self) -> TokenArtifactMetadata | None: ...


class RecordArtifactSessionFactory(Protocol):
    def __call__(
        self, request: RecordArtifactPublicationRequest
    ) -> AbstractContextManager[RecordArtifactSession]: ...


@dataclass(frozen=True)
class CountPublicationDependencies:
    group_artifacts: GroupArtifactPublisher
    partition_artifacts: PartitionArtifactPublisher
    comparison_artifacts: ComparisonArtifactPublisher
    run_report: RunReportPublisher
    record_artifacts: RecordArtifactSessionFactory

