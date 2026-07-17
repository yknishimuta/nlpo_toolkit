"""Production filesystem adapters for Count publication ports."""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

from ..analysis_records import TokenRecord
from ..diagnostic_trace import DiagnosticTraceWriter
from ..publication_models import (
    ComparisonArtifactPublication,
    GroupArtifactPublication,
    PartitionArtifactPublication,
    RecordArtifactPublicationRequest,
)
from ..publication_ports import RecordArtifactSession
from ..token_artifact.schema import TokenArtifactMetadata
from ..token_artifact.writer import TokenArtifactWriter
from .models import ArtifactKind
from .writers.comparison import (
    render_comparisons_json,
    write_comparison_csv_artifact,
    write_comparisons_json_artifact,
)
from .writers.group import write_group_artifacts
from .writers.partition import (
    render_partition_json,
    write_partition_csv_artifact,
    write_partition_json_artifact,
)


def publish_group_artifacts(request: GroupArtifactPublication) -> None:
    write_group_artifacts(
        artifact_plan=request.artifact_plan,
        group=request.group,
        counter=request.counter,
        dictionary=request.dictionary,
        reference_tag_counts=request.reference_tag_counts,
        csv_header=request.csv_header,
        reference_tags_enabled=request.reference_tags_enabled,
    )


def publish_partition_artifacts(request: PartitionArtifactPublication) -> None:
    csv_names: dict[str, str] = {}
    for spec, result in zip(request.specs, request.results):
        artifact = request.artifact_plan.require(
            ArtifactKind.PARTITION_VALIDATION_CSV, name=spec.name
        )
        csv_names[spec.name] = artifact.path.name
        write_partition_csv_artifact(artifact, result=result)
    if request.specs:
        artifact = request.artifact_plan.require(ArtifactKind.PARTITION_VALIDATION_JSON)
        write_partition_json_artifact(
            artifact,
            data=render_partition_json(request.specs, request.results, csv_names=csv_names),
        )


def publish_comparison_artifacts(request: ComparisonArtifactPublication) -> None:
    csv_names: dict[str, str] = {}
    for result in request.results:
        artifact = request.artifact_plan.require(
            ArtifactKind.GROUP_COMPARISON_CSV, name=result.spec.name
        )
        csv_names[result.spec.name] = artifact.path.name
        write_comparison_csv_artifact(artifact, result=result)
    if request.results:
        artifact = request.artifact_plan.require(ArtifactKind.GROUP_COMPARISONS_JSON)
        write_comparisons_json_artifact(
            artifact,
            data=render_comparisons_json(request.results, csv_names=csv_names),
        )


class _ProductionRecordArtifactSession:
    def __init__(
        self,
        token_writer: TokenArtifactWriter | None,
        trace_writer: DiagnosticTraceWriter | None,
    ) -> None:
        self._token_writer = token_writer
        self._trace_writer = trace_writer

    def write(self, record: TokenRecord) -> None:
        if self._token_writer is not None:
            self._token_writer.write(record)
        if self._trace_writer is not None:
            self._trace_writer.consider(record)

    @property
    def token_artifact_metadata(self) -> TokenArtifactMetadata | None:
        if self._token_writer is None:
            return None
        return self._token_writer.final_metadata


def _trace_temporary_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    return Path(raw)


@contextmanager
def open_record_artifact_session(
    request: RecordArtifactPublicationRequest,
) -> Iterator[RecordArtifactSession]:
    token_writer: TokenArtifactWriter | None = None
    trace_writer: DiagnosticTraceWriter | None = None
    trace_temporary: Path | None = None
    try:
        with ExitStack() as stack:
            if request.token_artifact is not None:
                if request.token_artifact_metadata is None:
                    raise ValueError("Token artifact metadata plan is required")
                token_writer = stack.enter_context(
                    TokenArtifactWriter(
                        request.token_artifact.path,
                        metadata_path=request.token_artifact_metadata.path,
                        descriptor=request.descriptor,
                    )
                )
            if request.diagnostic_trace is not None:
                trace_temporary = _trace_temporary_path(request.diagnostic_trace.path)
                trace_writer = stack.enter_context(
                    DiagnosticTraceWriter(
                        trace_temporary,
                        max_rows=request.trace_max_rows,
                        only_keys=request.trace_only_keys,
                        write_truncation_marker=request.trace_write_truncation_marker,
                    )
                )
            session = _ProductionRecordArtifactSession(token_writer, trace_writer)
            yield session
        if trace_temporary is not None and request.diagnostic_trace is not None:
            trace_temporary.replace(request.diagnostic_trace.path)
    finally:
        if trace_temporary is not None:
            trace_temporary.unlink(missing_ok=True)
