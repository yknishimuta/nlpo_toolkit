from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from nlpo_toolkit.serialization.types import JsonObject, JsonValue

from ..analysis_results import AnalysisResults
from ..artifacts.models import ArtifactKind
from ..runner_types import ComparisonRunResult, PartitionRunResult, RunContext
from .models import (
    AnalysisCacheGroupReport,
    AnalysisCacheReport,
    ComparisonReport,
    GeneratedArtifactReport,
    GroupingReport,
    PartitionReport,
    RunMetadata,
    RuntimeEnvironmentReport,
    TokenArtifactReport,
    TraceArtifactReport,
)


def build_run_metadata(*, context: RunContext, analysis: AnalysisResults, partitions: PartitionRunResult, comparisons: ComparisonRunResult, environment: RuntimeEnvironmentReport) -> RunMetadata:
    definition = context.session.corpus.plan.definition
    normalization = definition.config.normalization
    canonical = json.dumps(normalization.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    grouping = GroupingReport(
        mode="auto_single_cleaned" if definition.auto_mode else ("per_file" if definition.per_file else "groups"),
        auto_group_name=definition.config.grouping.auto_group_name if definition.auto_mode else None,
    )
    partition_reports = tuple(
        PartitionReport(result.name, result.whole, tuple(result.parts), result.exact_match, result.whole_target_tokens, result.parts_target_tokens, result.token_delta, result.whole_types, result.parts_union_types, result.mismatched_items, spec.on_mismatch)
        for spec, result in zip(definition.config.validations.partitions, partitions.validations)
    )
    comparison_reports = tuple(
        ComparisonReport(result.spec.name, result.spec.group_a, result.spec.group_b, result.spec.scale, result.spec.zero_correction, result.spec.min_total_count, result.analysis_unit, result.group_a_tokens, result.group_b_tokens, result.vocabulary_union_size, result.rows_after_filter, context.artifact_plan.require(ArtifactKind.GROUP_COMPARISON_CSV, name=result.spec.name).path.name)
        for result in comparisons.comparisons
    )
    traces = tuple(TraceArtifactReport(artifact.group or "", artifact.path) for artifact in context.artifact_plan.select(kinds={ArtifactKind.DIAGNOSTIC_TRACE}))
    tokens = []
    for group, result in analysis.groups.items():
        metadata = result.token_artifact
        if metadata is None:
            continue
        tokens.append(TokenArtifactReport(group, context.artifact_plan.require(ArtifactKind.TOKEN_ARTIFACT, group=group).path, context.artifact_plan.require(ArtifactKind.TOKEN_ARTIFACT_METADATA, group=group).path, metadata.schema_version, metadata.row_count, metadata.included_row_count, metadata.complete, metadata.sha256))
    generated = tuple(GeneratedArtifactReport(a.kind, a.path, a.group, a.name) for a in context.artifact_plan.artifacts)
    cache_stats = analysis.cache_stats
    cache = AnalysisCacheReport(
        enabled=cache_stats.enabled,
        directory=cache_stats.directory,
        hits=cache_stats.hits,
        misses=cache_stats.misses,
        objects_written=cache_stats.objects_written,
        records_read=cache_stats.records_read,
        records_written=cache_stats.records_written,
        groups=tuple(
            AnalysisCacheGroupReport(
                group.group, group.status, group.cache_key, group.record_count
            )
            for group in cache_stats.groups
        ),
    )
    return RunMetadata(
        generated_at=datetime.now(timezone.utc),
        groups_files={label: tuple(group.files) for label, group in analysis.groups.items()},
        analysis_unit=definition.analysis_mode.unit,
        nlp=context.session.backend.info,
        grouping=grouping,
        environment=environment,
        normalization=normalization,
        normalization_hash_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        partition_validations=partition_reports,
        group_comparisons=comparison_reports,
        traces=traces,
        token_artifacts=tuple(tokens),
        analysis_cache=cache,
        generated_artifacts=generated,
    )


def run_metadata_to_json_value(metadata: RunMetadata) -> JsonObject:
    environment = metadata.environment
    partitions: list[JsonValue] = [
        {"name": value.name, "whole": value.whole, "parts": list(value.parts), "exact_match": value.exact_match, "whole_target_tokens": value.whole_target_tokens, "parts_target_tokens": value.parts_target_tokens, "token_delta": value.token_delta, "whole_types": value.whole_types, "parts_union_types": value.parts_union_types, "mismatched_items": value.mismatched_items, "on_mismatch": value.on_mismatch}
        for value in metadata.partition_validations
    ]
    comparisons: list[JsonValue] = [
        {"name": value.name, "group_a": value.group_a, "group_b": value.group_b, "scale": value.scale, "zero_correction": value.zero_correction, "min_total_count": value.min_total_count, "group_a_tokens": value.group_a_tokens, "group_b_tokens": value.group_b_tokens, "vocabulary_union_size": value.vocabulary_union_size, "rows_after_filter": value.rows_after_filter, "csv": value.csv_name, "analysis_unit": value.analysis_unit}
        for value in metadata.group_comparisons
    ]
    return {
        "generated_at": metadata.generated_at.isoformat(),
        "groups_files": {group: [str(path) for path in paths] for group, paths in metadata.groups_files.items()},
        "analysis_unit": metadata.analysis_unit,
        "nlp": {"backend": metadata.nlp.name, "language": metadata.nlp.language,
                "package": metadata.nlp.package, "model": metadata.nlp.model,
                "device": metadata.nlp.device},
        "grouping": ({"mode": metadata.grouping.mode, "auto_group_name": metadata.grouping.auto_group_name} if metadata.grouping.auto_group_name is not None else {"mode": metadata.grouping.mode}),
        "environment": {"python_version": environment.python_version, "platform": environment.platform, "executable": str(environment.executable), "project_root": str(environment.project_root), "git_commit": environment.git_commit, "git_status": environment.git_status},
        "normalization": metadata.normalization.model_dump(mode="json"),
        "normalization_hash_sha256": metadata.normalization_hash_sha256,
        "partition_validations": partitions,
        "group_comparisons": comparisons,
        "trace": {"enabled": bool(metadata.traces), "files": {trace.group: str(trace.path) for trace in metadata.traces}},
        "token_artifacts": [{"group": token.group, "path": str(token.path), "metadata_path": str(token.metadata_path), "schema_version": token.schema_version, "row_count": token.row_count, "included_row_count": token.included_row_count, "complete": token.complete, "sha256": token.sha256} for token in metadata.token_artifacts],
        "analysis_cache": {
            "enabled": metadata.analysis_cache.enabled,
            "directory": metadata.analysis_cache.directory,
            "hits": metadata.analysis_cache.hits,
            "misses": metadata.analysis_cache.misses,
            "objects_written": metadata.analysis_cache.objects_written,
            "records_read": metadata.analysis_cache.records_read,
            "records_written": metadata.analysis_cache.records_written,
            "groups": [
                {
                    "group": group.group,
                    "status": group.status,
                    "cache_key": group.cache_key,
                    "record_count": group.record_count,
                }
                for group in metadata.analysis_cache.groups
            ],
        },
        "generated_outputs": [str(path) for path in metadata.generated_outputs],
    }
