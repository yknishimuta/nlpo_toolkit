from __future__ import annotations

from nlpo_toolkit.backends import render_backend_info

from ..analysis_results import AnalysisResults
from ..runner_types import ComparisonRunResult, PartitionRunResult, RunContext


def _format_normalization(normalization: object) -> str:
    values = normalization.model_dump(mode="python")
    preferred = ("enabled", "unicode_nf", "normalize_ligatures", "strip_diacritics", "map_u_v", "map_i_j", "casefold")
    keys = [key for key in preferred if key in values]
    keys.extend(sorted(key for key in values if key not in preferred))
    return " ".join(f"{key}={values[key]}" for key in keys)


def render_run_summary(*, context: RunContext, analysis: AnalysisResults, partitions: PartitionRunResult, comparisons: ComparisonRunResult) -> str:
    definition = context.session.corpus.plan.definition
    lines = ["# Summary", "", f"language: {definition.config.nlp.language}", f"stanza_package: {definition.config.nlp.stanza_package}", f"nlp_backend: {context.session.backend.info.name}", f"analysis_unit: {definition.analysis_mode.unit}", f"normalization: {_format_normalization(definition.config.normalization)}", ""]
    lines.extend(render_backend_info(context.session.backend.info))
    lines.append("")
    if definition.config.ref_tags.enabled:
        for label, group in analysis.groups.items():
            lines.append(f"- group={label} ref_tag_types={len(group.ref_tag_counts)} ref_tag_tokens={sum(group.ref_tag_counts.values())}")
    token_metadata = tuple(group.token_artifact for group in analysis.groups.values() if group.token_artifact is not None)
    if token_metadata:
        lines.extend(["", "# Token artifacts", ""])
        for metadata in token_metadata:
            lines.append(f"- token_artifact={metadata.group} path={metadata.artifact_path} rows={metadata.row_count} included={metadata.included_row_count} schema={metadata.schema_version}")
    cache = analysis.cache_stats.snapshot()
    if cache.enabled:
        lines.extend(["", "# Analysis cache", ""])
        lines.append(f"analysis_cache enabled={cache.enabled} hits={cache.hits} misses={cache.misses} records_read={cache.records_read} records_written={cache.records_written}")
    if partitions.validations:
        lines.extend(["", "# Partition validation", ""])
        for spec, result in zip(definition.config.validations.partitions, partitions.validations):
            status = "OK" if result.exact_match else ("ERROR" if spec.on_mismatch == "error" else "WARN")
            extra = "" if result.exact_match else f" token_delta={result.token_delta}"
            lines.append(f"- name={result.name} status={status} whole={result.whole} parts={','.join(result.parts)} target_tokens={result.whole_target_tokens} parts_target_tokens={result.parts_target_tokens}{extra} mismatched_items={result.mismatched_items}")
    if comparisons.comparisons:
        lines.extend(["", "# Group comparisons", ""])
        for result in comparisons.comparisons:
            spec = result.spec
            lines.append(f"- name={spec.name} group_a={spec.group_a} group_b={spec.group_b} analysis_unit={result.analysis_unit} group_a_tokens={result.group_a_tokens} group_b_tokens={result.group_b_tokens} items={result.rows_after_filter} scale={spec.scale} zero_correction={spec.zero_correction}")
    return "\n".join(lines) + "\n"
