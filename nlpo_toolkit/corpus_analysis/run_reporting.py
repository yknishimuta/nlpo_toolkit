from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from nlpo_toolkit.backends import render_backend_info

from .outputs import build_run_meta, collect_runtime_environment, write_run_meta
from .runner_types import (
    AnalysisResults,
    ComparisonRunResult,
    PartitionRunResult,
    RunContext,
    RunResult,
    deduplicate_resolved_paths,
)
from .config_references import build_config_file_inventory


@dataclass(frozen=True)
class RunReport:
    summary_path: Path
    metadata_path: Path
    generated_outputs: tuple[Path, ...]


def _format_normalization_kv(norm: object) -> str:
    if hasattr(norm, "__dataclass_fields__"):
        norm_dict = asdict(norm)
    elif isinstance(norm, dict):
        norm_dict = norm
    else:
        return "(none)"

    keys_first = [
        "enabled",
        "unicode_nf",
        "normalize_ligatures",
        "strip_diacritics",
        "map_u_v",
        "map_i_j",
        "casefold",
    ]
    parts: list[str] = []

    for k in keys_first:
        if k in norm_dict:
            parts.append(f"{k}={norm_dict[k]}")

    for k in sorted(norm_dict.keys()):
        if k in keys_first:
            continue
        parts.append(f"{k}={norm_dict[k]}")

    return " ".join(parts)


def build_summary_lines(
    *,
    context: RunContext,
    analysis: AnalysisResults,
    partitions: PartitionRunResult,
    comparisons: ComparisonRunResult,
) -> list[str]:
    plan = context.plan
    lines: list[str] = [
        "# Summary",
        "",
        f"language: {plan.config.nlp.language}",
        f"stanza_package: {plan.config.nlp.stanza_package}",
        f"nlp_backend: {context.backend_info.name}",
        f"analysis_unit: {plan.analysis_unit}",
        f"normalization: {_format_normalization_kv(plan.config.normalization)}",
        "",
    ]
    lines.extend(render_backend_info(context.backend_info))
    lines.append("")

    if plan.config.ref_tags.enabled:
        for label, counts in analysis.ref_tags_by_group.items():
            lines.append(
                f"- group={label} ref_tag_types={len(counts)} ref_tag_tokens={sum(counts.values())}"
            )

    if analysis.token_artifacts:
        lines.extend(["", "# Token artifacts", ""])
        for artifact in analysis.token_artifacts:
            lines.append(
                f"- token_artifact={artifact.get('group', '')} "
                f"path={artifact.get('path', '')} "
                f"rows={artifact.get('row_count', 0)} "
                f"included={artifact.get('included_row_count', 0)} "
                f"schema={artifact.get('schema_version', '')}"
            )

    if analysis.analysis_cache:
        cache_meta = analysis.analysis_cache
        lines.extend(["", "# Analysis cache", ""])
        lines.append(
            "analysis_cache "
            f"enabled={cache_meta.get('enabled', False)} "
            f"hits={cache_meta.get('hits', 0)} "
            f"misses={cache_meta.get('misses', 0)} "
            f"records_read={cache_meta.get('records_read', 0)} "
            f"records_written={cache_meta.get('records_written', 0)}"
        )

    if partitions.results:
        lines.extend(["", "# Partition validation", ""])
        for spec, result in zip(plan.partition_specs, partitions.results):
            if result.exact_match:
                lines.append(
                    f"- name={result.name} status=OK whole={result.whole} "
                    f"parts={','.join(result.parts)} "
                    f"target_tokens={result.whole_target_tokens} "
                    f"parts_target_tokens={result.parts_target_tokens} "
                    f"mismatched_items={result.mismatched_items}"
                )
            else:
                status = "ERROR" if spec.on_mismatch == "error" else "WARN"
                lines.append(
                    f"- name={result.name} status={status} whole={result.whole} "
                    f"parts={','.join(result.parts)} "
                    f"target_tokens={result.whole_target_tokens} "
                    f"parts_target_tokens={result.parts_target_tokens} "
                    f"token_delta={result.token_delta} "
                    f"mismatched_items={result.mismatched_items}"
                )

    if comparisons.results:
        lines.extend(["", "# Group comparisons", ""])
        for result in comparisons.results:
            spec = result.spec
            lines.append(
                f"- name={spec.name} group_a={spec.group_a} group_b={spec.group_b} "
                f"analysis_unit={result.analysis_unit} "
                f"group_a_tokens={result.group_a_tokens} "
                f"group_b_tokens={result.group_b_tokens} "
                f"items={result.rows_after_filter} scale={spec.scale} "
                f"zero_correction={spec.zero_correction}"
            )
    return lines


def write_summary(path: Path, lines: Sequence[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def merge_generated_outputs(*groups: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    merged: list[Path] = []
    for group in groups:
        for path in group:
            resolved = Path(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            merged.append(resolved)
    return tuple(merged)


def build_final_run_metadata(
    *,
    context: RunContext,
    analysis: AnalysisResults,
    partitions: PartitionRunResult,
    comparisons: ComparisonRunResult,
    generated_outputs: Sequence[Path],
) -> dict[str, object]:
    plan = context.plan
    meta = build_run_meta(
        groups_files={
            label: [str(path) for path in files]
            for label, files in analysis.files_by_group.items()
        },
    )
    meta["analysis_unit"] = plan.analysis_unit
    meta["nlp"] = context.backend_info.to_dict()
    if plan.auto_mode:
        meta["grouping"] = {
            "mode": "auto_single_cleaned",
            "auto_group_name": plan.auto_group_name,
        }
    else:
        meta["grouping"] = {"mode": "per_file" if plan.per_file else "groups"}
    meta["environment"] = collect_runtime_environment(plan.project_root)

    norm_dict = asdict(plan.config.normalization)
    norm_canon = json.dumps(norm_dict, ensure_ascii=False, sort_keys=True)
    meta["normalization"] = norm_dict
    meta["normalization_hash_sha256"] = hashlib.sha256(norm_canon.encode("utf-8")).hexdigest()
    meta["partition_validations"] = list(partitions.metadata)
    meta["group_comparisons"] = list(comparisons.metadata)
    meta["trace"] = {
        "enabled": plan.config.trace.enabled,
        "files": {
            label: str(path.resolve())
            for label, path in analysis.trace_paths.items()
        },
    }
    meta["token_artifacts"] = list(analysis.token_artifacts)
    meta["analysis_cache"] = dict(analysis.analysis_cache or {})
    meta["generated_outputs"] = [str(path.resolve()) for path in generated_outputs]
    return meta


def write_run_report(
    *,
    context: RunContext,
    analysis: AnalysisResults,
    partitions: PartitionRunResult,
    comparisons: ComparisonRunResult,
) -> RunReport:
    summary_path = write_summary(
        context.plan.out_dir / "summary.txt",
        build_summary_lines(
            context=context,
            analysis=analysis,
            partitions=partitions,
            comparisons=comparisons,
        ),
    )
    run_meta_path = context.plan.out_dir / "run_meta.json"
    generated_outputs = merge_generated_outputs(
        analysis.generated_outputs,
        partitions.generated_outputs,
        comparisons.generated_outputs,
        (summary_path, run_meta_path),
    )
    write_run_meta(
        build_final_run_metadata(
            context=context,
            analysis=analysis,
            partitions=partitions,
            comparisons=comparisons,
            generated_outputs=generated_outputs,
        ),
        context.plan.out_dir,
    )
    return RunReport(
        summary_path=summary_path,
        metadata_path=run_meta_path,
        generated_outputs=generated_outputs,
    )


def build_run_result(
    *,
    context: RunContext,
    analysis: AnalysisResults,
    partitions: PartitionRunResult,
    report: RunReport,
) -> RunResult:
    plan = context.plan
    groups_files = {
        label: deduplicate_resolved_paths(files)
        for label, files in analysis.files_by_group.items()
    }
    used_files = deduplicate_resolved_paths(
        path for files in groups_files.values() for path in files
    )
    if plan.cleaned_dir is None:
        input_files = used_files
        cleaned_files: tuple[Path, ...] = ()
    else:
        input_files = deduplicate_resolved_paths(
            plan.cleaner_inspection.input_files
            if plan.cleaner_inspection is not None
            else ()
        )
        cleaned_root = plan.cleaned_dir.resolve()
        cleaned_files = tuple(
            path for path in used_files if path.is_relative_to(cleaned_root)
        )
    trace_files = deduplicate_resolved_paths(analysis.trace_paths.values())
    trace_set = set(trace_files)
    output_files = deduplicate_resolved_paths(
        path for path in report.generated_outputs if Path(path).resolve() not in trace_set
    )
    return RunResult(
        exit_code=partitions.exit_code,
        plan=plan,
        groups_files=groups_files,
        input_files=input_files,
        cleaned_files=cleaned_files,
        output_files=output_files,
        trace_files=trace_files,
        config_files=build_config_file_inventory(
            config=plan.config,
            config_path=plan.config_path,
            project_root=plan.project_root,
            cleaner_inspection=plan.cleaner_inspection,
        ),
        summary_path=report.summary_path.resolve(),
        metadata_path=report.metadata_path.resolve(),
        partition_mismatches=partitions.mismatches,
    )
