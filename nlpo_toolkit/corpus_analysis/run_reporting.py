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
    RunnerDependencies,
)


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
    dependencies: RunnerDependencies,
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
    if context.backend_info.name == "stanza":
        lines.extend(
            dependencies.render_stanza_package_table(
                context.nlp,
                plan.config.nlp.stanza_package,
            )
        )
    else:
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
        hash_inputs=False,
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
    dependencies: RunnerDependencies,
) -> RunReport:
    summary_path = write_summary(
        context.plan.out_dir / "summary.txt",
        build_summary_lines(
            context=context,
            analysis=analysis,
            partitions=partitions,
            comparisons=comparisons,
            dependencies=dependencies,
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
