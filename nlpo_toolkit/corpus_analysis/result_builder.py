from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .analysis_results import AnalysisResults
from .runner_types import PartitionRunResult, RunContext, RunResult


def _unique(paths: Iterable[Path]) -> tuple[Path, ...]:
    seen: set[Path] = set()
    result: list[Path] = []
    for raw in paths:
        path = Path(raw).resolve()
        if path not in seen:
            seen.add(path)
            result.append(path)
    return tuple(result)


def build_run_result(*, context: RunContext, analysis: AnalysisResults, partitions: PartitionRunResult) -> RunResult:
    plan = context.session.corpus.plan
    definition = plan.definition
    groups_files = {label: _unique(group.files) for label, group in analysis.groups.items()}
    used_files = _unique(path for files in groups_files.values() for path in files)
    if plan.cleaned_dir is None:
        input_files = used_files
        cleaned_files: tuple[Path, ...] = ()
    else:
        input_files = _unique(definition.cleaner_plan.inspection.input_files if definition.cleaner_plan is not None else ())
        cleaned_root = plan.cleaned_dir.resolve()
        cleaned_files = tuple(path for path in used_files if path.is_relative_to(cleaned_root))
    return RunResult(partitions.exit_code, plan, groups_files, input_files, cleaned_files, context.artifact_plan, definition.config_files.references, partitions.mismatches)
