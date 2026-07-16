from __future__ import annotations

from pathlib import Path

from ..corpus import resolve_corpus_work_items
from ..ports import CorpusPreparationDependencies
from ..preprocessing import execute_preprocess
from .models import AnalysisPlan, ResolvedAnalysisPlan
from .validate import validate_count_group_references


def _resolve_analysis_inputs(
    plan: AnalysisPlan,
    *,
    cleaned_dir: Path | None,
) -> ResolvedAnalysisPlan:
    resolved = resolve_corpus_work_items(
        config=plan.config,
        project_root=plan.project_root,
        cleaned_dir=cleaned_dir,
        grouping_mode=plan.grouping_mode,
        error_on_empty_group=plan.error_on_empty_group,
    )
    return ResolvedAnalysisPlan(
        definition=plan,
        cleaned_dir=cleaned_dir,
        work_items=resolved.work_items,
        group_files=resolved.group_files,
    )


def inspect_analysis_plan(plan: AnalysisPlan) -> ResolvedAnalysisPlan:
    cleaned_dir = plan.cleaner_plan.output_path if plan.cleaner_plan is not None else None
    return _resolve_analysis_inputs(plan, cleaned_dir=cleaned_dir)


def prepare_analysis_plan(
    plan: AnalysisPlan,
    *,
    dependencies: CorpusPreparationDependencies,
) -> ResolvedAnalysisPlan:
    cleaned_dir = execute_preprocess(plan.cleaner_plan, dependencies=dependencies)
    return _resolve_analysis_inputs(plan, cleaned_dir=cleaned_dir)


def prepare_count_plan(
    plan: AnalysisPlan,
    *,
    dependencies: CorpusPreparationDependencies,
) -> ResolvedAnalysisPlan:
    resolved = prepare_analysis_plan(plan, dependencies=dependencies)
    validate_count_group_references(resolved)
    return resolved
