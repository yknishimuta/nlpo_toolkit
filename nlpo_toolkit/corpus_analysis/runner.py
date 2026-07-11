from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import analysis_pipeline, post_analysis, run_reporting, runtime
from .runner_types import RunnerDependencies, RunResult


def run(
    *,
    project_root: Path | None = None,
    script_dir: Path | None = None,
    config_path: Path,
    group_by_file: Optional[bool] = None,
    dependencies: RunnerDependencies,
    error_on_empty_group: bool = False,
    auto_single_cleaned: bool = False,
) -> RunResult:
    """Core runner. Dependencies are injectable for CLI and tests."""
    context = runtime.prepare_run_context(
        project_root=project_root,
        script_dir=script_dir,
        config_path=config_path,
        group_by_file=group_by_file,
        auto_single_cleaned=auto_single_cleaned,
        error_on_empty_group=error_on_empty_group,
        dependencies=dependencies,
    )
    analysis = analysis_pipeline.analyze_corpora(context, dependencies)
    partitions = post_analysis.execute_partition_validations(
        context=context,
        analysis=analysis,
    )
    comparisons = post_analysis.execute_group_comparisons(
        context=context,
        analysis=analysis,
    )
    report = run_reporting.write_run_report(
        context=context,
        analysis=analysis,
        partitions=partitions,
        comparisons=comparisons,
    )
    return run_reporting.build_run_result(
        context=context,
        analysis=analysis,
        partitions=partitions,
        report=report,
    )
